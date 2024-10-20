import fire
import os
import torch
import copy
import pdb
from scripts import utils
from collections import OrderedDict
import contextlib
import numpy as np
import torch.nn.functional as F
import random

class Llama:
    def __init__(
            self, 
            model_path, 
            tokenizer_path,
            max_seq_len: int = 500,
            max_gen_len: int = 400,
            max_batch_size: int = 6,
            model_parallel_size = None
            ) -> None:
        self.tokenizer = utils.load_tokenizer(tokenizer_path)
        self.model = utils.load_model(model_path, self.tokenizer, max_seq_len, max_batch_size, model_parallel_size)
        self.model.eval()

        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.max_batch_size = max_batch_size
        
    def update(self, max_seq_len, max_gen_len):
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        
    def inference(self, prompts, verbose=False, max_gen_len=200):
        inp_tokens, total_len, bsz, min_prompt_len, eos_reached, input_text_mask  = utils.prepare_input(prompts, self.tokenizer, self.model, bos_value=True, max_gen_len=max_gen_len)
        tokens = copy.deepcopy(inp_tokens)
        prev_pos = 0
        for cur_pos in range(min_prompt_len, total_len):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                    next_token == self.tokenizer.eos_id
                )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        try:
            tokens = [t[t != -1] for t in tokens]
            preds = [self.tokenizer.decode(each.tolist()) for each in tokens]
        except:
            print("Error in decoding")

        if verbose:
                print(preds)
                
        return preds

class StopForward(Exception):
        """
        If the only output needed from running a network is the retained
        submodule then Trace(submodule, stop=True) will stop execution
        immediately after the retained submodule by raising the StopForward()
        exception.  When Trace is used as context manager, it catches that
        exception and can be used as follows:

        with Trace(net, layername, stop=True) as tr:
            net(inp) # Only runs the network up to layername
        print(tr.output)
        """
        pass


class CorruptSingle:
    def __init__(self, 
                 model, 
                 layer, 
                 neuron, 
                 coeff_value=15, 
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.layer = layer
        self.verbose = False
        self.hook = None
        self.neuron = neuron
        self.coeff_value = coeff_value
        
        self.model.eval()
        def hook(module, input, output):
            if self.verbose:
                print("new coeff: " + str(self.coeff_value))
            num_tokens = output.shape[1]
            for i, each in enumerate(output[:, num_tokens-1, self.neuron]): # handling batches
                output[i, num_tokens-1, self.neuron] = self.coeff_value * output[i, num_tokens-1, self.neuron]
                
            # output[:, num_tokens-1, self.neuron] = self.coeff_value * 2
            # output[:, :, self.neuron] = self.coeff_value
            if stop:
                raise StopForward()
            return output
        self.hook1 = self.model.layers[self.layer].feed_forward.w3.register_forward_hook(hook)
        self.coeff_value = coeff_value
        self.hook2 = self.model.layers[self.layer].feed_forward.w1.register_forward_hook(hook)
        if self.verbose:
            print("Intervening on layer: " + str(self.layer) + " neuron: " + str(self.neuron) + " with coefficient: " + str(self.coeff_value))

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True
        
    def close(self):
        self.hook1.remove()
        self.hook2.remove()

class MultipleCorrupt(OrderedDict, contextlib.AbstractContextManager):
    def __init__(self, model, intervene, verbose = False) -> None:
        self.model = model
        self.layers = [each['layer'] for each in intervene]
        self.neurons = [each['neuron'] for each in intervene]
        self.coeff_values = [each['new_coeff'] for each in intervene]
        self.stop = False

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev
        for idx, (is_last, layer) in enumerate(flag_last_unseen(self.layers)):
            self[layer] = CorruptSingle(model, layer, self.neurons[idx], self.coeff_values[idx], stop = self.stop and is_last)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True
    
    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()


class InterveneSingle:
    def __init__(self, 
                 model, 
                 layer, 
                 neuron, 
                 coeff_value=15, 
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.layer = layer
        self.verbose = False
        self.hook = None
        self.neuron = neuron
        self.coeff_value = coeff_value

        self.model.eval()
        def hook(module, input, output):
            if self.verbose:
                print("new coeff: " + str(self.coeff_value))
            num_tokens = output.shape[1]
            for i, each in enumerate(output[:, num_tokens-1, self.neuron]): # handling batches
                output[i, num_tokens-1, self.neuron] = self.coeff_value * output[i, num_tokens-1, self.neuron]
                
            # output[:, num_tokens-1, self.neuron] = self.coeff_value * 2
            # output[:, :, self.neuron] = self.coeff_value
            if stop:
                raise StopForward()
            return output
        self.hook1 = self.model.layers[self.layer].feed_forward.w3.register_forward_hook(hook)
        self.coeff_value = coeff_value
        self.hook2 = self.model.layers[self.layer].feed_forward.w1.register_forward_hook(hook)
        if self.verbose:
            print("Intervening on layer: " + str(self.layer) + " neuron: " + str(self.neuron) + " with coefficient: " + str(self.coeff_value))

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True
        
    def close(self):
        self.hook1.remove()
        self.hook2.remove()

class MultipleIntervene(OrderedDict, contextlib.AbstractContextManager):
    def __init__(self, model, intervene, verbose = False) -> None:
        self.model = model
        self.layers = [each['layer'] for each in intervene]
        self.neurons = [each['neuron'] for each in intervene]
        self.coeff_values = [each['new_coeff'] for each in intervene]
        self.stop = False

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev
        for idx, (is_last, layer) in enumerate(flag_last_unseen(self.layers)):
            self[layer] = InterveneSingle(model, layer, self.neurons[idx], self.coeff_values[idx], stop = self.stop and is_last)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True
    
    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()

class Llama2(Llama):
    def __init__(self, model_path, tokenizer_path, max_seq_len: int = 500, max_gen_len: int = 400, max_batch_size: int = 4, model_parallel_size=None) -> None:
        super().__init__(model_path, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)

        self.n_layer = 32
        # Adding activations_ property in model class
        if not hasattr(self.model, "activations_"):
            setattr(self.model, "activations_", {})
    
    def save_activation_prev(self, name):
        final_layer = self.n_layer - 1
        def hook(module, input, output):
            if "mlp" in name or "attn" in name or "m_coef" in name:
                if "attn" in name:
                    num_tokens = output.shape[1]
                    self.model.activations_[name] = output[0][:, num_tokens - 1].detach()
                elif "mlp" in name:
                    num_tokens = output.shape[1]  # [num_tokens, 3072] for values;
                    self.model.activations_[name] = output[0][num_tokens - 1].detach()
                elif "m_coef" in name:
                    num_tokens = output.shape[1]  # (batch, sequence, hidden_state)
                    self.model.activations_[name] = input[0][:, num_tokens - 1].detach()
            elif "residual" in name or "embedding" in name:
                num_tokens = output.shape[1]
                self.model.activations_[name] = input[0][:,
                                                    num_tokens - 1].detach()
        return hook

    def save_activation(self, name):
        final_layer = self.n_layer - 1
        def hook(module, input, output):
            if "mlp" in name or "attn" in name or "m_coef" in name:
                if "attn" in name:
                    num_tokens = output.shape[1]
                    self.model.activations_[name] = output[0][:, num_tokens - 1].detach()
                elif "mlp" in name:
                    num_tokens = output.shape[1]  # [num_tokens, 3072] for values;
                    self.model.activations_[name] = output[0][num_tokens - 1].detach()
                elif "m_coef" in name:
                    num_tokens = output.shape[1]  # (batch, sequence, hidden_state)
                    self.model.activations_[name] = input[0][:, num_tokens - 1].detach()
                    # if neurons:
                    #     for each in neurons:
                    #         layer, neuron = each.split("N")
                    #         layer = int(layer[1:])
                    #         neuron = int(neuron)
                    #         curr_layer = int(name.split("_")[-1])
                            
                    #         if curr_layer == layer:
                    #             pass
                            
                
            elif "residual" in name or "embedding" in name:
                num_tokens = output.shape[1]
                self.model.activations_[name] = input[0][:,
                                                    num_tokens - 1].detach()
        return hook
    
    def init_hooks(self):
        for i in range(self.n_layer):
            # module = utils.get_module(self.model, "layers."+str(i)+".ffn_norm")
            # module.register_forward_hook(self.save_activation("layer_residual_" + str(i - 1)))
            self.model.layers[i].ffn_norm.register_forward_hook(self.save_activation("mlp_" + str(i)))
            self.model.layers[i].ffn_norm.register_forward_hook(self.save_activation("layer_residual_" + str(i)))
            self.model.layers[i].feed_forward.w1.register_forward_hook(self.save_activation("intermediate_residual_" + str(i)))
            self.model.layers[i].feed_forward.w2.register_forward_hook(self.save_activation("m_coef_" + str(i)))
            
    def extract_sub_updates(self, TOP_K=10, verbose=False, neurons=None):
        records = []
        top_coef_idx = []
        top_coef_vals = []
        sub_update_tok = []
        residual_preds_probs = []
        residual_preds_tokens = []
        layer_preds_probs = []
        layer_preds_tokens = []
        sent_to_hidden_states = self.model.activations_.copy()
        sent_to_preds = {}
        sent_to_preds["layer_resid_preds"] = self.model.layer_resid_preds
        sent_to_preds["intermed_residual_preds"] = self.model.intermed_residual_preds
        all_sub_updates = []
        test_i = 0
        for LAYER in range(self.n_layer):
            coefs_ = []
            m_coefs = sent_to_hidden_states["m_coef_" + str(LAYER)].squeeze(0).cpu().numpy() # step 1: extracts only layer wise m_coeff
            
            # res_vec = sent_to_hidden_states["layer_residual_" + str(LAYER)].squeeze(0).cpu().numpy()
            value_norms = torch.linalg.norm(self.model.layers[LAYER].feed_forward.w2.weight.data, dim=0).cpu() 

            scaled_coefs = np.absolute(m_coefs) * value_norms.numpy() # Step 2: Scales the coeffs

            for index, prob in enumerate(scaled_coefs):
                coefs_.append((index, prob))
            top_values = sorted(coefs_, key=lambda x: x[1], reverse=True)[:TOP_K] # sort the coeffs and only extract the top-10 coeffs

            c_idx, c_vals = zip(*top_values) # store the index and actual coeffs
            top_coef_idx.append(c_idx)
            top_coef_vals.append(c_vals)
            residual_p_probs, residual_p_tokens, residual_p_tokens_ids = zip(*sent_to_preds['intermed_residual_preds'][LAYER])
            residual_preds_probs.append(residual_p_probs)
            residual_preds_tokens.append(residual_p_tokens)
            layer_p_probs, layer_p_tokens, layer_p_tokens_ids = zip(*sent_to_preds['layer_resid_preds'][LAYER])
            layer_preds_probs.append(layer_p_probs)
            layer_preds_tokens.append(layer_p_tokens)
            sub_update = []
            for ci, idx in enumerate(c_idx):
                # pdb.set_trace()
                # logits = torch.matmul(self.model.output.weight, self.model.layers[LAYER].feed_forward.w2.weight.data[:, idx]*np.absolute(m_coefs[idx]).T) 
                logits = torch.matmul(self.model.output.weight, self.model.layers[LAYER].feed_forward.w2.weight.data[:, idx]) 
                
                try:
                    probs = F.softmax(logits.T, dim=-1)
                except:
                    pdb.set_trace()
                
                probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()
                probs_ = []
                for index, prob in enumerate(probs):
                    probs_.append((index, prob))
                top_k_id = sorted(probs_, key=lambda x: x[1], reverse=True)[:TOP_K]
                top_k = [(t[1].item(), self.tokenizer.decode(t[0]), t[0]) for t in top_k_id]
                # top_k = [self.tokenizer.decode(t[0]) for t in top_k_id]
                sub_update.append(top_k)
            
            sub_update_tok.append(sub_update)
            all_sub_updates.append(sub_update)
        
        i = 0
        # Rename top_coef_idx
        revised_top_coef_idx = []
        for i, each in enumerate(top_coef_idx):
            revised_top_coef_idx.append([])
            for j, each_ in enumerate(each):
                neuron_name = "L" +str(i)+ "N"+ str(top_coef_idx[i][j])
                neuron_value = str(top_coef_vals[i][j])
                neuron_promotes = sub_update_tok[i][j]
                temp = {}
                temp[neuron_name] = {}
                temp[neuron_name]["coeff"] = neuron_value
                temp[neuron_name]["promotes"] = neuron_promotes
                revised_top_coef_idx[i].append(temp)
    
        # Save list of neurons
        if neurons:
            # neurons_sub_updates = []
            # for idx, each_n in enumerate(neurons):
            #     layer, neuron = each_n.split("N")
            #     layer = int(layer[1:])
            #     neuron = int(neuron)

            #     m_coefs = sent_to_hidden_states["m_coef_" + str(layer)].squeeze(0).cpu().numpy() # step 1: extracts only layer wise m_coeff

            #     # logits = torch.matmul(self.model.output.weight, self.model.layers[layer].feed_forward.w2.weight.data[:, neuron]*np.absolute(m_coefs[neuron]).T)
            #     logits = torch.matmul(self.model.output.weight, self.model.layers[layer].feed_forward.w2.weight.data[:, neuron])

            #     probs = F.softmax(logits.T, dim=-1)
            #     probs_ = []
            #     for index, prob in enumerate(probs):
            #         probs_.append((index, prob))
            #     top_k_id = sorted(probs_, key=lambda x: x[1], reverse=True)[:TOP_K]
            #     top_k = [(t[1].item(), self.tokenizer.decode(t[0]), t[0]) for t in top_k_id]
            #     neurons_sub_updates.append(top_k)


            revised_neurons_sub_updates = {}
            for i, each_n in enumerate(neurons):
                layer, neuron = each_n.split("N")
                layer = int(layer[1:])
                neuron = int(neuron)
                neuron_name = each_n

                m_coefs = sent_to_hidden_states["m_coef_" + str(layer)].squeeze(0).cpu().numpy()
                value_norms = torch.linalg.norm(self.model.layers[layer].feed_forward.w2.weight.data, dim=0).cpu()
                scaled_coefs = np.absolute(m_coefs) * value_norms.numpy()
                neuron_value = str(scaled_coefs[neuron])
                unnormalized_neuron_value = str(m_coefs[neuron])

                # neuron_promotes = neurons_sub_updates[i]

                revised_neurons_sub_updates[neuron_name] = {}
                revised_neurons_sub_updates[neuron_name]["coeff"] = neuron_value
                revised_neurons_sub_updates[neuron_name]["unnormalized_coeff"] = unnormalized_neuron_value
                # revised_neurons_sub_updates[neuron_name]["promotes"] = neuron_promotes
                
            return revised_top_coef_idx, all_sub_updates, revised_neurons_sub_updates
        
        else:
            return revised_top_coef_idx, all_sub_updates

    def extract_mlp_updates(self, TOP_K=10, verbose=False):
        activations = self.model.activations_
        layer_residual_preds = []
        intermed_residual_preds = []
        layer_residual_preds_idx = []
        for layer in activations.keys():
            if "layer_residual" in layer or "intermediate_residual" in layer:
                normed = self.model.norm(self.model.activations_[layer])
                # pdb.set_trace()
                logits = torch.matmul(self.model.output.weight, normed.T)
                
                
                probs = F.softmax(logits.T[0], dim=-1)
                probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()
                assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs) - 1)) + layer
                probs_ = []

                for index, prob in enumerate(probs):
                    probs_.append((index, prob, logits[index].item()))
                top_k_idx = sorted(probs_, key=lambda x: x[1], reverse=True)[:TOP_K]
                
                top_k = [(t[1].item(), self.tokenizer.decode(t[0]), t[0]) for t in top_k_idx]
                
            if "layer_residual" in layer:
                layer_residual_preds.append(top_k)
                layer_residual_preds_idx.append(top_k_idx)
            elif "intermediate_residual" in layer:
                intermed_residual_preds.append(top_k)
            
            for attr in ["layer_resid_preds", "intermed_residual_preds"]:
                if not hasattr(self.model, attr):
                    setattr(self.model, attr, [])
            
            self.model.layer_resid_preds = layer_residual_preds
            self.model.intermed_residual_preds = intermed_residual_preds

        if verbose:
            print("Layer Residual Preds")
            for i in range(len(layer_residual_preds)):
                print("---------------")
                print("Layer: " + str(i))
                print(layer_residual_preds[i])
                print(layer_residual_preds_idx[i])

        return layer_residual_preds
    
    def extract_mlp_updates_multiple(self, TOP_K=10, verbose=False):
        activations = self.model.activations_
        layer_residual_preds = []
        intermed_residual_preds = []
        layer_residual_preds_idx = []
        for layer in activations.keys():
            if "layer_residual" in layer or "intermediate_residual" in layer:
                normed = self.model.norm(self.model.activations_[layer])
                # pdb.set_trace()
                logits = torch.matmul(self.model.output.weight, normed.T)
                
                probs = [[] for _ in range(logits.shape[1])]
                probs_ = [[] for _ in range(logits.shape[1])]
                top_k = [[] for _ in range(logits.shape[1])]
                top_k_idx = [[] for _ in range(logits.shape[1])]
                for batch_i in range(logits.shape[1]):
                    probs[batch_i] = F.softmax(logits.T[batch_i], dim=-1)
                    # pdb.set_trace()
                    probs[batch_i] = torch.reshape(probs[batch_i], (-1,)).detach().cpu().numpy()
                    assert np.abs(np.sum(probs[batch_i]) - 1) <= 0.01, str(np.abs(np.sum(probs[batch_i]) - 1)) + layer

                    for index, prob in enumerate(probs[batch_i]):
                        # pdb.set_trace()
                        probs_[batch_i].append((index, prob, logits[index][batch_i].item()))
                    top_k_idx[batch_i] = sorted(probs_[batch_i], key=lambda x: x[1], reverse=True)[:TOP_K]
                    top_k[batch_i] = [(t[1].item(), self.tokenizer.decode(t[0]), t[0]) for t in top_k_idx[batch_i]]
                # pdb.set_trace()
            if "layer_residual" in layer:
                layer_residual_preds.append(top_k)
                layer_residual_preds_idx.append(top_k_idx)
            elif "intermediate_residual" in layer:
                intermed_residual_preds.append(top_k)
            
            for attr in ["layer_resid_preds", "intermed_residual_preds"]:
                if not hasattr(self.model, attr):
                    setattr(self.model, attr, [])
            
            self.model.layer_resid_preds = layer_residual_preds
            self.model.intermed_residual_preds = intermed_residual_preds

        if verbose:
            print("Layer Residual Preds")
            for i in range(len(layer_residual_preds)):
                print("---------------")
                print("Layer: " + str(i))
                print(layer_residual_preds[i])
                print(layer_residual_preds_idx[i])

        return layer_residual_preds

    def init_activations(self, prompt, intervene=None, verbose=False, max_gen_len=1):
        self.init_hooks() # initializes all the hooks
        inp_tokens, total_len, bsz, min_prompt_len, eos_reached, input_text_mask = utils.prepare_input(prompt, self.tokenizer, self.model, bos_value=True, max_gen_len=max_gen_len)
        tokens = copy.deepcopy(inp_tokens)
        prev_pos = 0
        for cur_pos in range(min_prompt_len, total_len):
            if intervene:
                with MultipleIntervene(self.model, intervene):
                    with torch.no_grad():
                        logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
                    # intervene = None
            else:
                with torch.no_grad():
                    logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token
        if verbose:
            tokens = [t[t != -1] for t in tokens]
            preds = [self.tokenizer.decode(each.tolist()) for each in tokens]
            print(preds)
        return self.tokenizer.decode(tokens.tolist())
    
    def get_activations(self, prompt, intervene = None, verbose=False, TOP_K=10, save_path=None, max_gen_len=1, neurons=None):
        pred = self.init_activations(prompt, intervene, max_gen_len=max_gen_len)
        layer_residual_preds = self.extract_mlp_updates(TOP_K=TOP_K, verbose=False)
        if neurons:
            top_coefs, all_sub_updates, neuron_subupdates = self.extract_sub_updates(TOP_K=TOP_K, verbose=False, neurons=neurons)
        else:
            top_coefs, all_sub_updates = self.extract_sub_updates(TOP_K=TOP_K, verbose=False, neurons=neurons)
        # pdb.set_trace()
        if save_path:
             # prepare data to save
            data = {'pred': pred}
            for i, each in enumerate(layer_residual_preds):
                data['layer_'+ str(i)] = {}
                data['layer_'+ str(i)]['layer_residual_preds'] = layer_residual_preds[i]
                data['layer_'+ str(i)]['top_coef_idx'] = top_coefs[i]
                utils.save_data(data, save_path)
            # pdb.set_trace()
            if neurons:
                save_path = save_path.replace(".json", "_neuron_subupdates.json")
                utils.save_data(neuron_subupdates, save_path)
            # for each in neuron_subupdates:
            #     data['neuron_'+ str(each)] = {}
            #     data['neuron_'+ str(each)]['neuron_subupdates'] = neuron_subupdates[each]
            #     utils.save_data(data, save_path)
        
        return data
    

    def get_activations_with_interventions(self, prompt, intervene = None, verbose=False, TOP_K=10, save_path=None, max_gen_len=1, neurons=None):
        pred = self.init_activations(prompt, intervene, max_gen_len=max_gen_len)
        layer_residual_preds = self.extract_mlp_updates(TOP_K=TOP_K, verbose=False)
        if neurons:
            top_coefs, all_sub_updates, neuron_subupdates = self.extract_sub_updates(TOP_K=TOP_K, verbose=False, neurons=neurons, intervene=intervene)
        else:
            top_coefs, all_sub_updates = self.extract_sub_updates(TOP_K=TOP_K, verbose=False, neurons=neurons, intervene=intervene)
        
        if save_path:
             # prepare data to save
            data = {'pred': pred}
            for i, each in enumerate(layer_residual_preds):
                data['layer_'+ str(i)] = {}
                data['layer_'+ str(i)]['layer_residual_preds'] = layer_residual_preds[i]
                data['layer_'+ str(i)]['top_coef_idx'] = top_coefs[i]
                utils.save_data(data, save_path)

            if neurons:
                save_path = save_path.replace(".json", "_neuron_subupdates.json")
                utils.save_data(neuron_subupdates, save_path)
        
        return data
    
    def get_neuron_activations(self, prompt, intervene = None, verbose=False, TOP_K=10, save_path=None, max_gen_len=1):
        pred = self.init_activations(prompt, intervene, max_gen_len=max_gen_len)
        layer_residual_preds = self.extract_mlp_updates(TOP_K=TOP_K, verbose=False)
        top_coefs, all_sub_updates = self.extract_sub_updates(TOP_K=TOP_K, verbose=False)
        if save_path:
             # prepare data to save
            data = {'pred': pred}
            for i, each in enumerate(layer_residual_preds):
                data['layer_'+ str(i)] = {}
                data['layer_'+ str(i)]['layer_residual_preds'] = layer_residual_preds[i]
                data['layer_'+ str(i)]['top_coef_idx'] = top_coefs[i]
                # pdb.set_trace()
                utils.save_data(data, save_path)
        return data
    
    def inference(self, prompts, intervene = None, verbose=False, max_gen_len=200):
        inp_tokens, total_len, bsz, min_prompt_len, eos_reached, input_text_mask  = utils.prepare_input(prompts, self.tokenizer, self.model, bos_value=True, max_gen_len=max_gen_len)
        tokens = copy.deepcopy(inp_tokens)
        prev_pos = 0
        counter = 0
        for cur_pos in range(min_prompt_len, total_len):
            if intervene:
                with MultipleIntervene(self.model, intervene):
                    with torch.no_grad():
                        logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            else:
                with torch.no_grad():
                    logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # intervene = None #test
            next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                    next_token == self.tokenizer.eos_id
                )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        try:
            preds = [self.tokenizer.decode(each.tolist()) for each in tokens]
        except:
            preds = []
            for bat in tokens:
                t_idx = int((tokens[0] == 2).nonzero(as_tuple=True)[0][0])
                preds.append(self.tokenizer.decode(bat[:t_idx].tolist()))

        if verbose:
                print(preds)
                
        return preds

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 500,
    max_gen_len: int = 500,
    max_batch_size: int = 6,
    model_parallel_size = None
):
    """
    Each experiments are maintained in their own scripts. Main.py is used to demonstrate following functionalities:
    1. Load a model (Llama) and tokenizer
    2. Run inference on the Llama
    """
    generator = Llama(ckpt_dir, 
                      tokenizer_path, 
                      max_seq_len, 
                      max_gen_len, 
                      max_batch_size, 
                      model_parallel_size)
    prompts = ["Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? A: Let's think step by step."]
    generator.inference(prompts, verbose=True)


if __name__ == '__main__':
    fire.Fire(main)