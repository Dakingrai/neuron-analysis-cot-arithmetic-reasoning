import json
import pdb
from tqdm import tqdm
import random
import re
import copy
import torch
import os
from collections import OrderedDict
import contextlib
import numpy

from scripts import utils
from experiments import prompts, gsm8k_inference
from scripts.Llama import Llama

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
            neuron = self.neuron
            prng = numpy.random.RandomState(1)
            num_tokens = output.shape[1]
            # module.weight.data[:, neuron] = module.weight.data[:, neuron] + torch.from_numpy(prng.normal(-1, 1, module.weight.data[:, neuron].shape[0])).to(module.weight.data.device)
            module.weight.data[:, neuron] = module.weight.data[:, neuron] 
            
            output1 = torch.matmul(input[0], module.weight.T)
            try:
                assert torch.allclose(output, output1)
            except:
                print("Error")
                print("-------------------")
            if stop:
                raise StopForward()
            output = output1
            return output
        self.hook1 = self.model.layers[self.layer].feed_forward.w2.register_forward_hook(hook)
        # self.coeff_value = coeff_value
        # self.hook2 = self.model.layers[self.layer].feed_forward.w1.register_forward_hook(hook)
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
        # self.hook2.remove()

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



class MyLlama(Llama):
    def __init__(self, model_path, tokenizer_path, max_seq_len: int = 500, max_gen_len: int = 400, max_batch_size: int = 6, model_parallel_size=None) -> None:
        super().__init__(model_path, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)
    
    def inference(self, prompts, interve_neurons = None, verbose=False, max_gen_len=200):
        inp_tokens, total_len, bsz, min_prompt_len, eos_reached, input_text_mask  = utils.prepare_input(prompts, self.tokenizer, self.model, bos_value=True, max_gen_len=max_gen_len)
        tokens = copy.deepcopy(inp_tokens)
        prev_pos = 0
        counter = 0
        for cur_pos in range(min_prompt_len, total_len):
            if interve_neurons:
                with MultipleCorrupt(self.model, interve_neurons):
                    with torch.no_grad():
                        logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
                    if counter > 4:
                        intervene = None
                    counter += 1
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


def get_intervene_dict(intervene, new_coeff):
    intervene_dict = []
    for idx, each in enumerate(intervene):
        each_dict = {}
        each = each.split("N")
        each_dict['layer'] = int(each[0][1:])
        each_dict['neuron'] = int(each[1])
        if type(new_coeff) == list:
            each_dict['new_coeff'] = new_coeff[idx]
        else:
            each_dict['new_coeff'] = new_coeff
        intervene_dict.append(each_dict)

    return intervene_dict

def run_inference(generator, data, type="cot", batch_size=8, max_gen_len=300, interve_neurons=None, save_path=None):
    # inference in batch size of 6
    results = []
    data_copy = copy.deepcopy(data)
    for i in tqdm(range(0, len(data_copy), batch_size)):
        batch = data_copy[i:i+batch_size]
        for i in range(len(batch)):
            batch[i]['question'] = ' '.join(batch[i]['question'].split()) # remove multiple spaces
            if type == "cot":
                batch[i]['question'] = 'Q: '+ batch[i]['question'] + " A: Let's think step by step."

            elif type == "my_cot_prompts":
                batch[i]['question'] = prompts.my_cot_prompts + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            else:
                assert False, "Invalid type"
        
        final_prompts = [each['question'] for each in batch]

        preds = generator.inference(final_prompts, max_gen_len=max_gen_len, interve_neurons=interve_neurons)
        results.extend(preds)

    if save_path:
        utils.save_data(results, save_path)

def load_neurons(neurons_path="results/neurons/"):
    concept_neurons = ['connections', 'add', 'subtract', 'multiply', 'division', 'equals_to', 'formula']
    neuron_names = []
    for each in concept_neurons:
         neurons = utils.load_data(f"results/neurons/neurons_{each}_all_150.json")
         neuron_names.extend(neurons)
    return neuron_names
    
def get_random_neurons(num_neurons=10):
    # random list of integers from 0 to 31
    random.seed(42)
    layers_pop = [i for i in range(32)]
    layers = [random.choice(layers_pop) for _ in range(num_neurons)]
    neurons = random.sample(range(0, 1024), num_neurons)
    neuron_names = []
    for layer, neuron in zip(layers, neurons):
        neuron_names.append(f"L{layer}N{neuron}")
    return neuron_names

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 4500,
    max_gen_len: int = 500,
    max_batch_size: int = 16,
    model_parallel_size = None
    ):
    print("Running ablation study...")
    # Configs
    batch_size = 8
    data_path = "data/test.jsonl"
    root_dir = f"results/ablation_study"
    use_test = True

    max_seq_len = 4500
    max_gen_len = 500
    
    # Create directories
    if not os.path.exists(root_dir):
        print("Creating directory: ", root_dir)
        os.makedirs(root_dir)

    # neurons = load_neurons()
    random_neurons = get_random_neurons(num_neurons=10)
    use_random_neurons = True
    if use_random_neurons:
        interve_neurons = get_intervene_dict(random_neurons, 1)
        result_path = f"{root_dir}/random_neurons"
    else:
        interve_neurons = get_intervene_dict(neurons, 1)
        result_path = f"{root_dir}/neurons"
    if not os.path.exists(result_path):
        print("Creating directory: ", result_path)
        os.makedirs(result_path)

    data = utils.read_data(data_path)
    generator = MyLlama(ckpt_dir, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)
    

    inference_save_path = f"{root_dir}/results.jsonl"
    extract_pred_path = f"{root_dir}/clean_results.jsonl"
    final_save_path = f"{root_dir}/final.json"
    print("Saving to: ", inference_save_path)
    run_inference(generator, data, batch_size=batch_size, type='my_cot_prompts', save_path=inference_save_path, max_gen_len=max_gen_len, interve_neurons=interve_neurons)
    
    gsm8k_inference.compute_cot_with_example_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)
    
    