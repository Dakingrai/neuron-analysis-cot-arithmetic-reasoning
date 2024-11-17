import pdb
import random
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 
import numpy as np
import torch
import torch.nn.functional as F
from statistics import mean
from scripts import utils
from scripts.Llama_refined import Llama3
from scripts.Llama_refined import MultipleIntervene
import copy
import fire


class MyLlama(Llama3):
    def __init__(self, model_path, tokenizer_path, max_seq_len: int = 500, max_gen_len: int = 400, max_batch_size: int = 4, model_parallel_size=None) -> None:
        super().__init__(model_path, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)
    
    def extract_sub_updates(self, TOP_K=20, verbose=False, neurons=None):
        top_coef_idx = []
        top_coef_vals = []
        sent_to_hidden_states = self.model.activations_.copy()
        for LAYER in range(self.n_layer):
            m_coefs = sent_to_hidden_states["m_coef_" + str(LAYER)].squeeze(0).cpu() # step 1: extracts only layer wise m_coeff
            
            value_norms = torch.linalg.norm(self.model.layers[LAYER].feed_forward.w2.weight.data, dim=0).cpu()

            scaled_coefs = np.absolute(m_coefs) * value_norms # Step 2: Scales the coeffs
            top_values = torch.topk(scaled_coefs.cuda(), TOP_K)
            c_idx = top_values.indices
            c_vals = top_values.values
            top_coef_idx.append(c_idx.tolist())
            top_coef_vals.append(c_vals.tolist()) 
        
        results = {'top_coef_idx': top_coef_idx, 'top_coef_vals': top_coef_vals}
        return results

    def algorithm_one(self, prompt, TOP_K=20, save_path=None, max_gen_len=1):
        pred_tokens = self.init_activations(prompt, max_gen_len=max_gen_len)
        results = self.extract_sub_updates(TOP_K=TOP_K, verbose=False)

        if save_path:
            utils.save_data(results, save_path)

        
def save_activations(generator, data, prompt, prompt_type, verbose=False, TOP_K=20, max_gen_len=1, save_path=None, few_shot = True):
    for i, each in tqdm(enumerate(data)):
        if few_shot:
            if prompt_type == "equation_only":
                each = prompt + 'Question: '+ each + "\n\n"
            else:
                each = prompt + 'Question: '+ each + "\nLet's think step by step\n"
        
        else:
            each = "Q: " + each + " A: Let's think step by step."
        
        save_path_ = save_path + f"_{i}.json"
        _ = generator.algorithm_one([each], TOP_K=TOP_K, save_path=save_path_, max_gen_len=max_gen_len)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    data_dir: str,
    prompt: str,
    few_shot: bool = True,
    max_seq_len: int = 2500,
    max_gen_len: int = 200,
    max_batch_size: int = 6,
    model_parallel_size = None,
    verbose = True, 
    results_dir = "results/algorithm_one",
):
    print("**************************")
    
    # load the data
    pdb.set_trace()
    data = utils.load_data(data_dir)
    n_samples = 20
    seed = 42
    random.seed(seed)
    all_samples = utils.sample_data(data, sample_correct=True, n=n_samples, seed=seed)


    # load prompt
    if few_shot:
        if '.txt' not in prompt:
            raise SystemExit("Provide path to your prompt file (.txt) containing few-shot demostration!")
        else:
            prompt1 = utils.read_file(prompt)
            clean_prompt = '\n'.join(prompt1) + '\n\n'
            prompt_type = prompt.split("/")[-1].replace(".txt", "")
    else:
        prompt_type = prompt.split()[0] + "_" + prompt.split()[-1] # Use the first and last word of the single-shot for prompt_type, which is used to create directories for saving the results!
        clean_prompt = prompt
    
    # create directory for saving results
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_dir = f"{results_dir}/{prompt_type}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # load model
    generator = MyLlama(ckpt_dir, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)

    # save activations
    TOP_K = 20
    max_gen_len = 150
    save_activations(generator, all_samples, prompt = clean_prompt, prompt_type=prompt_type, TOP_K=TOP_K, max_gen_len=max_gen_len, save_path=f"{results_dir}/cot_all", few_shot = few_shot)

if __name__ == "__main__":
    fire.Fire(main) 
    # torchrun --nproc_per_node 1 main.py --ckpt_dir ../../../downloads/huggingface/models/llama2-7b/ --tokenizer_path ../../../downloads/huggingface/models/llama2-7b/tokenizer.model --experiment algorithm_one --prompt data/prompts/equation_only.txt --data_dir results/gsm8k_inference/text_only/final.json
    
    