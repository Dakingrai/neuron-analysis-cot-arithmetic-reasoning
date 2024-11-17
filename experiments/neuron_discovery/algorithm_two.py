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
import time

class MyLlama(Llama3):
    def __init__(self, model_path, tokenizer_path, max_seq_len: int = 500, max_gen_len: int = 400, max_batch_size: int = 4, model_parallel_size=None) -> None:
        super().__init__(model_path, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)
    
    
    def algorithm_two(self, all_coef_idx, TOP_K=20):
        all_proj = {}
        start_time = time.time()
        for i in range(32):
            w2 = self.model.layers[i].feed_forward.w2.weight
            logits = torch.matmul(self.model.output.weight, w2)
            logits = logits.T
            probs = F.softmax(logits, dim=-1)
            for j in all_coef_idx[i]:
                top_proj_idx = torch.topk(probs[j], TOP_K)
                # get the top k tokens using tokenizer from the top_proj_idx.indices
                top_proj = [((self.tokenizer.decode([int(t)])), int(t), float(v)) for t, v in zip(top_proj_idx.indices, top_proj_idx.values)]
                all_proj[f"L{i}N{j}"] = top_proj #[3813]

        print(f"Time taken: {time.time() - start_time}")
        
        return all_proj

    # increase the speed of the

def save_activations(generator, data, prompt, prompt_type, verbose=False, TOP_K=20, max_gen_len=1, save_path=None):
    all_coef_idx = [set() for _ in range(32)]
    # all_coef_idx_list = [[] for _ in range(32)]
    for i, each in tqdm(enumerate(data)):
        saved_path = f"results/algorithm_one/{prompt_type}/cot_all_{i}.json"
        saved_activation_idx = utils.load_data(saved_path)
        
        for j in range(32):
            for k in saved_activation_idx['top_coef_idx'][j]:
                all_coef_idx[j].update(k)
                # all_coef_idx_list[j].extend(k)
    
    print(f"Length of all_coef_idx: {len(all_coef_idx[0])}")
    # print(f"Length of all_coef_idx_list: {len(all_coef_idx_list[0])}")
    proj_results = generator.algorithm_two(all_coef_idx)

    start_time = time.time()
    for example_idx, each in tqdm(enumerate(data)):
        saved_path = f"results/algorithm_one/{prompt_type}/cot_all_{example_idx}.json"
        saved_activation_idx = utils.load_data(saved_path)
        each_proj_results = {}
        
        for decoding_idx in range(len(saved_activation_idx['top_coef_idx'][0])):
            for layer_idx in range(32):
                each_proj_results['layer_'+ str(layer_idx)] = {}
                for ni, neuron_idx in enumerate(saved_activation_idx['top_coef_idx'][layer_idx][decoding_idx]):
                    each_proj_results['layer_'+ str(layer_idx)][f'L{layer_idx}N{neuron_idx}'] = {}

                    each_proj_results['layer_'+ str(layer_idx)][f'L{layer_idx}N{neuron_idx}']['coeffs'] = saved_activation_idx['top_coef_vals'][layer_idx][decoding_idx][ni]

                    each_proj_results['layer_'+ str(layer_idx)][f'L{layer_idx}N{neuron_idx}']['promotes'] = proj_results[f"L{layer_idx}N{neuron_idx}"]
                
            utils.save_data(each_proj_results, f"{save_path}/{example_idx}_{decoding_idx}.json")
    print(f"Time taken: {time.time() - start_time}")


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
    results_dir = "results/algorithm_two",
):

    # load the data
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
    
    # load the data
    data = utils.load_data(data_dir)
    n_samples = 20
    seed = 42
    random.seed(seed)
    all_samples = utils.sample_data(data, sample_all=True, n=n_samples, seed=seed)

    # load model
    generator = MyLlama(ckpt_dir, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)

    # save activations
    TOP_K = 20
    max_gen_len = 150
    save_activations(generator, all_samples, prompt = clean_prompt, prompt_type=prompt_type, TOP_K=TOP_K, max_gen_len=max_gen_len, save_path=f"{results_dir}")


if __name__ == "__main__":
    fire.Fire(main)
    # torchrun --nproc_per_node 1 main.py --ckpt_dir ../../../downloads/huggingface/models/llama2-7b/ --tokenizer_path ../../../downloads/huggingface/models/llama2-7b/tokenizer.model --experiment algorithm_two --prompt data/prompts/equation_only.txt --data_dir results/gsm8k_inference/equation_only/final.json