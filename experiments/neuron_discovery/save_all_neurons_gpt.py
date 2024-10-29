import os
import pdb
import argparse
from collections import Counter
from tqdm import tqdm
import json

def load_data(file_path):
    return json.load(open(file_path))

def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent = 4, default=float)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="results/gpt-4")
    parser.add_argument("--data_dir", type=str, default="results/algorithm_two/test/my_cot_prompts")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--len_forward", type=int, default=149)
    parser.add_argument("--n_dominant_neurons", type=int, default=20)
    parser.add_argument("--overlap_threshold", type=float, default=0.5, help="Threshold for the overlap of dominant neurons across different time steps")
    args = parser.parse_args()
    num_layers = 32

    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    
    neuron_names = [set() for _ in range(32)]
    all_neurons = [{} for _ in range(32)]
    for sample_idx in tqdm(range(args.n_samples)):
        for idx in range(args.len_forward):
            data = load_data(os.path.join(args.data_dir, f"{sample_idx}_{idx}.json"))
            layer_no = 0
            for key, value in data.items():
                for each in value.keys():
                    if each not in neuron_names[layer_no]:
                        all_neurons[layer_no][each] = value[each]
                neuron_names[layer_no].update(list(value.keys()))
                layer_no += 1
            
    save_data(all_neurons, os.path.join(args.root_dir, "all_neurons.json"))

