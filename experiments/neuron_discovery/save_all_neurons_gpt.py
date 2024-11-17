import os
import pdb
from collections import Counter
from tqdm import tqdm
import json

def load_data(file_path):
    return json.load(open(file_path))

def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent = 4, default=float)

def main(
    root_dir: str = "results/gpt-4",
    data_dir: str = "results/algorithm_two/cot_prompt",
    n_samples: int = 20,
    len_forward: int = 150,
    n_dominant_neurons: int = 20,
    overlap_threshold: float = 0.5, # Threshold for the overlap of dominant neurons across different time steps
    num_layers = 32
):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    neuron_names = [set() for _ in range(32)]
    all_neurons = [{} for _ in range(32)]
    for sample_idx in tqdm(range(n_samples)):
        for idx in range(len_forward):
            data = load_data(os.path.join(data_dir, f"{sample_idx}_{idx}.json"))
            layer_no = 0
            for key, value in data.items():
                for each in value.keys():
                    if each not in neuron_names[layer_no]:
                        all_neurons[layer_no][each] = value[each]
                neuron_names[layer_no].update(list(value.keys()))
                layer_no += 1
            
    save_data(all_neurons, os.path.join(root_dir, "all_neurons.json"))

