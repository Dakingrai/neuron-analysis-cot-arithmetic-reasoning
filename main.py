# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# torchrun --nproc_per_node 1 main.py --ckpt_dir ../../../downloads/huggingface/models/llama2-7b/ --tokenizer_path ../../../downloads/huggingface/models/llama2-7b/tokenizer.model --experiment algorithm_one --prompt equation_only
# module load gnu10 openmi

from typing import List

import fire

from llama import Llama
from experiments.neuron_discovery import algorithm_one, algorithm_two, prompt_gpt

def main(
    experiment: str,
    ckpt_dir: str = None,
    tokenizer_path: str = None,
    data_dir: str = None,
    results_dir: str = None,
    prompt: str = None,
    few_shot: bool = True,
    max_seq_len: int = 2500,
    max_gen_len: int = 300,
    max_batch_size: int = 6,
):
    if experiment == "algorithm_one":
        algorithm_one.main(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, data_dir = data_dir, prompt = prompt, max_seq_len=max_seq_len, max_gen_len=max_gen_len, max_batch_size=max_batch_size, results_dir = results_dir)
    elif experiment == "algorithm_two":
        algorithm_two.main(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, data_dir = data_dir, prompt = prompt, max_seq_len=max_seq_len, max_gen_len=max_gen_len, max_batch_size=max_batch_size, results_dir = results_dir)
    elif experiment == "prompt_gpt":
        prompt_gpt.main(data_dir = data_dir, results_dir = results_dir)
    else:
        raise ValueError("Invalid value for --experiment!")


if __name__ == "__main__":
    fire.Fire(main)