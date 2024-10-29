# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# torchrun --nproc_per_node 1 main.py --ckpt_dir ../../downloads/huggingface/models/Meta-Llama-3-8B/original/ --tokenizer_path ../../downloads/huggingface/models/Meta-Llama-3-8B/original/tokenizer.model
# module load gnu10 openmi

from typing import List

import fire

from llama import Llama
from experiments.neuron_discovery import algorithm_one, algorithm_two

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    experiment: str,
    data_dir: str,
    prompt: str,
    few_shot: bool = True,
    max_seq_len: int = 2500,
    max_gen_len: int = 300,
    max_batch_size: int = 6,
):
    if experiment == "algorithm_one":
        algorithm_one.main(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, data_dir = data_dir, prompt = prompt, max_seq_len=max_seq_len, max_gen_len=max_gen_len, max_batch_size=max_batch_size)
    elif experiment == "algorithm_two":
        algorithm_two.main(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, data_dir = data_dir, prompt = prompt, max_seq_len=max_seq_len, max_gen_len=max_gen_len, max_batch_size=max_batch_size)


if __name__ == "__main__":
    fire.Fire(main)