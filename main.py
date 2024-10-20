# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# torchrun --nproc_per_node 1 main.py --ckpt_dir ../../downloads/huggingface/models/Meta-Llama-3-8B/original/ --tokenizer_path ../../downloads/huggingface/models/Meta-Llama-3-8B/original/tokenizer.model
# module load gnu10 openmi
from typing import List

import fire

from llama import Llama
from experiments import gsm8k_inference

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2500,
    max_gen_len: int = 300,
    max_batch_size: int = 2,
):
    gsm8k_inference.main(ckpt_dir, tokenizer_path, max_seq_len, max_gen_len, max_batch_size)

    
    # """
    # Examples to run with the pre-trained models (no fine-tuning). Prompts are
    # usually in the form of an incomplete text prefix that the model can then try to complete.

    # The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    # `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    # """
#     generator = Llama.build(
#         ckpt_dir=ckpt_dir,
#         tokenizer_path=tokenizer_path,
#         max_seq_len=max_seq_len,
#         max_batch_size=max_batch_size,
#     )

#     prompts: List[str] = [
#         # For these prompts, the expected answer is the natural continuation of the prompt
#         "I believe the meaning of life is",
#         # Few shot prompt (providing a few examples before asking model to complete more);
#         """Translate English to French:

#         sea otter => loutre de mer
#         peppermint => menthe poivrée
#         plush girafe => girafe peluche
#         cheese =>""",
#     ]
#     results = generator.text_completion(
#         prompts,
#         max_gen_len=max_gen_len,
#         temperature=temperature,
#         top_p=top_p,
#     )
#     for prompt, result in zip(prompts, results):
#         print(prompt)
#         print(f"> {result['generation']}")
#         print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)