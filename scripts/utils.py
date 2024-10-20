from llama.model import ModelArgs, Transformer
import torch
from llama.tokenizer import Tokenizer
import json, pdb, random
from pathlib import Path
import os
import sys
import time
import ast
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from matplotlib import pyplot as plt

def save_file(data, file_path):
    with open(file_path, 'w') as f:
        for each in data:
            f.write(each + "\n")
            
def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent = 4, default=float)

def load_data(file_path):
    return json.load(open(file_path))

def read_data(file_path):
    raw_data = []
    with open(file_path, "r", encoding='utf-8') as fin:
        for line in fin:
            raw_data.append(json.loads(line))
    return raw_data

def read_file(file_path):
    raw_data = []
    with open(file_path, 'r') as file:
        for line in file:
            raw_data.append(line.strip())
    return raw_data

def load_model(ckpt_dir, tokenizer, max_seq_len: int = 128, max_batch_size: int = 4, model_parallel_size=None):
    if not torch.distributed.is_initialized():
        # torch.distributed.init_process_group("gloo")
        torch.distributed.init_process_group("nccl")
    
    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # seed must be the same in all processes
    torch.manual_seed(1)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    assert model_parallel_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
    ckpt_path = checkpoints[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    # tokenizer.add_special_tokens('[ADD_MASK]')
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = Tokenizer(model_path=tokenizer_path)
    return tokenizer

def prepare_input(prompts, tokenizer, model, bos_value=False, eos_value=False, max_gen_len=1):
    prompt_tokens = [tokenizer.encode(x, bos=bos_value, eos=eos_value) for x in prompts]
    params = model.params
    bsz = len(prompt_tokens)
    try:
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
    except:
        pdb.set_trace()
    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    try:
        assert max_prompt_len <= params.max_seq_len
    except:
        print(max_prompt_len, params.max_seq_len)
        pdb.set_trace()
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    
    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id

    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id
    return tokens, total_len, bsz, min_prompt_len, eos_reached, input_text_mask


def sample_data(data, sample_all = False, sample_correct = True, n=50, seed=42, inference=False):
    random.seed(seed)
    # Get correct predictions only from data if sample_correct is True
    preds = []
    if sample_all:
        for each in data:
            if inference:
                preds.append(each['prediction'])
            else:
                each['question'] = ' '.join(each['question'].split()) # remove multiple spaces
                preds.append(each['question'])
    
    else:
        for each in data:
            if sample_correct:
                if each["correct"]:
                    if inference:
                        preds.append(each['prediction'])
                    else:
                        each['question'] = ' '.join(each['question'].split()) # remove multiple spaces
                        preds.append(each['question'])
               
            else:
                if each["correct"]==False:
                    if inference:
                        preds.append(each['prediction'])
                    else:
                        each['question'] = ' '.join(each['question'].split()) # remove multiple spaces
                        preds.append(each['question'])
                    
    random.shuffle(preds)
    return preds[:n]