import json
import pdb
from tqdm import tqdm
import random
import re
import copy
import torch
import os
import sys
from scripts import utils
from typing import List
import fire
from scripts.Llama import Llama


def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent = 4)

def load_data(file_path):
    return json.load(open(file_path))

def read_data(file_path):
    raw_data = []
    with open(file_path, "r", encoding='utf-8') as fin:
        for line in fin:
            raw_data.append(json.loads(line))
    return raw_data

def extract_answers(data):
    answers = []
    for each in data:
        each_answer = int(each['answer'].split('\n#### ')[1].replace(",", ""))
        answers.append(each_answer)
    return answers

def extract_pred_answers(data, verbose=False):
    preds = []
    for each in data:
        each_pred = each.split('answer (arabic numerals) is ')[1].split('.\n')[0]
        each_pred = re.sub(r'[a-zA-Z%$=\-]', ' ', each_pred) # remove alphabets and insert whitespace
        each_pred = each_pred.replace(",", "") # remove commas with no space
        each_pred = ' '.join(each_pred.split()) # remove multiple spaces
        each_pred = each_pred.split()[-1]
        try:
            each_pred = int(float(each_pred))
        except:
            each_pred = 0
        if verbose:
            print(each)
            print(each_pred)
            print("----")
        preds.append(each_pred)
    return preds


def compute_accuracy(gold_data_path, pred_data_path, cot=False, save_path=None, verbose=True, sample=None):
    if sample:
        data1 = utils.read_data(gold_data_path)
        random.seed(sample)
        random.shuffle(data1)
        gold_data = data1[:20]
    else:
        gold_data = read_data(gold_data_path)

    gold_answers = extract_answers(gold_data)
    pred_data = load_data(pred_data_path)
    if cot:
        pred_answers = extract_cot_with_example_pred_answers(pred_data)
    else:
        pred_answers = extract_pred_answers(pred_data)
    correct = 0

    for i in range(len(gold_answers)):
        if gold_answers[i] == pred_answers[i]:
            correct += 1
            gold_data[i]['prediction'] = pred_data[i]
            gold_data[i]['correct'] = True
        else:
            gold_data[i]['prediction'] = pred_data[i]
            gold_data[i]['correct'] = False

    if verbose:
        print("Accuracy: ", correct/len(gold_answers))
        print("Total: ", len(gold_answers))
        print("Correct: ", correct)
        stat = {
            "correct": correct,
            "total": len(gold_answers),
            "accuracy": correct/len(gold_answers)
        }
        save_data(stat, 'results.txt')

    result_path = save_path.split('.')[0] + "_results.txt"
    if save_path:
        save_data(gold_data, save_path)
        save_data(stat, result_path)

def extract_cot_with_example_pred_answers(preds):
    refined_preds = []
    all_preds = []
    for pred in preds:
        try:
            pred1 = pred.split("The answer is ")[9]
        except:
            pred1 = "234234"
        all_preds.append(pred1)
        pred2 = pred1.split("\n\nQuestion:")[0]
        pred3 = re.sub(r'[a-zA-Z%$=\-\.]', ' ', pred2)
        pred4 = pred3.replace(",", "") # remove commas with no space
        pred5 = ' '.join(pred4.split()) # remove multiple spaces
        try:
            pred6 = pred5.split()[-1]
            pred7 = int(float(pred6))
        except:
            pred7 = 234234
        refined_preds.append(pred7)
    return refined_preds, all_preds


def compute_cot_with_example_accuracy(gold_data_path, pred_data_path, save_path=None, verbose=True, sample=None):
    if sample:
        pdb.set_trace()
        data1 = utils.read_data(gold_data_path)
        random.seed(sample)
        random.shuffle(data1)
        gold_data = data1[:20]
    else:
        gold_data = read_data(gold_data_path)
    gold_answers = extract_answers(gold_data)
    pred_data = load_data(pred_data_path)
    pred_answers, pred_all = extract_cot_with_example_pred_answers(pred_data)
    correct = 0
    for i in range(len(gold_answers)):
        if gold_answers[i] == pred_answers[i]:
            correct += 1
            gold_data[i]['prediction'] = pred_data[i]
            gold_data[i]['correct'] = True
            gold_data[i]['pred_answer'] = pred_answers[i]
            gold_data[i]['pred_only'] = pred_all[i]
        else:
            
            gold_data[i]['prediction'] = pred_data[i]
            gold_data[i]['correct'] = False
            gold_data[i]['pred_answer'] = pred_answers[i]
            gold_data[i]['pred_only'] = pred_all[i]

    if verbose:
        print("Accuracy: ", correct/len(gold_answers))
        print("Total: ", len(gold_answers))
        print("Correct: ", correct)
    stat = {
            "correct": correct,
            "total": len(gold_answers),
            "accuracy": correct/len(gold_answers)
        }
    result_path = save_path.split('.')[0] + "_results.txt"
    if save_path:
        save_data(gold_data, save_path)
        save_data(stat, result_path)


def extract_cot_pred_answers(generator, max_gen_len=300, data_path = "results/batch/test_cot_results.jsonl", save_path=None):
    data = load_data(data_path)
    clean_data = clean_cot_pred_answers(data)
    # Inference with batch size 6
    results = []
    for i in tqdm(range(0, len(clean_data), 6)):
        batch = clean_data[i:i+6]
        prompts = [each['cleaned'] for each in batch]
        preds = generator.inference(prompts, max_gen_len=max_gen_len)
        results.extend(preds)
    
    if save_path:
        save_data(results, save_path)

def clean_cot_pred_answers(data, save_path=None):
    clean_data = []
    for each in data:
        each_dict = {}
        each_dict['original'] = each
        each_dict['cleaned'] = each.split('\nQ:')[0].strip() + " Therefore, the answer (arabic numerals) is "
        clean_data.append(each_dict)
    if save_path:
        utils.save_data(clean_data, save_path)

    return clean_data
    
def run_inference(generator, data, prompt, type="cot", batch_size=6, max_gen_len=300, save_path=None, few_shot=False):
    results = []
    data_copy = copy.deepcopy(data)
    for i in tqdm(range(0, len(data_copy), batch_size)):
        batch = data_copy[i:i+batch_size]

        for i in range(len(batch)):
            batch[i]['question'] = ' '.join(batch[i]['question'].split()) # remove multiple spaces
            if few_shot:
                if type == "equation_only":
                    batch[i]['question'] = prompt + 'Question: '+ batch[i]['question'] + "\n\n"
                else:
                    batch[i]['question'] = prompt + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"
            
            else:
                batch[i]['question'] = 'Q: '+ batch[i]['question'] + " A: " + prompt
        
        final_prompts = [each['question'] for each in batch]

        preds = generator.inference(final_prompts, max_gen_len=max_gen_len)
        results.extend(preds)

    if save_path:
        save_data(results, save_path)

class MyLlama(Llama):
    def __init__(self, model_path, tokenizer_path, max_seq_len: int = 500, max_gen_len: int = 400, max_batch_size: int = 6, model_parallel_size=None) -> None:
        super().__init__(model_path, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    prompt: str,
    few_shot: False,
    max_seq_len: int = 4500,
    max_gen_len: int = 500,
    max_batch_size: int = 6,
    model_parallel_size = None,
    data_path = "data/test.jsonl",
    results_dir = "results/gsm8k_inference/"
    ):

    batch_size = max_batch_size

    results_dir = results_dir + "{model}"
    data = read_data(data_path)
    
    # Create root folder to save the results
    if not os.path.exists(results_dir):
        print("Creating directory: ", results_dir)
        os.makedirs(results_dir)
    
    # Read prompts from .txt file for few-shot prompts
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

    generator = MyLlama(ckpt_dir, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)

    # Create a new folder for the prompt_type 
    if not os.path.exists(f"{results_dir}/{prompt_type}"):
        print("Creating directory: ", f"{results_dir}/{prompt_type}")
        os.makedirs(f"{results_dir}/{prompt_type}")

    inference_save_path = f"{results_dir}/{prompt_type}/results.jsonl"
    extract_pred_path = f"{results_dir}/{prompt_type}/clean_results.jsonl"
    final_save_path = f"{results_dir}/{prompt_type}/final.json"
    print("Saving to: ", inference_save_path)
    run_inference(generator, data, batch_size=batch_size, prompt = clean_prompt, type=prompt_type, few_shot=few_shot, save_path=inference_save_path, max_gen_len=max_gen_len)
    if few_shot:
        compute_cot_with_example_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)
    else:
        extract_cot_pred_answers(generator, data_path = inference_save_path, save_path=extract_pred_path, max_gen_len=100)
        compute_accuracy(gold_data_path=data_path, pred_data_path=extract_pred_path, save_path=final_save_path)


if __name__ == "__main__":
    fire.Fire(main)

        
    