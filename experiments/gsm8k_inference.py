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
# from experiments import prompts
from data import prompts
from scripts.Llama import Llama

# from experiments.prompts import my_cot_prompts_equation_only, my_cot_prompts_equation_only_prior



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

# def extract_cot_with_example_pred_answers(data, verbose=False):
#     preds = []
#     for each in data:
#         each_pred = each.split('The answer is ')[1].split('.\n')[0]
#         pdb.set_trace()
#         each_pred = re.sub(r'[a-zA-Z%$=\-]', ' ', each_pred) # remove alphabets and insert whitespace
#         each_pred = each_pred.replace(",", "") # remove commas with no space
#         each_pred = ' '.join(each_pred.split()) # remove multiple spaces
#         each_pred = each_pred.split()[-1]
#         try:
#             each_pred = int(float(each_pred))
#         except:
#             each_pred = 0
#         if verbose:
#             print(each)
#             print(each_pred)
#             print("----")
#         preds.append(each_pred)
#     return preds

def compute_accuracy(gold_data_path, pred_data_path, cot=False, save_path=None, verbose=True, sample=None):
    if sample:
        data1 = utils.read_data(gold_data_path)
        random.seed(sample)
        random.shuffle(data1)
        gold_data = data1[:20]
    else:
        gold_data = read_data(gold_data_path)
    # gold_data = read_data(gold_data_path)
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

def extract_cot_with_understanding(preds):
    refined_preds = []
    invalid = 0
    for pred in preds:
        try:
            pred1 = pred.split("The answer is ")[9]
        except:
            invalid += 1
            pred1 = "234234"
        pred2 = pred1.split("\n\n\Q:")[0]
        pred3 = re.sub(r'[a-zA-Z%$=\-\.#]', ' ', pred2)
        pred4 = pred3.replace(",", "") # remove commas with no space
        pred5 = ' '.join(pred4.split()) # remove multiple spaces
        try:
            pred6 = pred5.split()[-1]
            pred7 = int(float(pred6))
        except:
            invalid += 1
            pred7 = 234234
        refined_preds.append(pred7)
    print("Invalid: ", invalid)
    print()
    return refined_preds

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

def compute_cot_with_understanding_cot(gold_data_path, pred_data_path, save_path=None, verbose=True):
    gold_data = read_data(gold_data_path)
    gold_answers = extract_answers(gold_data)
    pred_data = load_data(pred_data_path)
    pred_answers = extract_cot_with_understanding(pred_data)
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
    
def run_inference(generator, data, type="cot", batch_size=8, max_gen_len=300, save_path=None, few_shot=True):
    # inference in batch size of 6
    results = []
    data_copy = copy.deepcopy(data)
    for i in tqdm(range(0, len(data_copy), batch_size)):
        batch = data_copy[i:i+batch_size]
        if few_shot:
            prompt = prompts.type
            pdb.set_trace()
            clean_prompt = ' \n '.join(prompt)
            
        for i in range(len(batch)):
            batch[i]['question'] = ' '.join(batch[i]['question'].split()) # remove multiple spaces
            batch[i]['question'] = clean_prompt + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "cot_prompts":
            #     # from CoT Hub
            #     batch[i]['question'] = prompts.cot_prompts + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "cot_one":
            #     batch[i]['question'] = 'Q: '+ batch[i]['question'] + " A: Take a deep breath and work on this problem step-by-step."
            
            # elif type == "cot_two":
            #     batch[i]['question'] = 'Q: '+ batch[i]['question'] + " A: Break this down."
            
            # elif type == "cot_three":
            #     batch[i]['question'] = 'Q: '+ batch[i]['question'] + " A: A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem."
            
            # elif type == "cot":
            #     batch[i]['question'] = 'Q: '+ batch[i]['question'] + " A: Let's think step by step."

            # elif type == "my_cot_prompts":
            #     batch[i]['question'] = prompts.my_cot_prompts + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "my_cot_prompts_without_step":
            #     batch[i]['question'] = my_cot_prompts_without_step + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "my_cot_prompts_equation_only":
            #     batch[i]['question'] = my_cot_prompts_equation_only + 'Question: '+ batch[i]['question'] + "\n"
            
            # elif type == "my_cot_prompts_equation_only_prior":
            #     batch[i]['question'] = my_cot_prompts_equation_only_prior + 'Question: '+ batch[i]['question'] + "\n"
            
            # elif type == "my_cot_prompts_only_add_sub":
            #     batch[i]['question'] = my_cot_prompts_only_add_sub + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "my_cot_prompts_mult_only":
            #     batch[i]['question'] = my_cot_prompts_mult_only + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "my_cot_prompts_mult_only1":
            #     batch[i]['question'] = my_cot_prompts_mult_only1 + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"
            
            # elif type == "my_cot_prompts_add_only":
            #     batch[i]['question'] = my_cot_prompts_add_only + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"
            
            # elif type == "my_cot_prompts_mix":
            #     batch[i]['question'] = my_cot_prompts_mix + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "my_cot_prompts_mask1_complementary":
            #     batch[i]['question'] = my_cot_prompts_mask1_complementary + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "my_cot_prompts_mask2_complementary":
            #     batch[i]['question'] = my_cot_prompts_mask2_complementary + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"
            
            # elif type == "my_cot_prompts_incorrect_complementary":
            #     batch[i]['question'] = my_cot_prompts_incorrect_complementary + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"
            
            # elif type == "my_cot_prompts_wrong_equation":
            #     batch[i]['question'] = my_cot_prompts_wrong_equation + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"
            
            # elif type == "my_cot_prompts_wrong_label":
            #     batch[i]['question'] = my_cot_prompts_wrong_label + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "my_cot_prompts_wrong_both":
            #     batch[i]['question'] = my_cot_prompts_wrong_both + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"
            
            # elif type == "no_coherence_prompts_towards":
            #     batch[i]['question'] = no_coherence_prompts_towards + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "no_relevance_prompts_towards":
            #     batch[i]['question'] = no_relevance_prompts_towards + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"
                
            # elif type == "no_num_relevance_prompts_towards":
            #     batch[i]['question'] = no_num_relevance_prompts_towards + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "no_num_coherence_prompts_towards":
            #     batch[i]['question'] = no_num_coherence_prompts_towards + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"
            
            # elif type == "no_lang_coherence_prompts_towards":
            #     batch[i]['question'] = no_lang_coherence_prompts_towards + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type == "no_lang_relevance_prompts_towards":
            #     batch[i]['question'] = no_lang_relevance_prompts_towards + 'Question: '+ batch[i]['question'] + "\nLet's think step by step\n"

            # elif type =="my_std_prompts":
            #     batch[i]['question'] = my_std_prompts + 'Question: '+ batch[i]['question'] + "\nThe answer is "
            
            # elif type == "normal1":
            #     batch[i]['question'] = 'Q: '+ batch[i]['question'] + " A: The answer (arabic numerals) is "
            
            # elif type == "normal2":
            #     batch[i]['question'] = 'Q: '+ batch[i]['question'] + " A: "
            
            # # elif type == "standard_prompt_understanding_cot":
            # #     batch[i]['question'] = standard_prompt_understanding_cot + 'Q: '+ batch[i]['question'] + "\nA: "
            
            # elif type == "normal3":
            #     batch[i]['question'] = 'Q: '+ batch[i]['question'] + " A: First "
            
            # elif type == "normal3_originally":
            #     batch[i]['question'] = 'Q: '+ batch[i]['question'] + " A: Originally "
        
            # else:
            #     assert False, "Invalid type"
        
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
    max_seq_len: int = 2500,
    max_gen_len: int = 500,
    max_batch_size: int = 2,
    model_parallel_size = None
    ):
    print("Running gsm8k_inference.py...")
    # Configs
    model = "7b"
    if model == "13b":
        batch_size = 2
    else:
        batch_size = 2
    data_path = "data/test.jsonl"
    root_dir = f"results/gsm8k_inference/{model}"
    use_test = True

    type = ['my_cot_prompts']
    max_seq_len = 4500
    max_gen_len = 500
    
    # Create directories
    if not os.path.exists(root_dir):
        print("Creating directory: ", root_dir)
        os.makedirs(root_dir)
    
    if use_test:
        root_dir = f"{root_dir}/test"
    else:
        root_dir = f"{root_dir}/train"
    
    if not os.path.exists(root_dir):
        print("Creating directory: ", root_dir)
        os.makedirs(root_dir)

    data = read_data(data_path)
    # pdb.set_trace()
    generator = MyLlama(ckpt_dir, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)

    
    for each in type:
        print("Running: ", each)
        if not os.path.exists(f"{root_dir}/{each}"):
            print("Creating directory: ", f"{root_dir}/{each}")
            os.makedirs(f"{root_dir}/{each}")
        inference_save_path = f"{root_dir}/{each}/results.jsonl"
        extract_pred_path = f"{root_dir}/{each}/clean_results.jsonl"
        final_save_path = f"{root_dir}/{each}/final.json"
        print("Saving to: ", inference_save_path)
        run_inference(generator, data, batch_size=batch_size, type=each, save_path=inference_save_path, max_gen_len=max_gen_len)
        
        if each == "normal1":
            compute_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)

        elif each == "cot_prompts":
            compute_cot_with_example_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)

        elif each == "my_std_prompts":
            compute_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)
        
        elif each == "my_cot_prompts":
            compute_cot_with_example_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)
        
        elif "my_cot" in each:
            if each == "standard_prompt_understanding_cot":
                 compute_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)
            else:
                compute_cot_with_example_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)
        
        elif "towards" in each:
            compute_cot_with_example_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)

        elif "complementary" in each:
            compute_cot_with_example_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)
        
        elif each == "my_cot_prompts_revised1":
            compute_cot_with_example_accuracy(gold_data_path=data_path, pred_data_path=inference_save_path, save_path=final_save_path)

        elif each == "cot_one":
            extract_cot_pred_answers(generator, data_path = inference_save_path, save_path=extract_pred_path, max_gen_len=100)
            compute_accuracy(gold_data_path=data_path, pred_data_path=extract_pred_path, save_path=final_save_path)
        
        elif each == "cot_two":
            extract_cot_pred_answers(generator, data_path = inference_save_path, save_path=extract_pred_path, max_gen_len=100)
            compute_accuracy(gold_data_path=data_path, pred_data_path=extract_pred_path, save_path=final_save_path)
        
        elif each == "cot_three":
            extract_cot_pred_answers(generator, data_path = inference_save_path, save_path=extract_pred_path, max_gen_len=100)
            compute_accuracy(gold_data_path=data_path, pred_data_path=extract_pred_path, save_path=final_save_path)
        
        else:
            extract_cot_pred_answers(generator, data_path = inference_save_path, save_path=extract_pred_path, max_gen_len=100)
            compute_accuracy(gold_data_path=data_path, pred_data_path=extract_pred_path, save_path=final_save_path)


if __name__ == "__main__":
    fire.Fire(main)

        
    