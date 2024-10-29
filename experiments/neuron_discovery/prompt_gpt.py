import os
import openai
import pdb
import argparse
from tqdm import tqdm
import json

def load_data(file_path):
    return json.load(open(file_path))

def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent = 4, default=float)

def get_concept_token(concept):
  if concept == 'connections':
    tokens = ['first', 'originally', 'initially', 'so', 'meaning', 'therefore', 'then', 'next', 'hence'] + ['now']
    message = "Is this neuron promoting logical order or logical connections required for reasoning?"
  
  elif concept == 'add':
    tokens = ['add', 'addition', '+', 'sum', 'plus']
    message = "Is this neuron promoting arithmetic addition ?"
  
  elif concept == 'subtract':
    tokens = ['subtract', '-', 'minus', 'sub']
    message = "Is this neuron promoting arithmetic subtraction?"
  
  elif concept == 'multiply':
    tokens = ['mult', 'multiply', 'product', 'times', '*', 'x']
    message = "Is this neuron promoting arithmetic multiplication?"
  
  elif concept == 'division':
    tokens = ['divide', 'division', '/', '%', 'div']
    message = "Is this neuron promoting arithmetic division?"
  
  elif concept == 'equals_to':
    tokens = ['=', 'total', 'equals', 'equal', 'equivalent']
    message = "Is this neuron promoting equals to from equation?"
  
  elif concept == 'formula':
    tokens = ['formula', 'equation', 'calculation', 'algorithm', 'expression', 'computation']
    message = "Is this neuron promoting concept related to formula or calculation?"
    
  else:
    raise ValueError("Invalid type")
  
  return tokens, message

def filter_neurons(data, concept_token, threshold=1):
  found_neurons = []
  for layer_idx, layer in enumerate(data):
    if layer_idx < 6:
      continue
    layer_names = list(layer.keys())
    for each in layer_names:
      each_promotes = [e[0].strip() for e in data[layer_idx][each]['promotes']]
      count = 0
      for t in each_promotes:
        if t.lower() in concept_token:
          count += 1
          if count > threshold:
            found_neurons.append((each, each_promotes))
            break
  return found_neurons

def previous_results():
  prev_results = {}
  save_concepts = ['connections', 'add', 'subtract', 'multiply', 'division', 'equals_to', 'formula']
  for concept in save_concepts:
    save_path = f'results/promptgptv2/results_{concept}_all_150.jsonl'
    if os.path.exists(save_path):
      data = load_data(save_path)
      for each in data:
        prev_results[each['name']] = each
  return prev_results

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--concept", type=str, default='formula') # 54, 21, 7, 6, 8, 8, 10   
  parser.add_argument("--threshold", type=int, default=1)
  args = parser.parse_args()

  root_dir = "results/prompt-gpt"
  if not os.path.exists(root_dir):
        print("Creating directory: ", root_dir)
        os.makedirs(root_dir)
  
  concept_token, concept_message = get_concept_token(args.concept)
  data = load_data('results/gpt-4/all_neurons.json')

  openai.api_type = "azure"
  openai.api_base = "https://openai-access-west-us-azure.openai.azure.com/"
  openai.api_version = "2023-07-01-preview"
  openai.api_key = os.getenv("OPENAI_API_KEY")
  
  filtered_neurons = filter_neurons(data, concept_token)
  prev_results = previous_results()
  prev_results_names = list(prev_results.keys())
  count = 0
  results = []

  for name, promotes in tqdm(filtered_neurons):
    if name in prev_results_names:
      results.append(prev_results[name])
      count += 1
      continue

    count += 1
    promotes_str = ' ,'.join(promotes)
    message_role = f"A neuron in language model promotes the following set of words: {promotes_str}. {concept_message} First, answer in Yes or No format and provide an explanation."
    message_text = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": message_role}
    ]

    temp = {}
    temp['name'] = name
    temp['promotes'] = promotes_str
    temp['message'] = message_role
    try:
      completion = openai.ChatCompletion.create(
          engine="gpt-4-turbo", 
          messages = message_text, 
          temperature=0,  
          max_tokens=50,  
          top_p=1, 
          frequency_penalty=0,  
          presence_penalty=0,  
          stop=None)
      response = completion['choices'][0]['message']['content']
    except:
      print("Error in response")
      print(message_text)
      response = "No response"
    temp['gpt_response'] = response
    results.append(temp)
    print()
    print("--------------")
    print(name)
    print(message_role)
    print()
    print(response)
  print(f"Number of gpt prompt: {count}")
  save_path = f'{root_dir}/results_{args.concept}_all.jsonl'
  save_data(results, save_path)

  data = load_data(save_path)
  count = 0
  neurons = []
  for each in data:
    if 'yes' in each['gpt_response'].split(',')[0].lower():
      count +=1
      neurons.append(each['name'])
  print(count)
  save_data(neurons, f'{root_dir}/neurons_{args.concept}_all.json')


  
