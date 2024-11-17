import os
import openai
import pdb
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


def save_all_neurons_gpt(
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


def main(
    results_dir: str = "results/prompt-gpt",
    data_dir: str = "results/algorithm_two/cot_prompt",
):

  neurons_dir = f"{results_dir}/all_neurons.json"
  print("hello")
  save_all_neurons_gpt(root_dir = neurons_dir, data_dir=data_dir)
  pdb.set_trace()
  if not os.path.exists(results_dir):
        print("Creating directory: ", results_dir)
        os.makedirs(results_dir)

  openai.api_type = "[PLEASE SET]" # "Azure"
  openai.api_base = "[PLEASE SET]"
  openai.api_version = "[PLEASE SET]"
  openai.api_key = os.getenv("OPENAI_API_KEY")
  
  data = load_data(neurons_dir)

  concept_list = ['connections', 'add', 'subtract', 'multiply', 'division', 'equals_to', 'formula']
  for concept in concept_list:
    threshold = 1,
    concept_token, concept_message = get_concept_token(concept)

    filtered_neurons = filter_neurons(data, concept_token)
    count = 0
    results = []

    for name, promotes in tqdm(filtered_neurons):

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
    save_path = f'{root_dir}/results_{concept}_all.jsonl'
    save_data(results, save_path)

    data = load_data(save_path)
    count = 0
    neurons = []
    for each in data:
      if 'yes' in each['gpt_response'].split(',')[0].lower():
        count +=1
        neurons.append(each['name'])
    print(count)
    save_data(neurons, f'{root_dir}/neurons_{concept}_all.json')


  
