# neuron-analysis-cot-arithmetic-reasoning

This repository contains the code for ACL 2024 paper: [An Investigation of Neuron Activation as a Unified Lens to Explain Chain-of-Thought Eliciting Arithmetic Reasoning of LLMs](https://arxiv.org/abs/2406.12288).

**How to Cite:** If you find our survey useful for your research, please cite our paper:
```
@article{rai2024investigation,
  title={An Investigation of Neuron Activation as a Unified Lens to Explain Chain-of-Thought Eliciting Arithmetic Reasoning of LLMs},
  author={Rai, Daking and Yao, Ziyu},
  journal={arXiv preprint arXiv:2406.12288},
  year={2024}
}
```

## Environment Setup
This project is tested in Python 3.8.5.

To get started, set up the environment:
```
python -m venv env 
source env/bin/activate
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Now, clone the repository. 
```
git clone https://github.com/Dakingrai/ood-generalization-semantic-boundary-techniques.git
cd ood-generalization-semantic-boundary-techniques](https://github.com/Dakingrai/neuron-analysis-cot-arithmetic-reasoning)
```

## Running Llama2 on GSM8K test set
```
torchrun --nproc_per_node 1 gsm8k_inference.py --ckpt_dir <path_to_model> --tokenizer_path <path_to_model>/tokenizer.model --prompt equation_only --few_shot True
```
Note: The --prompt argument should specify the filename of the prompt located in the data/prompts directory. For instance, to run inference using a prompt that contains only equations without any text, the argument should be set to "equation_only".
