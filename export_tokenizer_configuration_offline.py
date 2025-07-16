from transformers import AutoTokenizer,AutoConfig

# Load the tokenizer for LLaMA 3.3 70B Instruct (or equivalent)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
tokenizer.save_pretrained("./llama3-tokenizer")

config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
config.save_pretrained("./llama3-tokenizer")

