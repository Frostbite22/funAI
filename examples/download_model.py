from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "meta-llama/Llama-2-13b-chat-hf"
model_dir = "./llama-2-13b-chat-hf"

# Set Hugging Face token
api_token = os.getenv("HUGGINGFACE_TOKEN")
if api_token is None:
    raise ValueError("Hugging Face API token is not set. Please set the HUGGINGFACE_TOKEN environment variable.")

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_token)

# Save the model and tokenizer locally
tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)

print("Model and tokenizer downloaded and saved locally.")
