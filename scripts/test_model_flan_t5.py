import os
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
script_dir = Path(__file__).resolve().parent
models_dir = script_dir / "../models"

# Find the latest fine-tuned model
def find_latest_model():
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    fine_tuned_dirs = [d for d in model_dirs if re.match(r"fine_tuned_model_v\d+", d.name)]
    if not fine_tuned_dirs:
        raise FileNotFoundError("No fine-tuned model found in the models directory.")
    fine_tuned_dirs.sort(key=lambda x: int(re.search(r"\d+", x.name).group()), reverse=True)
    return fine_tuned_dirs[0]

# Resolve model path
latest_model_path = find_latest_model()
print(f"Using latest fine-tuned model: {latest_model_path}")

# Load the fine-tuned model and tokenizer
print(f"Loading model from: {latest_model_path}")
model = T5ForConditionalGeneration.from_pretrained(str(latest_model_path), device_map="auto", torch_dtype=torch.float16)
tokenizer = T5Tokenizer.from_pretrained(str(latest_model_path))

# Move the model to the appropriate device
model.to(device)

# Test input
prompt = "Translate English to French: Amit is living in the USA."
print(f"Input prompt: {prompt}")

# Tokenize the input
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generate text
print("Generating output...")
with torch.no_grad():
    output = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1, do_sample=False)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)
