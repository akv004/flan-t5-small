import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Resolve paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
model_path = os.path.join(script_dir, "../models/fine_tuned_model/")  # Adjust as needed

# Verify the resolved model path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory not found at: {model_path}")

# Load the fine-tuned model and tokenizer
print(f"Loading model from: {model_path}")
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = T5Tokenizer.from_pretrained(model_path)

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
