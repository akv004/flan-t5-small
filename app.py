from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Test input
input_text = "Translate English to French: What is your name?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output
outputs = model.generate(inputs.input_ids, max_length=50)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {input_text}")
print(f"Output: {decoded_output}")
