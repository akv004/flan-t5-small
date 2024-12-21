import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define the model name
model_name = "google/flan-t5-small"

# Paths to save the model and tokenizer
current_dir = os.getcwd()  # Use current working directory
save_path = os.path.join(current_dir, "../models/base_model/")

# Download and save the model
print("Downloading the base model...")
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Save to the specified directory
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Print the full path
full_save_path = os.path.abspath(save_path)
print(f"Base model downloaded and saved to {full_save_path}.")
