import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_DIR = "../models/gpt2_json_model_v1"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

prompt = (
    "generate sample json"
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=100,
        num_beams=1,        # for open-ended text, might use sampling
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
output_text = tokenizer.decode(output_ids[0])
print("Completion:", output_text)

# The model hopefully learned to produce JSON after the "->"
