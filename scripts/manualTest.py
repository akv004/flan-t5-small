from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "/home/amit/DataspellProjects/flan-t5-small/scripts/../models/fine_tuned_model_v3"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Test input
input_text = "Convert this to JSON: name is Alice, age is 25, and city is Los Angeles."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output_ids = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,  # Use beam search
    temperature=0.7,
    do_sample=False  # Disable random sampling
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated JSON:", output_text)
