from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import os

BASE_MODEL = "../models/gpt2_json_model"  # or "gpt2-medium"/"EleutherAI/gpt-neo-125M"
OUTPUT_DIR = "../models/gpt2_json_model_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Minimal data
TRAIN_FILE = "../data/train.txt"
VALID_FILE = "../data/valid.txt"

dataset = load_dataset("text", data_files={
    "train": TRAIN_FILE,
    "validation": VALID_FILE
})

tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
# GPT-2 doesn't have a pad token by default
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)

def format_example(text):
    # e.g. "Convert this to JSON: name is ... -> {...}"
    # We'll keep that exact format but feed it as a single string:
    return text.strip()

def tokenize_function(examples):
    formatted = [format_example(t) for t in examples["text"]]
    return tokenizer(formatted, truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    fp16=False,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
