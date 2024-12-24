
# Key Enhancements:
# Incremental Saving:
# Automatically detects the latest version of the model and increments it for new training runs.
# If no previous model is found, starts with the base google/flan-t5-small.
#
# Task-Specific Preprocessing:
# Prepares the dataset for summarization by appending the task prefix ("summarize: ").
# Handles tokenization and ensures padding and truncation for consistent input lengths.
#
# FP16 Training:
# Enables mixed precision (FP16) automatically if a compatible GPU is available.
#
# Dataset Integration:
# Dynamically uses Hugging Face datasets for fine-tuning.
#     Can easily adapt to other datasets by modifying the preprocess_data function.
#
# Improved Logging:
# Outputs memory usage and tokenized examples for better visibility.
#
#

import os
import re
import torch
from pathlib import Path
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset


# Print GPU memory usage
def print_memory_stats():
    if torch.cuda.is_available():
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths
script_dir = Path(__file__).resolve().parent
models_dir = script_dir / "../models"

# Find the latest fine-tuned model
def find_latest_model():
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    fine_tuned_dirs = [d for d in model_dirs if re.match(r"fine_tuned_model_v\d+", d.name)]
    if not fine_tuned_dirs:
        return None
    fine_tuned_dirs.sort(key=lambda x: int(re.search(r"\d+", x.name).group()), reverse=True)
    return fine_tuned_dirs[0]

# Determine base model path
latest_model = find_latest_model()
if latest_model:
    print(f"Using latest fine-tuned model as base: {latest_model}")
    base_model_path = latest_model
else:
    print("No fine-tuned model found. Using base model.")
    base_model_path = "google/flan-t5-small"  # Use the base Flan-T5 model

# Generate new versioned fine-tuned model directory
new_version = int(re.search(r"\d+", latest_model.name).group()) + 1 if latest_model else 1
fine_tuned_model_path = models_dir / f"fine_tuned_model_v{new_version}"

# Load dataset from Hugging Face
print("Loading dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Load tokenizer and model
print(f"Loading tokenizer and model from: {base_model_path}")
tokenizer = T5Tokenizer.from_pretrained(str(base_model_path))
model = T5ForConditionalGeneration.from_pretrained(str(base_model_path))

# Preprocess the dataset
def preprocess_data(examples):
    inputs = ["summarize: " + article for article in examples["article"]]
    summaries = [highlight for highlight in examples["highlights"]]

    # Tokenize inputs and summaries
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(summaries, max_length=128, truncation=True, padding="max_length")

    # Replace pad token IDs with -100 in labels (ignored by loss)
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_ids]
        for label_ids in labels["input_ids"]
    ]
    return model_inputs

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset["train"].column_names)
print("Sample tokenized input_ids:", tokenized_dataset["train"][0]["input_ids"])
print("Sample tokenized labels:", tokenized_dataset["train"][0]["labels"])

# Data collator for seq2seq tasks
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=str(fine_tuned_model_path),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=2,
    logging_dir=str(script_dir / "../logs"),
    logging_steps=10,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),  # Use FP16 if GPU supports it
    dataloader_num_workers=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=True,
    greater_is_better=False,
)

# Initialize the Trainer
print("Initializing Trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
print("Starting training...")
trainer.train()

# Print memory usage after training
print("Memory stats after training:")
print_memory_stats()

# Save the fine-tuned model
print(f"Saving fine-tuned model to: {fine_tuned_model_path}")
trainer.save_model(str(fine_tuned_model_path))
tokenizer.save_pretrained(str(fine_tuned_model_path))
print("Fine-tuned model saved successfully!")
print(f"Model saved at: {fine_tuned_model_path}")
