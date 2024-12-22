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
from peft import get_peft_model, LoraConfig, TaskType

# Print GPU memory usage
def print_memory_stats():
    if torch.cuda.is_available():
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths
script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "../data"
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
    base_model_path = models_dir / "base_model"

# Generate new versioned fine-tuned model directory
new_version = int(re.search(r"\d+", latest_model.name).group()) + 1 if latest_model else 1
fine_tuned_model_path = models_dir / f"fine_tuned_model_v{new_version}"

# Check if training data exists
train_file = data_dir / "train.txt"
valid_file = data_dir / "valid.txt"
if not train_file.exists() or not valid_file.exists():
    raise FileNotFoundError("Training data not found. Please add 'train.txt' and 'valid.txt' in the data directory.")

# Load dataset
print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": str(train_file), "validation": str(valid_file)})

# Load tokenizer and model
print(f"Loading tokenizer and model from: {base_model_path}")
tokenizer = T5Tokenizer.from_pretrained(str(base_model_path))
model = T5ForConditionalGeneration.from_pretrained(str(base_model_path))

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)

# Tokenize the dataset
def tokenize_function(examples):
    input_texts = []
    target_texts = []
    for text in examples["text"]:
        if "->" in text:
            input_text, target_text = text.split("->")
            input_texts.append(input_text.strip())
            target_texts.append(target_text.strip())

    tokenized_inputs = tokenizer(input_texts, max_length=512, truncation=True, padding="max_length")
    tokenized_targets = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length")
    labels = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in tokenized_targets["input_ids"]
    ]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Sample tokenized input_ids:", tokenized_datasets["train"][0]["input_ids"])
print("Sample tokenized labels:", tokenized_datasets["train"][0]["labels"])

# Data collator for seq2seq tasks
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=str(fine_tuned_model_path),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_total_limit=2,
    logging_dir=str(script_dir / "../logs"),
    logging_steps=10,
    learning_rate=1e-4,
    fp16=False,
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
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
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
