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
    TrainerCallback
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
# Generate new versioned fine-tuned model directory
new_version = int(re.search(r"\d+", latest_model.name).group()) + 1 if latest_model else 1
fine_tuned_model_path = models_dir / f"fine_tuned_model_v{new_version}"
print(f"New fine-tuned model will be saved to: {fine_tuned_model_path}")

# Load dataset from Hugging Face
print("Loading dataset...")
dataset = load_dataset("ChristianAzinn/json-training")

# Create validation split from train data
print("Splitting dataset into train and validation...")
dataset = dataset["train"].train_test_split(test_size=0.1)  # Use 10% as validation set

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

# Preprocess the dataset
def preprocess_data(examples):
    inputs = []
    outputs = []

    # Filter invalid examples
    for query, schema, response in zip(examples["query"], examples["schema"], examples["response"]):
        if query and schema and response:  # Ensure fields are not empty
            inputs.append(f"generate response: {query} \n Desired Schema: {schema}")
            outputs.append(response)

    # Tokenize inputs and outputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length")

    # Replace pad token ID with -100 in labels (ignored by loss)
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_ids]
        for label_ids in labels["input_ids"]
    ]

    return model_inputs

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset["train"].column_names)
print("Sample tokenized input_ids:", tokenized_dataset["train"][0]["input_ids"])
print("Sample tokenized labels:", tokenized_dataset["train"][0]["labels"])

# Debugging: Check dataset after tokenization
for i in range(5):
    print(f"Debug Sample {i}:")
    print(f"Input IDs: {tokenized_dataset['train'][i]['input_ids']}")
    print(f"Labels: {tokenized_dataset['train'][i]['labels']}")

# Inspect one raw sample for debugging
print("Inspecting one raw sample:")
print("Query:", dataset["train"][0]["query"])
print("Schema:", dataset["train"][0]["schema"])
print("Response:", dataset["train"][0]["response"])

# Data collator for seq2seq tasks
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Custom callback to log gradient norms
class GradientNormLogger(TrainerCallback):
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        if logs:
            grad_norm = logs.get('grad_norm', None)
            if grad_norm is not None:
                print(f"Gradient norm at step {state.global_step}: {grad_norm}")

# Additional callback to debug gradients
class GradientDebugger(TrainerCallback):
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.norm().item():.4f}")

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=str(fine_tuned_model_path),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=2,
    logging_dir=str(script_dir / "../logs"),
    logging_steps=10,
    learning_rate=5e-6,  # Lower learning rate
    fp16=False,  # Disable FP16 for stability
    dataloader_num_workers=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=True,
    greater_is_better=False,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,  # Gradient clipping
)


# Initialize the Trainer
print("Initializing Trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],  # Use test split for validation
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[GradientNormLogger(), GradientDebugger()],  # Custom callbacks
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
