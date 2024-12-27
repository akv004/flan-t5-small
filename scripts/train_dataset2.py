import os
import re
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

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
new_version = int(re.search(r"\d+", latest_model.name).group()) + 1 if latest_model else 1
fine_tuned_model_path = models_dir / f"fine_tuned_model_v{new_version}"
print(f"New fine-tuned model will be saved to: {fine_tuned_model_path}")

# Load the dataset
ds = load_dataset("knowrohit07/saraswati-stem")

# Create validation split from train data
print("Splitting dataset into train and validation...")
ds = ds["train"].train_test_split(test_size=0.1)  # Use 10% as validation set

# Debug: Print the first few entries of the dataset
print("First few entries of the train dataset:")
for i, entry in enumerate(ds['train']):
    if i >= 5:
        break
    print(entry)

print("First few entries of the validation dataset:")
for i, entry in enumerate(ds['test']):
    if i >= 5:
        break
    print(entry)

# Prepare the data for training
train_data = []
validation_data = []

for entry in ds['train']:
    for message in entry['conversation']:
        if message['role'] == 'User':
            user_input = message['content']
        elif message['role'] == 'Assistant' and 'do_train' in message and message['do_train']:
            train_data.append({
                "input_text": f"User: {user_input}",
                "target_text": message['content']
            })

for entry in ds['test']:
    for message in entry['conversation']:
        if message['role'] == 'User':
            user_input = message['content']
        elif message['role'] == 'Assistant' and 'do_train' in message and message['do_train']:
            validation_data.append({
                "input_text": f"User: {user_input}",
                "target_text": message['content']
            })

# Convert lists to dictionaries
train_data_dict = {"input_text": [item["input_text"] for item in train_data], "target_text": [item["target_text"] for item in train_data]}
validation_data_dict = {"input_text": [item["input_text"] for item in validation_data], "target_text": [item["target_text"] for item in validation_data]}

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict(train_data_dict)
validation_dataset = Dataset.from_dict(validation_data_dict)

# Load tokenizer and model
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
def preprocess_data(examples):
    inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(examples["target_text"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess_data, batched=True)
validation_dataset = validation_dataset.map(preprocess_data, batched=True)

# Specify training arguments
training_args = TrainingArguments(
    output_dir=str(fine_tuned_model_path),
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir=str(script_dir / "../logs"),
    logging_steps=100,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(str(fine_tuned_model_path))
tokenizer.save_pretrained(str(fine_tuned_model_path))
print("Fine-tuned model saved successfully!")
print(f"Model saved at: {fine_tuned_model_path}")