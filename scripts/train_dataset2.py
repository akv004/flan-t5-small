from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Load the dataset
ds = load_dataset("knowrohit07/saraswati-stem")

# Prepare the data for training
train_data = []
validation_data = []

for entry in ds['train']:
    if entry['role'] == 'User':
        user_input = entry['content']
    elif entry['role'] == 'Assistant' and 'do_train' in entry and entry['do_train']:
        train_data.append({
            "input_text": f"User: {user_input}",
            "target_text": entry['content']
        })

for entry in ds['validation']:
    if entry['role'] == 'User':
        user_input = entry['content']
    elif entry['role'] == 'Assistant' and 'do_train' in entry and entry['do_train']:
        validation_data.append({
            "input_text": f"User: {user_input}",
            "target_text": entry['content']
        })

# Convert to Hugging Face Dataset format
from datasets import Dataset
train_dataset = Dataset.from_dict(train_data)
validation_dataset = Dataset.from_dict(validation_data)

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

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
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
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
trainer.save_model("./trained_model")
