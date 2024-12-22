import os
import pandas as pd
import json
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Ensure the data directory exists
data_dir = '../data'
os.makedirs(data_dir, exist_ok=True)

# Generate mock data
def generate_mock_data(num_records):
    users = []
    for _ in range(num_records):
        user_info = {
            "name": fake.name(),
            "age": random.randint(18, 60),  # Random age between 18 and 60
            "city": fake.city()
        }
        users.append(user_info)
    return users

# Generate 100 records
num_records = 1000
users = generate_mock_data(num_records)

# Convert to DataFrame
df = pd.DataFrame(users)
print("DataFrame created:\n", df)

# Split data into train and validation sets
train_df = df.sample(frac=0.8, random_state=42)
valid_df = df.drop(train_df.index)

# Function to convert DataFrame to required format
def convert_to_json_format(df):
    lines = []
    for _, row in df.iterrows():
        input_text = f"Convert this to JSON: name is {row['name']}, age is {row['age']}, and city is {row['city']}."
        output_json = {"name": row['name'], "age": row['age'], "city": row['city']}
        lines.append(f"{input_text} -> {json.dumps(output_json)}")
    return lines

# Convert data to required format
train_lines = convert_to_json_format(train_df)
valid_lines = convert_to_json_format(valid_df)

# Write data to files
def write_lines_to_file(file_path, lines):
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(f"{line}\n")

# Write train and validation data
write_lines_to_file(os.path.join(data_dir, 'train.txt'), train_lines)
write_lines_to_file(os.path.join(data_dir, 'valid.txt'), valid_lines)

# Verify that the files have been written
print("Data written to train.txt and valid.txt successfully")
