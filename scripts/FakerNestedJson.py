import os
import pandas as pd
import json
import random
from faker import Faker

fake = Faker()

# Ensure the data directory exists
data_dir = '../data'
os.makedirs(data_dir, exist_ok=True)

def generate_phone_number():
    """
    Return a phone number in a reasonable U.S. format, e.g. (541) 612-2857.
    We skip some invalid area codes to keep it consistent.
    """
    area_code = random.randint(200, 999)    # skip 000, 100, 911, etc.
    prefix = random.randint(100, 999)
    line_number = random.randint(1000, 9999)
    return f"({area_code}) {prefix}-{line_number}"

def generate_nested_mock_data(num_records):
    data = []
    for _ in range(num_records):
        # Basic user info
        user_info = {
            "name": fake.name(),
            "age": random.randint(18, 60),
            "city": fake.city(),
            # Two phone numbers with custom generator
            "phone_numbers": [
                {
                    "type": "home",
                    "number": generate_phone_number()
                },
                {
                    "type": "mobile",
                    "number": generate_phone_number()
                }
            ]
        }

        # Some orders array
        orders = []
        for _ in range(random.randint(1, 3)):  # 1 to 3 orders
            orders.append({
                "id": fake.uuid4(),
                "product": fake.word(),
                "price": round(random.uniform(5, 500), 2)
            })

        record = {
            "user": user_info,
            "orders": orders
        }
        data.append(record)
    return data

# Generate, say, 1,000 records
num_records = 1000
records = generate_nested_mock_data(num_records)

# Convert to a DataFrame (just for splitting convenience)
df = pd.DataFrame(records)
print("Top-level DataFrame columns:", df.columns)

# Split data into train and validation sets
train_df = df.sample(frac=0.8, random_state=42)
valid_df = df.drop(train_df.index)

def convert_to_json_format(df):
    lines = []
    for _, row in df.iterrows():
        user_info = row['user']  # nested dict
        orders = row['orders']   # list of dicts

        # Build prompt text (short summary)
        prompt_text = (
            f"Convert this to JSON: user name is {user_info['name']}, "
            f"age is {user_info['age']}, city is {user_info['city']}, "
            f"and they have {len(orders)} orders."
        )

        # The target JSON is the full nested structure
        output_json = {
            "user": user_info,
            "orders": orders
        }

        # Format: "PROMPT -> JSON_string"
        line = f"{prompt_text} -> {json.dumps(output_json)}"
        lines.append(line)
    return lines

train_lines = convert_to_json_format(train_df)
valid_lines = convert_to_json_format(valid_df)

def write_lines_to_file(file_path, lines):
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line + "\n")

# Write train and validation data
write_lines_to_file(os.path.join(data_dir, 'train.txt'), train_lines)
write_lines_to_file(os.path.join(data_dir, 'valid.txt'), valid_lines)

print("Nested data with clean phone numbers written to train.txt and valid.txt successfully!")
