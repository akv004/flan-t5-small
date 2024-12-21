train_file = "../data/train.txt"
valid_file = "../data/valid.txt"

def validate_file(file_path):
    with open(file_path, "r") as file:
        for line in file:
            if "->" not in line:
                print(f"Invalid line (missing '->'): {line}")
            else:
                input_text, output_text = line.split("->", 1)
                if not input_text.strip() or not output_text.strip():
                    print(f"Invalid line (empty input/output): {line}")

validate_file(train_file)
validate_file(valid_file)
