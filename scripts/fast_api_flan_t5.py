import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Function to find the latest fine-tuned model
def find_latest_model(models_dir):
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    fine_tuned_dirs = [d for d in model_dirs if re.match(r"fine_tuned_model_v\d+", d.name)]
    if not fine_tuned_dirs:
        raise FileNotFoundError("No fine-tuned model found.")
    fine_tuned_dirs.sort(key=lambda x: int(re.search(r"\d+", x.name).group()), reverse=True)
    return fine_tuned_dirs[0]

# Paths
script_dir = Path(__file__).resolve().parent
models_dir = script_dir / "../models"

# Find the latest fine-tuned model directory
latest_model_dir = find_latest_model(models_dir)

# Load the fine-tuned model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model from: {latest_model_dir}")
model = T5ForConditionalGeneration.from_pretrained(latest_model_dir).to(device)
tokenizer = T5Tokenizer.from_pretrained(latest_model_dir)
print("Model loaded.")

# Define the input/output structure for the API
class TaskRequest(BaseModel):
    input_text: str
    max_length: int = 100  # Default maximum length for the response
    temperature: float = 0.7  # Default temperature for sampling
    num_beams: int = 5  # Beam search for better quality
    do_sample: bool = False  # Sampling disabled for deterministic results

class TaskResponse(BaseModel):
    output_text: str

# Define the text-to-text endpoint
@app.post("/task", response_model=TaskResponse)
async def perform_task(request: TaskRequest):
    try:
        # Generate response from the model
        input_ids = tokenizer.encode(request.input_text, return_tensors="pt").to(device)
        output_ids = model.generate(
            input_ids,
            max_length=request.max_length,
            temperature=request.temperature,
            num_beams=request.num_beams,
            do_sample=request.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return TaskResponse(output_text=output_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
