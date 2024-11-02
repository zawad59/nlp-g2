import os
import numpy as np
import pandas as pd
import torch
from transformers import pipeline

# Set CUDA_LAUNCH_BLOCKING for debugging CUDA issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load the training data (if needed for other purposes, we load it here)
train_data = np.load('SP_train.npy', allow_pickle=True)
train_texts = [item['question'] for item in train_data]
train_labels = [item['label'] for item in train_data]  # Load labels if needed for training

# Load the test data
test_data = np.load('SP_test.npy', allow_pickle=True)
test_texts = [item['question'] for item in test_data]

# Initialize the text-generation pipeline with Llama
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Define the system prompt for pirate speak
system_prompt = "You are a pirate chatbot who always responds in pirate speak!"

# Generate responses for each question in the test set
generated_responses = []
for question in test_texts:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    output = pipe(
        messages,
        max_new_tokens=256,
    )
    generated_response = output[0]["generated_text"]
    generated_responses.append(generated_response)

# Save generated responses to a CSV file
df_responses = pd.DataFrame({
    'Question': test_texts,
    'Generated Response': generated_responses
})
df_responses.to_csv('generated_responses.csv', index=False)
print("Generated responses saved to generated_responses.csv.")
