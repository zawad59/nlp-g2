import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Set CUDA_LAUNCH_BLOCKING for debugging CUDA issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load the training data
train_data = np.load('SP_train.npy', allow_pickle=True)
train_texts = [item['question'] for item in train_data]
train_labels = [item['label'] for item in train_data]  # Ensure train_labels is defined

# Convert labels to tensor format with the correct dtype
train_labels = torch.tensor(train_labels, dtype=torch.long)

# Load the test data
test_data = np.load('SP_test.npy', allow_pickle=True)
test_texts = [item['question'] for item in test_data]
# If test labels are available, uncomment the following line
# test_labels = [item['label'] for item in test_data]

# Load a pretrained tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if the tokenizer has a padding token; if not, add one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(train_labels)))
model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings if new tokens are added

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Create a PyTorch Dataset for training and test
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.encodings['input_ids'])

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    evaluation_strategy="no",  # Disable evaluation during training
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    num_train_epochs=3,              
    weight_decay=0.01,
)

# Define Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
)

# Train the model
trainer.train()

# Predict on the test set
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# If test labels are available, calculate accuracy and F1 score
# Uncomment the following lines if test labels are available
test_labels_tensor = torch.tensor(test_labels)
accuracy = accuracy_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

# Save predictions to a CSV file
df_predictions = pd.DataFrame({
    'Text': test_texts,
    'Predicted Label': predicted_labels
})
df_predictions.to_csv('predictions.csv', index=False)
print("Predicted labels saved to predictions.csv.")
