import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Load the training data
train_data = np.load('SP_train.npy', allow_pickle=True)
train_texts = [item['question'] for item in train_data]
train_labels = [item['label'] for item in train_data]

# Load the test data
test_data = np.load('SP_test.npy', allow_pickle=True)
test_texts = [item['question'] for item in test_data]

# Load a pretrained tokenizer and model
model_name = "distilbert-base-uncased"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(train_labels)))

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Convert labels to tensor format
train_labels = torch.tensor(train_labels)

# Create a PyTorch Dataset for training
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
    evaluation_strategy="epoch",     
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    num_train_epochs=3,              
    weight_decay=0.01,
)

# Define a simple Trainer with only training dataset
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
)

# Train the model
trainer.train()

# Predict on the test set (without true labels)
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Print or save predicted labels
print("Predicted labels for test set:", predicted_labels)
