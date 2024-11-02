import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Load your Llama model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"  # Replace with the actual path to the Llama model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Check if the tokenizer has a padding token; if not, add one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
train_labels = torch.tensor(train_labels, dtype=torch.long)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(train_labels)))
model.resize_token_embeddings(len(tokenizer))

# Load the training data
train_data = np.load('SP_train.npy', allow_pickle=True)

# Prepare training data
train_texts = []
train_labels = []

for item in train_data:
    question = item['question']
    correct_answer = item['answer']
    choices = item['choice_list']
    # Assuming choices is defined somewhere in your code
    try:
        correct_index = choices.index(correct_answer)
    except ValueError:
        print(f"'{correct_answer}' is not in choices list:", choices)
        correct_index = None  # or assign a default index or value if needed

    
    # Encode question with each choice
    for i, choice in enumerate(choices):
        input_text = f"Question: {question} Choice: {choice}"
        train_texts.append(input_text)
        train_labels.append(1 if i == correct_index else 0)  # Label 1 for the correct answer, 0 otherwise

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
train_labels = torch.tensor(train_labels, dtype=torch.long)

# Create a PyTorch Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    evaluation_strategy="no",    
    per_device_train_batch_size=8,   
    num_train_epochs=3,              
    weight_decay=0.01,
)

# Define evaluation metrics
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(pred.label_ids, preds)
    f1 = f1_score(pred.label_ids, preds, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}

# Ensure model and dataset are on CPU for debugging (remove `.to('cpu')` after debugging)
model.to('cpu')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Function to predict answers on new questions
def predict_answers(new_data):
    new_texts = []
    question_ids = []
    for item in new_data:
        question = item['question']
        choices = item['choice_list']
        question_ids.append(item['id'])
        
        # Prepare input for each choice
        for choice in choices:
            input_text = f"Question: {question} Choice: {choice}"
            new_texts.append(input_text)

    # Tokenize new data
    new_encodings = tokenizer(new_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**new_encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).reshape(-1, len(choices))

    # Extract predictions for each question
    predicted_answers = []
    for i, question_id in enumerate(question_ids):
        correct_choice_index = predictions[i].item()
        predicted_answer = new_data[i]['choice_list'][correct_choice_index]
        predicted_answers.append((question_id, predicted_answer))
    
    return predicted_answers

# Example usage with new test data
test_data = np.load('SP_test.npy', allow_pickle=True)
predicted_answers = predict_answers(test_data)
for question_id, answer in predicted_answers:
    print(f"Question ID: {question_id}, Predicted Answer: {answer}")
