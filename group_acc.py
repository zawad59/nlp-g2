import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import accuracy_score

# Load the data
data = [
    {'id': 'SP-164', 'question': 'Not a single parent objected when the teacher spanked every child in the class. How come?', 'answer': 'The teacher was in an orphanage school.'},
    {'id': 'SP-164_SR', 'question': 'When the instructor spanked every child in the class, not a single parent protested. Why is this so?', 'answer': 'The teacher was in an orphanage school.'},
    {'id': 'SP-164_CR', 'question': "Mark was in the kitchen cooking everyday and yet he wasn't getting paid. He had to buy all the ingredients and everything himself as well. why?", 'answer': 'He was a chef and a dad at the same time. when he was home, nobody was paying him to cook'},
    # Add more data here...
]

# Group questions by prefix
groups = {}
for item in data:
    group_id = item['id'].split('_')[0]
    if group_id not in groups:
        groups[group_id] = []
    groups[group_id].append(item)

# Initialize sentence embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the model for predictions
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model_id, tokenizer=tokenizer, device_map="auto")

# Set a similarity threshold
similarity_threshold = 0.85
group_accuracies = []

# Process each group
for group_id, items in groups.items():
    # Generate sentence embeddings for the group's questions
    questions = [item['question'] for item in items]
    answers = [item['answer'] for item in items]
    embeddings = embedder.encode(questions)

    # Calculate pairwise cosine similarities
    correct_predictions = True
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            # Calculate cosine similarity
            similarity = util.cos_sim(embeddings[i], embeddings[j]).item()

            # If the questions are semantically similar, check if predictions align
            if similarity > similarity_threshold:
                prompt = questions[i] + "\nChoices: " + answers[i]
                result = pipe(prompt, max_new_tokens=30)
                
                # Extract generated text and compare with answer
                generated_text = result[0]['generated_text']
                predicted_answer = generated_text.strip()

                if predicted_answer != answers[i]:
                    correct_predictions = False

    group_accuracies.append(correct_predictions)

# Calculate group-based accuracy
group_accuracy = sum(group_accuracies) / len(group_accuracies)
print(f"Group-Based Accuracy: {group_accuracy * 100:.2f}%")
