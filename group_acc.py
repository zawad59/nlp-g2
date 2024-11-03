import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import accuracy_score

# Load the data from SP_dev.npy
data = np.load('SP_dev.npy', allow_pickle=True)

# Group questions by prefix
groups = {}
for item in data:
    group_id = item['id'].split('_')[0]  # Extract the group prefix
    if group_id not in groups:
        groups[group_id] = []
    groups[group_id].append(item)

# Initialize sentence embedding model for similarity checks
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Llama model for text generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model_id, tokenizer=tokenizer, device_map="auto")

# Set a similarity threshold
similarity_threshold = 0.85
interval_accuracies = []
all_results = []
batch_size = 5  # Number of groups per time interval

# Process each interval
for start in range(0, len(groups), batch_size):
    interval_correct_predictions = []
    interval_results = []
    
    # Get the groups for this interval
    interval_groups = dict(list(groups.items())[start:start + batch_size])
    
    # Process each group in the interval
    for group_id, items in interval_groups.items():
        # Generate sentence embeddings for the group's questions
        questions = [item['question'] for item in items]
        answers = [item['answer'] for item in items]
        embeddings = embedder.encode(questions)

        # Calculate pairwise cosine similarities and process predictions
        correct_predictions = True
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                # Calculate cosine similarity
                similarity = util.cos_sim(embeddings[i], embeddings[j]).item()

                # If questions are semantically similar, check if predictions align
                if similarity > similarity_threshold:
                    prompt = f"{questions[i]}\nChoices: {', '.join(items[i]['choice_list'])}"
                    result = pipe(prompt, max_new_tokens=30)
                    
                    # Extract generated text and compare with answer
                    generated_text = result[0]['generated_text']
                    predicted_answer = generated_text.strip()

                    # Record if the prediction matches the actual answer
                    correct_prediction = predicted_answer == answers[i]
                    if not correct_prediction:
                        correct_predictions = False

                    interval_results.append({
                        'Interval': start // batch_size + 1,
                        'Group ID': group_id,
                        'Question 1': questions[i],
                        'Question 2': questions[j],
                        'Similarity': similarity,
                        'Predicted Answer': predicted_answer,
                        'Actual Answer': answers[i],
                        'Correct Prediction': correct_prediction
                    })

        # Record if all predictions for this group were consistent
        interval_correct_predictions.append(correct_predictions)

    # Calculate accuracy for this interval
    interval_accuracy = sum(interval_correct_predictions) / len(interval_correct_predictions)
    interval_accuracies.append({
        'Interval': start // batch_size + 1,
        'Accuracy': interval_accuracy * 100
    })
    all_results.extend(interval_results)

# Save interval accuracies to CSV
df_interval_accuracies = pd.DataFrame(interval_accuracies)
df_interval_accuracies.to_csv('interval_accuracies.csv', index=False)
print("Interval accuracies saved to 'interval_accuracies.csv'.")

# Save detailed results to CSV
df_results = pd.DataFrame(all_results)
df_results.to_csv('interval_predictions_dev.csv', index=False)
print("Prediction details with intervals saved to 'interval_predictions_dev.csv'.")
