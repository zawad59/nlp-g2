import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, util

# Load the training and test data
train_data = np.load('SP_train.npy', allow_pickle=True)
test_data = np.load('SP_test.npy', allow_pickle=True)
test_answers = np.load('SP_test_answer.npy', allow_pickle=True)

# Extract question IDs and correct answer indices from test_answers
question_ids = [answer[0] for answer in test_answers]  # First element is the question ID
test_answer_indices = [int(answer[1]) for answer in test_answers]  # Second element is the correct answer index

# Prepare test data
test_texts = []
test_choices = []  # Store choices for each test question
for item in test_data:
    question = item['question']
    choice_list = item['choice_list']
    context = f"{question} Choices: {', '.join(choice_list)}"
    
    test_texts.append(context)
    test_choices.append(choice_list)

# Initialize the tokenizer and model for text generation with Llama
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Initialize the text generation pipeline with bfloat16 for reduced memory usage
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Initialize a sentence transformer model for semantic similarity on CPU
similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Example for one-shot learning
example_question = train_data[0]['question']
example_answer = train_data[0]['choice_list'][train_data[0]['label']]
example_context = f"Example Question: {example_question} Correct Answer: {example_answer}\n\n"

# Function to generate predictions and explanations with semantic similarity
def generate_predictions_and_evaluate(texts, choices, true_indices, question_ids, mode="zero-shot"):
    predicted_labels = []
    predicted_answers = []
    actual_answers = []
    explanations = []

    for i, (context, choice_list) in enumerate(zip(texts, choices)):
        # Prepare context for zero-shot or one-shot
        if mode == "zero-shot":
            prompt = context
        elif mode == "one-shot":
            prompt = example_context + "Question: " + context

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        # Generate response
        output = pipe(messages, max_new_tokens=50)
        
        # Extract generated text safely
        generated_text = ""
        if isinstance(output, list) and len(output) > 0:
            generated_content = output[0].get("generated_text", "")
            if isinstance(generated_content, list):
                generated_text = " ".join([str(part) for part in generated_content])
            elif isinstance(generated_content, str):
                generated_text = generated_content

        # Ensure generated_text is a single string
        generated_text = str(generated_text)

        # Compute similarity between the generated text and each choice
        generated_embedding = similarity_model.encode(generated_text, convert_to_tensor=True)
        choice_embeddings = similarity_model.encode(choice_list, convert_to_tensor=True)
        similarities = util.cos_sim(generated_embedding, choice_embeddings)[0].cpu().numpy()

        # Select the choice with the highest similarity score
        predicted_label = int(np.argmax(similarities))
        explanation = (
            f"The predicted answer '{choice_list[predicted_label]}' was chosen because it has the highest "
            f"semantic similarity ({similarities[predicted_label]:.2f}) to the generated response. "
            f"Generated Response: '{generated_text}'"
        )

        predicted_labels.append(predicted_label)

        # Store actual and predicted answers in text form
        actual_answer = choice_list[true_indices[i]]
        predicted_answer = choice_list[predicted_label]
        
        actual_answers.append(actual_answer)
        predicted_answers.append(predicted_answer)
        explanations.append(explanation)

    # Calculate accuracy and F1 score based on predicted vs true labels
    accuracy = accuracy_score(true_indices, predicted_labels)
    f1 = f1_score(true_indices, predicted_labels, average='weighted')
    print(f"Accuracy ({mode}): {accuracy}")
    print(f"F1 Score ({mode}): {f1}")

    return question_ids, actual_answers, predicted_answers, explanations

# Function to generate predictions in batches to manage memory usage
def generate_predictions_in_batches(texts, choices, true_indices, question_ids, mode="zero-shot", batch_size=4):
    all_question_ids = []
    all_actual_answers = []
    all_predicted_answers = []
    all_explanations = []

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        batch_choices = choices[batch_start:batch_start + batch_size]
        batch_true_indices = true_indices[batch_start:batch_start + batch_size]
        batch_question_ids = question_ids[batch_start:batch_start + batch_size]

        # Process each batch
        ids, actuals, predictions, expls = generate_predictions_and_evaluate(
            batch_texts, batch_choices, batch_true_indices, batch_question_ids, mode=mode
        )

        all_question_ids.extend(ids)
        all_actual_answers.extend(actuals)
        all_predicted_answers.extend(predictions)
        all_explanations.extend(expls)
        torch.cuda.empty_cache()  # Clear memory between batches

    return all_question_ids, all_actual_answers, all_predicted_answers, all_explanations

# Generate predictions and evaluate on test data for zero-shot learning
print("Zero-Shot Learning:")
question_ids_zero, actual_answers_zero, predicted_answers_zero, explanations_zero = generate_predictions_in_batches(
    test_texts, test_choices, test_answer_indices, question_ids, mode="zero-shot", batch_size=4
)

# Save zero-shot predictions to a CSV file for review
df_predictions_zero_shot = pd.DataFrame({
    'Question ID': question_ids_zero,
    'Question': test_texts,
    'Actual Correct Answer': actual_answers_zero,
    'Predicted Answer': predicted_answers_zero,
    'Explanation': explanations_zero
})
df_predictions_zero_shot.to_csv('predictions_zero_shot.csv', index=False)
print("Zero-shot predictions saved to predictions_zero_shot.csv.")

# Generate predictions and evaluate on test data for one-shot learning
print("One-Shot Learning:")
question_ids_one, actual_answers_one, predicted_answers_one, explanations_one = generate_predictions_in_batches(
    test_texts, test_choices, test_answer_indices, question_ids, mode="one-shot", batch_size=4
)

# Save one-shot predictions to a CSV file for review
df_predictions_one_shot = pd.DataFrame({
    'Question ID': question_ids_one,
    'Question': test_texts,
    'Actual Correct Answer': actual_answers_one,
    'Predicted Answer': predicted_answers_one,
    'Explanation': explanations_one
})
df_predictions_one_shot.to_csv('predictions_one_shot.csv', index=False)
print("One-shot predictions saved to predictions_one_shot.csv.")
