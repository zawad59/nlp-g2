# Function to generate predictions and compare with true labels
def generate_predictions_and_evaluate(texts, true_labels):
    predicted_labels = []
    for i, context in enumerate(texts):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context},
        ]
        
        # Generate response
        output = pipe(messages, max_new_tokens=50)
        
        # Debugging: Print the output structure
        print(f"Output structure for message {i}: {output}")
        
        # Extract generated text based on observed structure
        generated_text = ""
        if isinstance(output, list) and len(output) > 0:
            # Try to extract the generated text safely
            generated_text = output[0].get("generated_text", "")
        
        # Ensure generated_text is a single string
        if isinstance(generated_text, list):
            generated_text = " ".join(generated_text)  # Join list into a single string if necessary

        # Extract predicted answer based on presence of known options
        choices = context.split("Choices: ")[1].split(", ")
        predicted_label = -1  # Default to -1 if no match is found
        for idx, choice in enumerate(choices):
            if choice.lower() in generated_text.lower():
                predicted_label = idx
                break
        predicted_labels.append(predicted_label)

    # Calculate accuracy and F1 score based on predicted vs true labels
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    return predicted_labels

# Run the function and observe the output structure
predicted_labels = generate_predictions_and_evaluate(test_texts, test_labels)
