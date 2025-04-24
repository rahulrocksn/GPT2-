import torch
from model import GPT2ForEmotionClassification
from transformers import GPT2Tokenizer
from dataset import get_dataset_and_emotions

def predict_emotion(text, model, tokenizer, emotions, device):
    model.eval()
    
    # Tokenize the input text
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
    
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)
    
    # Get the emotions with probability > 0.5
    predictions = (probs > 0.5).float()
    
    # Convert predictions to emotion labels
    result = []
    for i, pred in enumerate(predictions[0]):
        if pred.item() == 1:
            result.append((emotions[i], probs[0][i].item()))
    
    # Sort by probability (highest first)
    result.sort(key=lambda x: x[1], reverse=True)
    
    # If no emotion exceeds threshold, return the one with highest probability
    if not result:
        max_idx = torch.argmax(probs[0]).item()
        result = [(emotions[max_idx], probs[0][max_idx].item())]
    
    return result

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get emotions list
    _, emotions = get_dataset_and_emotions()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("emotion_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = GPT2ForEmotionClassification(num_labels=len(emotions))
    model.load_state_dict(torch.load("gpt2_emotion_classifier.pt"))
    model.to(device)
    
    # Test examples
    test_sentences = [
        "This sentence is the sentence",
        "I am feeling so happy today!",
        "I am extremely angry about what happened",
        "The news made me very sad",
        "I just won the lottery!"
    ]
    
    for sentence in test_sentences:
        predicted_emotions = predict_emotion(sentence, model, tokenizer, emotions, device)
        print(f"Text: '{sentence}'")
        print(f"Predicted emotions: {[f'{e} ({p:.2f})' for e, p in predicted_emotions]}")
        print("---")
    
    # Interactive mode
    print("\nEnter text to predict emotion (type 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break
        
        predicted_emotions = predict_emotion(user_input, model, tokenizer, emotions, device)
        print(f"Predicted emotions: {[f'{e} ({p:.2f})' for e, p in predicted_emotions]}")

if __name__ == "__main__":
    main()