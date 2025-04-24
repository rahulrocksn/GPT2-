import torch
from model import GPT2ForEmotionClassification
from dataset import prepare_dataloaders, get_dataset_and_emotions
from evaluation import evaluate_emotion_classifier

def benchmark():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataloaders and emotions
    train_loader, val_loader, tokenizer, emotions = prepare_dataloaders(batch_size=32)
    
    # Load trained model
    model = GPT2ForEmotionClassification(num_labels=len(emotions))
    model.load_state_dict(torch.load("gpt2_emotion_classifier.pt"))
    model.to(device)
    
    # Evaluate the model
    print("Evaluating model on validation set...")
    # Inside your benchmark() function, after calling evaluate_emotion_classifier:
    metrics = evaluate_emotion_classifier(model, val_loader, device)

    print("\nOverall metrics:")
    print(f"Exact match accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"Flattened accuracy: {metrics['flattened_accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # For comparison with other benchmarks
    print("\nReference metrics for comparison:")
    print("Note: These metrics are from different tasks and not directly comparable")
    print("HellaSwag - GPT-2 (117M): ~33% accuracy")
    print("HellaSwag - BERT-Large: ~41% accuracy")
    print("HellaSwag - RoBERTa-Large: ~85% accuracy")

if __name__ == "__main__":
    benchmark()