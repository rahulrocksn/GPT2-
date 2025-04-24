import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

# def evaluate_emotion_classifier(model, test_loader, device):
#     model.eval()
#     all_predictions = []
#     all_labels = []
    
#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
            
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs["logits"]
            
#             # Convert logits to predictions (threshold at 0.5)
#             predictions = (torch.sigmoid(logits) > 0.5).float()
            
#             # Move to CPU for evaluation
#             all_predictions.append(predictions.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())
    
#     # Concatenate batches
#     all_predictions = np.vstack(all_predictions)
#     all_labels = np.vstack(all_labels)
    
#     # Calculate metrics
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         all_labels, all_predictions, average='weighted', zero_division=0
#     )
    
#     accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())
    
#     # Detailed report
#     print(classification_report(all_labels, all_predictions, zero_division=0))
    
#     return {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1
#     }

def evaluate_emotion_classifier(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            # Convert logits to predictions (threshold at 0.5)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            # Move to CPU for evaluation
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate batches
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Flattened accuracy (element-wise)
    flattened_accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())
    
    # Per-sample exact match accuracy
    exact_match_accuracy = calculate_exact_match_accuracy(all_labels, all_predictions)
    
    # Detailed report
    print(classification_report(all_labels, all_predictions, zero_division=0))
    
    print(f"Flattened accuracy: {flattened_accuracy:.4f}")
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    
    return {
        'flattened_accuracy': flattened_accuracy,
        'exact_match_accuracy': exact_match_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_exact_match_accuracy(y_true, y_pred):
    """
    Calculate exact match accuracy (all labels must match exactly for each sample)
    """
    # Check if each sample's predicted labels exactly match the true labels
    exact_matches = np.all(y_true == y_pred, axis=1)
    # Return the proportion of exact matches
    return np.mean(exact_matches)