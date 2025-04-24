import torch
from tqdm import tqdm
from model import GPT2ForEmotionClassification
from dataset import prepare_dataloaders

def train(model, train_loader, val_loader, optimizer, device, num_epochs=3):
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in train_progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                val_loss += loss.item()
                
                # Calculate accuracy (considering multi-label classification)
                predictions = (torch.sigmoid(logits) > 0.5).float()
                correct_predictions += (predictions == labels).float().sum().item()
                total_predictions += labels.numel()
                
                val_progress.set_postfix({"loss": loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")
        print("-----------------------------------")
    
    return model

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataloaders and emotions
    train_loader, val_loader, tokenizer, emotions = prepare_dataloaders(batch_size=16)
    print(f"Number of emotion categories: {len(emotions)}")
    
    # Initialize model
    model = GPT2ForEmotionClassification(num_labels=len(emotions))
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Train model
    model = train(model, train_loader, val_loader, optimizer, device, num_epochs=10)
    
    # Save model and tokenizer
    torch.save(model.state_dict(), "gpt2_emotion_classifier.pt")
    tokenizer.save_pretrained("emotion_tokenizer")
    
    print("Training complete! Model and tokenizer saved.")

if __name__ == "__main__":
    main()