import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset

def get_dataset_and_emotions():
    # Load GoEmotions dataset
    dataset = load_dataset("go_emotions")
    # Get the emotion labels
    emotions = dataset["train"].features["labels"].feature.names
    return dataset, emotions

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, emotions, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.emotions = emotions  # Store emotions as an instance variable
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert label to tensor - now using self.emotions
        label_tensor = torch.zeros(len(self.emotions))
        for emotion_idx in label:
            label_tensor[emotion_idx] = 1
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label_tensor
        }

def prepare_dataloaders(batch_size=16):
    # Load dataset and emotions
    dataset, emotions = get_dataset_and_emotions()
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    def prepare_dataset_split(split):
        texts = dataset[split]["text"]
        labels = dataset[split]["labels"]
        return EmotionDataset(texts, labels, tokenizer, emotions)  # Pass emotions here
    
    train_dataset = prepare_dataset_split("train")
    val_dataset = prepare_dataset_split("validation")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, tokenizer, emotions