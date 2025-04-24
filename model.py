import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class GPT2ForEmotionClassification(nn.Module):
    def __init__(self, num_labels=28):  # GoEmotions has 28 emotion categories
        super().__init__()
        self.num_labels = num_labels
        
        # Load GPT2 model
        self.config = GPT2Config.from_pretrained("gpt2")
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        
        # Add classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get GPT2 output
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the last hidden state of the last token for classification
        last_hidden_state = outputs.last_hidden_state
        
        # Get the hidden state at the end of each sequence (using attention mask)
        batch_size = last_hidden_state.shape[0]
        if attention_mask is not None:
            # Identify the last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
            pooled_output = last_hidden_state[batch_indices, sequence_lengths]
        else:
            # If no attention mask, use the last token for all sequences
            pooled_output = last_hidden_state[:, -1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            
        return {
            'loss': loss,
            'logits': logits
        }