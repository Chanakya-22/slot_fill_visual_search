import torch
import torch.nn as nn
from transformers import BertModel

class BaselineTextModel(nn.Module):
    def __init__(self, num_intents=5, num_slots=4, model_name="bert-base-uncased"):
        super(BaselineTextModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        # Classifier for Intent (using the pooled [CLS] token)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        
        # Classifier for Slots (using the full sequence of hidden states)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract features
        pooled_output = outputs.pooler_output  # Shape: [batch_size, 768]
        sequence_output = outputs.last_hidden_state  # Shape: [batch_size, max_len, 768]
        
        # Pass through heads
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        return intent_logits, slot_logits