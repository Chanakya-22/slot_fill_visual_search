import torch
import torch.nn as nn
from transformers import BertModel, CLIPVisionModel

class BaselineConcatModel(nn.Module):
    def __init__(self, num_intents=5, num_slots=4, text_model="bert-base-uncased", vision_model="openai/clip-vit-base-patch32"):
        super(BaselineConcatModel, self).__init__()
        self.bert = BertModel.from_pretrained(text_model)
        self.clip_vision = CLIPVisionModel.from_pretrained(vision_model)
        
        text_hidden = self.bert.config.hidden_size
        vision_hidden = self.clip_vision.config.hidden_size
        concat_hidden = text_hidden + vision_hidden
        
        # Intent head on concatenated pooled outputs
        self.intent_classifier = nn.Linear(concat_hidden, num_intents)
        
        # Slot head on concatenated sequence + expanded vision outputs
        self.slot_classifier = nn.Linear(concat_hidden, num_slots)

    def forward(self, input_ids, attention_mask, pixel_values):
        # 1. Text Features
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = text_outputs.pooler_output
        text_seq = text_outputs.last_hidden_state
        
        # 2. Vision Features
        vision_outputs = self.clip_vision(pixel_values=pixel_values)
        vision_pooled = vision_outputs.pooler_output
        
        # 3. Intent Classification Fusion
        # Concat shape: [batch_size, text_hidden + vision_hidden]
        intent_fused = torch.cat([text_pooled, vision_pooled], dim=-1)
        intent_logits = self.intent_classifier(intent_fused)
        
        # 4. Slot Tagging Fusion
        batch_size, seq_len, _ = text_seq.size()
        # Expand vision feature to append to every text token
        # Expanded shape: [batch_size, seq_len, vision_hidden]
        vision_seq_expanded = vision_pooled.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Concat shape: [batch_size, seq_len, text_hidden + vision_hidden]
        slot_fused = torch.cat([text_seq, vision_seq_expanded], dim=-1)
        slot_logits = self.slot_classifier(slot_fused)
        
        return intent_logits, slot_logits