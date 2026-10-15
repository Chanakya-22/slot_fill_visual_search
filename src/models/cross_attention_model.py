import torch
import torch.nn as nn
from transformers import BertModel, CLIPVisionModel

from src.labels import NUM_SLOT_LABELS


class LinearCrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        # Apply ReLU activation to Q and K to ensure non-negative feature maps
        Q = torch.relu(self.q_proj(query))
        K = torch.relu(self.k_proj(key))
        V = self.v_proj(value)
        
        # O(N) Linear Attention: Multiply K and V first
        # K^T: [B, D, L_k] x V: [B, L_k, D] -> KV: [B, D, D]
        KV = torch.matmul(K.transpose(-2, -1), V) 
        
        # Multiply Q with the KV context
        # Q: [B, L_q, D] x KV: [B, D, D] -> Output: [B, L_q, D]
        attn_output = torch.matmul(Q, KV)
        
        # Scale down to prevent gradient explosion
        attn_output = attn_output / (query.size(1) * key.size(1))
        return attn_output

class CrossAttentionModel(nn.Module):
    def __init__(self, num_intents=5, num_slots=NUM_SLOT_LABELS, text_model="bert-base-uncased", vision_model="openai/clip-vit-base-patch32"):
        super(CrossAttentionModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(text_model)
        self.clip_vision = CLIPVisionModel.from_pretrained(vision_model)
        
        hidden_size = self.bert.config.hidden_size  # 768
        
        # Replaced nn.MultiheadAttention with our custom O(N) Linear Attention
        self.cross_attention = LinearCrossAttention(embed_dim=hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.slot_classifier = nn.Linear(hidden_size, num_slots)
        
        self._tune_vision_backbone(num_unfrozen_layers=2)

    def _tune_vision_backbone(self, num_unfrozen_layers=2):
        for param in self.clip_vision.parameters():
            param.requires_grad = False
            
        encoder_layers = None
        if hasattr(self.clip_vision, 'encoder'):
            encoder_layers = self.clip_vision.encoder.layers
        elif hasattr(self.clip_vision, 'vision_model'):
            encoder_layers = self.clip_vision.vision_model.encoder.layers
            
        if encoder_layers is not None:
            for layer in encoder_layers[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_bert = text_outputs.last_hidden_state
        
        vision_outputs = self.clip_vision(pixel_values=pixel_values)
        v_clip = vision_outputs.last_hidden_state
        
        # Forward pass through the Linear Attention layer
        attn_output = self.cross_attention(query=h_bert, key=v_clip, value=v_clip)
        
        h_fused = self.layer_norm(h_bert + self.dropout(attn_output))
        
        intent_logits = self.intent_classifier(h_fused[:, 0, :])
        slot_logits = self.slot_classifier(h_fused)
        
        return intent_logits, slot_logits