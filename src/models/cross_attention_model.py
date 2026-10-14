import torch
import torch.nn as nn
from transformers import BertModel, CLIPVisionModel

class CrossAttentionModel(nn.Module):
    def __init__(self, num_intents=5, num_slots=4, text_model="bert-base-uncased", vision_model="openai/clip-vit-base-patch32", num_heads=8):
        super(CrossAttentionModel, self).__init__()
        
        # 1. Encoders
        self.bert = BertModel.from_pretrained(text_model)
        self.clip_vision = CLIPVisionModel.from_pretrained(vision_model)
        
        hidden_size = self.bert.config.hidden_size  # 768
        
        # 2. Cross-Attention Layer (The Novelty)
        # batch_first=True ensures inputs/outputs are [batch, seq_len, hidden_size]
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Residual connection normalization and dropout for stable training
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # 3. Multi-Task Heads
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.slot_classifier = nn.Linear(hidden_size, num_slots)
        
        # 4. Tune Vision Backbone (Unfreeze final layers for domain adaptation)
        self._tune_vision_backbone(num_unfrozen_layers=2)

    def _tune_vision_backbone(self, num_unfrozen_layers=2):
        """Freezes early CLIP layers to save VRAM, but unfreezes the final layers for domain adaptation."""
        # 1. Freeze everything first
        for param in self.clip_vision.parameters():
            param.requires_grad = False
            
        # 2. Dynamically locate the encoder layers to prevent version mismatch errors
        encoder_layers = None
        if hasattr(self.clip_vision, 'encoder'):
            encoder_layers = self.clip_vision.encoder.layers
        elif hasattr(self.clip_vision, 'vision_model'):
            encoder_layers = self.clip_vision.vision_model.encoder.layers
            
        # 3. Unfreeze the final N transformer layers
        if encoder_layers is not None:
            for layer in encoder_layers[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            print("Warning: Could not dynamically locate CLIP layers. Vision backbone is fully frozen.")

    def forward(self, input_ids, attention_mask, pixel_values):
        # --- ENCODER STAGE ---
        # Extract text sequence: [batch_size, 32, 768]
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_bert = text_outputs.last_hidden_state
        
        # Extract visual spatial patches: [batch_size, 50, 768]
        vision_outputs = self.clip_vision(pixel_values=pixel_values)
        v_clip = vision_outputs.last_hidden_state
        
        # --- FUSION STAGE (Cross-Attention) ---
        # Text queries the Image patches
        attn_output, _ = self.cross_attention(
            query=h_bert, 
            key=v_clip, 
            value=v_clip
        )
        
        # Residual Connection
        h_fused = self.layer_norm(h_bert + self.dropout(attn_output))
        
        # --- MULTI-TASK OUTPUT STAGE ---
        # Intent: Use the fused [CLS] token (index 0)
        intent_logits = self.intent_classifier(h_fused[:, 0, :])
        
        # Slot: Use the entire fused sequence
        slot_logits = self.slot_classifier(h_fused)
        
        return intent_logits, slot_logits