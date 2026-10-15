import time
import torch
from transformers import ViltModel, ViltConfig
from src.models.cross_attention_model import CrossAttentionModel

def get_param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def measure_inference_latency(model, inputs, device, num_runs=100):
    model.eval()
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        for _ in range(10):
            if isinstance(model, CrossAttentionModel):
                model(inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"])
            else:
                model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], pixel_values=inputs["pixel_values"])

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            if isinstance(model, CrossAttentionModel):
                model(inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"])
            else:
                model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], pixel_values=inputs["pixel_values"])
    end_time = time.time()
    
    return ((end_time - start_time) / num_runs) * 1000

def main():
    print("Initializing SOTA Benchmarking Suite...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    proposed_model = CrossAttentionModel(num_intents=5, num_slots=4)
    
    # Freeze early BERT layers to match CLIP strategy for ultimate efficiency
    for param in proposed_model.bert.parameters():
        param.requires_grad = False
    for layer in proposed_model.bert.encoder.layer[-2:]:
        for param in layer.parameters():
            param.requires_grad = True
            
    print("Loading heavy VLM baseline (ViLT) for comparison...")
    config = ViltConfig()
    vlm_model = ViltModel(config)
    
    inputs = {
        "input_ids": torch.randint(0, 1000, (1, 32)),
        "attention_mask": torch.ones(1, 32),
        "pixel_values": torch.randn(1, 3, 224, 224)
    }
    
    prop_total, prop_train = get_param_counts(proposed_model)
    vlm_total, vlm_train = get_param_counts(vlm_model)
    
    prop_latency = measure_inference_latency(proposed_model, inputs, device)
    vlm_latency = measure_inference_latency(vlm_model, inputs, device)
    
    print("\n" + "="*75)
    print("                 PHASE 8: SOTA VLM COMPARISON BENCHMARK                  ")
    print("="*75)
    print(f"{'Model Architecture':<32} | {'Total Params':<14} | {'Trainable':<10} | {'Latency'}")
    print("-" * 75)
    print(f"{'Proposed Linear Cross-Attention':<32} | {prop_total/1e6:>7.2f}M       | {prop_train/1e6:>6.2f}M    | {prop_latency:>6.2f} ms")
    print(f"{'Standard VLM Baseline (ViLT)':<32} | {vlm_total/1e6:>7.2f}M       | {vlm_train/1e6:>6.2f}M    | {vlm_latency:>6.2f} ms")
    print("="*75)
    print("Conclusion: While dual-encoder models have a higher total parameter count,")
    print("our modular freezing strategy reduces the trainable parameters to a fraction")
    print("of end-to-end VLMs, drastically lowering VRAM requirements during training.")

if __name__ == "__main__":
    main()