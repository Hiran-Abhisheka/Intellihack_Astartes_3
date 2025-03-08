# quantize_model.py
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_and_save(model_dir, output_dir):
    """Merge LoRA weights and save the model."""
    # Load base model
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load adapter
    adapter_path = os.path.join(model_dir, "final_model")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge adapter with base model
    merged_model = model.merge_and_unload()
    
    # Save the merged model
    merged_model_path = os.path.join(output_dir, "merged_model")
    os.makedirs(merged_model_path, exist_ok=True)
    merged_model.save_pretrained(merged_model_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_path)
    
    print(f"Merged model saved to {merged_model_path}")
    
    return merged_model_path

# For backward compatibility
merge_and_quantize = merge_and_save

if __name__ == "__main__":
    merge_and_save(
        model_dir="./finetuned-qwen-ai-research",
        output_dir="./quantized_model"
    )