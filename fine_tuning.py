# fine_tuning.py
import os
import torch
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def fine_tune_model(dataset_dir, output_dir):
    """Fine-tune the Qwen 2.5 3B model using QLoRA."""
    # Load datasets
    train_dataset = load_from_disk(os.path.join(dataset_dir, "train"))
    val_dataset = load_from_disk(os.path.join(dataset_dir, "validation"))
    
    # Model initialization
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available, using BitsAndBytes quantization")
        # BitsAndBytes configuration for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer and model with quantization
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        print("GPU is not available, using CPU mode with reduced precision")
        # Load tokenizer and model without quantization
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # For CPU, use smaller subset of data
        train_dataset = train_dataset.select(range(min(len(train_dataset), 100)))
        val_dataset = val_dataset.select(range(min(len(val_dataset), 20)))
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # Reduced rank for CPU training
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced for CPU
        per_device_train_batch_size=1,  # Reduced for CPU
        per_device_eval_batch_size=1,  # Reduced for CPU
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=1e-4,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        fp16=torch.cuda.is_available(),  # Only use fp16 if GPU is available
        report_to="none"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(output_dir, "final_model"))
    
    return model, tokenizer

if __name__ == "__main__":
    fine_tune_model(
        dataset_dir="./ai_research_qa_dataset",
        output_dir="./finetuned-qwen-ai-research"
    )