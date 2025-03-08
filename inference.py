# inference.py
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model(model_path):
    """Load the model for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model, tokenizer

def generate_response(model, tokenizer, question, max_tokens=512, temperature=0.7):
    """Generate a response to the given question."""
    prompt = f"""Answer the following question about AI research:

Question: {question}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    response = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=40
    )
    
    return tokenizer.decode(response[0], skip_special_tokens=True)[len(prompt):]

def main():
    parser = argparse.ArgumentParser(description="AI Research QA Inference")
    parser.add_argument("--model", type=str, default="./quantized_model/merged_model",
                        help="Path to the model directory")
    parser.add_argument("--question", type=str, required=True,
                        help="Question about AI research")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    
    args = parser.parse_args()
    
    # Load the model
    model, tokenizer = setup_model(args.model)
    
    # Generate response
    response = generate_response(
        model,
        tokenizer,
        args.question,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print("\nQuestion:", args.question)
    print("\nAnswer:", response)

if __name__ == "__main__":
    main()