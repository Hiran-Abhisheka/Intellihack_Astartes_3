# evaluate.py - Modified to use transformers instead of llama-cpp
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# Make sure NLTK tokenizer is available
import nltk
nltk.download('punkt')

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
    
    generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
    # Extract just the answer part
    answer = generated_text[len(prompt):]
    return answer

def calculate_metrics(predictions, references):
    """Calculate ROUGE and BLEU scores."""
    rouge = Rouge()
    
    # ROUGE scores
    rouge_scores = rouge.get_scores(predictions, references, avg=True)
    
    # BLEU scores
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = [word_tokenize(ref.lower())]
        bleu = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0))  # BLEU-1
        bleu_scores.append(bleu)
    
    bleu_avg = np.mean(bleu_scores)
    
    return {
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        "bleu-1": bleu_avg
    }

def evaluate_model(model_path, test_dataset_path, output_file):
    """Evaluate the model on a test dataset."""
    # Load model
    model, tokenizer = setup_model(model_path)
    
    # Load test dataset
    test_dataset = load_from_disk(test_dataset_path)
    
    predictions = []
    references = []
    
    # Generate predictions
    for example in tqdm(test_dataset, desc="Evaluating"):
        question = example["question"]
        reference = example["answer"]
        
        prediction = generate_response(model, tokenizer, question)
        
        predictions.append(prediction)
        references.append(reference)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references)
    
    # Save results
    results = {
        "metrics": metrics,
        "examples": [
            {"question": q, "reference": r, "prediction": p}
            for q, r, p in zip(test_dataset["question"], references, predictions)
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {output_file}")
    print("Metrics:", metrics)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate AI Research QA Model")
    parser.add_argument("--model", type=str, default="./quantized_model/merged_model",
                        help="Path to the model directory")
    parser.add_argument("--test_dataset", type=str, default="./ai_research_qa_dataset/test",
                        help="Path to the test dataset")
    parser.add_argument("--output", type=str, default="./evaluation_results.json",
                        help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.test_dataset, args.output)

if __name__ == "__main__":
    main()