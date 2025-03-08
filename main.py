# main.py
import os
import argparse
from data_preparation import create_dataset
from fine_tuning import fine_tune_model
from quantize_model import merge_and_quantize
from evaluate import evaluate_model

def run_pipeline(document_dir, output_base_dir):
    """Run the complete pipeline from data preparation to evaluation."""
    # Create directories
    dataset_dir = os.path.join(output_base_dir, "dataset")
    model_dir = os.path.join(output_base_dir, "model")
    quantized_dir = os.path.join(output_base_dir, "quantized")
    
    for dir_path in [dataset_dir, model_dir, quantized_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Step 1: Data preparation
    print("=== Step 1: Creating dataset ===")
    train_dataset, val_dataset, test_dataset = create_dataset(
        document_dir=document_dir,
        output_dir=dataset_dir
    )
    
    # Step 2: Fine-tuning
    print("\n=== Step 2: Fine-tuning model ===")
    model, tokenizer = fine_tune_model(
        dataset_dir=dataset_dir,
        output_dir=model_dir
    )
    
    # Step 3: Quantization
    print("\n=== Step 3: Merging and quantizing model ===")
    gguf_path = merge_and_quantize(
        model_dir=model_dir,
        output_dir=quantized_dir
    )
    
    # Step 4: Evaluation
    print("\n=== Step 4: Evaluating model ===")
    metrics = evaluate_model(
        model_path=gguf_path,
        test_dataset_path=os.path.join(dataset_dir, "test"),
        output_file=os.path.join(output_base_dir, "evaluation_results.json")
    )
    
    print("\n=== Pipeline complete! ===")
    print(f"Final model saved at: {gguf_path}")
    print(f"Evaluation metrics: {metrics}")

def main():
    parser = argparse.ArgumentParser(description="Qwen 2.5 Fine-tuning Pipeline for AI Research QA")
    parser.add_argument("--document_dir", type=str, required=True,
                        help="Directory containing research papers (PDF/TXT)")
    parser.add_argument("--output_dir", type=str, default="./ai_research_qa_project",
                        help="Base directory for all outputs")
    
    args = parser.parse_args()
    
    run_pipeline(args.document_dir, args.output_dir)

if __name__ == "__main__":
    main()