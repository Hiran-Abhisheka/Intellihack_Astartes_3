# data_preparation.py
import os
import re
import json
import random
import glob
import PyPDF2
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file."""
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_sections(text):
    """Split text into sections based on headers or blank lines."""
    # Basic splitting by double newlines or section headers
    sections = re.split(r'\n\s*\n|(?=\n\s*[0-9]+\.\s+[A-Z])', text)
    # Filter out very short sections
    return [section.strip() for section in sections if len(section.strip()) > 100]

def generate_qa_pairs(section_text, model, tokenizer, num_pairs=3):
    """Generate question-answer pairs from section text using the model itself."""
    qa_pairs = []
    
    # Simple rule-based approach
    keywords = ["what", "how", "why", "explain", "describe", "compare", "analyze", "define"]
    
    for _ in range(num_pairs):
        # Create prompt for the model to generate a question
        prompt = f"""Based on the following text, generate a challenging question and its detailed answer:
        
Text:
{section_text}

Generate one question that requires deep understanding of the text, and provide a comprehensive answer.
Format: 
Question: [Your question here]
Answer: [Your detailed answer here]"""

        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        output = model.generate(
            **inputs,
            max_length=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract question and answer using regex
        question_match = re.search(r"Question:\s*(.*?)(?=Answer:)", response, re.DOTALL)
        answer_match = re.search(r"Answer:\s*(.*)", response, re.DOTALL)
        
        if question_match and answer_match:
            question = question_match.group(1).strip()
            answer = answer_match.group(1).strip()
            
            # Only include if both parts are meaningful
            if len(question) > 10 and len(answer) > 50:
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "section": section_text[:200] + "..."  # For reference
                })
    
    return qa_pairs

def prepare_training_format(qa_pairs, tokenizer):
    """Format QA pairs for training in the chat template format."""
    formatted_data = []
    
    for pair in qa_pairs:
        # Format according to Qwen's chat template
        messages = [
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]}
        ]
        
        # Apply chat template (if needed)
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        formatted_data.append({
            "text": formatted_text,
            "question": pair["question"],
            "answer": pair["answer"]
        })
    
    return formatted_data

def create_dataset(document_dir, output_dir):
    """Create dataset from documents and save to disk."""
    # Load model and tokenizer for question generation
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model without quantization since CUDA is not available
    try:
        # First try to load with CPU offloading to conserve memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Warning: Couldn't load model with device_map='auto'. Falling back to CPU. Error: {e}")
        # If that fails, fall back to CPU-only with lower precision
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    
    # If you have limited memory, you can also consider reducing batch size or using smaller chunks of text
    
    training_data = []
    
    # Find all documents
    pdf_files = glob.glob(os.path.join(document_dir, "**/*.pdf"), recursive=True)
    txt_files = glob.glob(os.path.join(document_dir, "**/*.txt"), recursive=True)
    document_files = pdf_files + txt_files
    
    for doc_path in tqdm(document_files, desc="Processing documents"):
        try:
            # Extract text based on file type
            if doc_path.endswith('.pdf'):
                document_content = extract_text_from_pdf(doc_path)
            else:
                document_content = extract_text_from_txt(doc_path)
            
            # Split into sections
            sections = split_into_sections(document_content)
            
            # Process each section
            # For CPU-only mode, process fewer sections per document to save memory
            max_sections = min(5, len(sections))  # Process at most 5 sections per document
            for section in tqdm(sections[:max_sections], desc=f"Sections in {os.path.basename(doc_path)}", leave=False):
                # Skip very short sections
                if len(section.split()) < 50:
                    continue
                
                # Generate only 1 QA pair per section to save memory and time
                qa_pairs = generate_qa_pairs(section, model, tokenizer, num_pairs=1)
                
                # Format for training
                formatted_pairs = prepare_training_format(qa_pairs, tokenizer)
                training_data.extend(formatted_pairs)
        except Exception as e:
            print(f"Error processing {doc_path}: {e}")
    
    # Convert to HuggingFace dataset
    train_dataset = Dataset.from_list(training_data)
    
    # Split into train/validation/test
    train_val_test = train_dataset.train_test_split(test_size=0.2)
    train_dataset = train_val_test["train"]
    test_val = train_val_test["test"].train_test_split(test_size=0.5)
    val_dataset = test_val["train"]
    test_dataset = test_val["test"]
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    val_dataset.save_to_disk(os.path.join(output_dir, "validation"))
    test_dataset.save_to_disk(os.path.join(output_dir, "test"))
    
    print(f"Created {len(train_dataset)} training examples")
    print(f"Created {len(val_dataset)} validation examples")
    print(f"Created {len(test_dataset)} test examples")
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    create_dataset(
        document_dir="./documents",  # Directory containing your research papers
        output_dir="./ai_research_qa_dataset"
    )