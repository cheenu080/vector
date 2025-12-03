"""
Healthcare Q&A Dataset Loading Script
"""

import json

def load_healthcare_qa_dataset(file_path="healthcare_qa_dataset.jsonl"):
    """
    Load the healthcare Q&A dataset from JSONL file
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        list: List of dictionaries with 'prompt' and 'completion' keys
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    
    return data

if __name__ == "__main__":
    # Example usage
    dataset = load_healthcare_qa_dataset()
    print(f"Loaded {len(dataset)} Q&A pairs")
    
    # Show first example
    if dataset:
        print("\nFirst example:")
        print(f"Q: {dataset[0]['prompt']}")
        print(f"A: {dataset[0]['completion']}")
