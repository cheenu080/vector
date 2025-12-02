---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- healthcare
- medical
- qa
- question-answering
- health
- jsonl
size_categories:
- n<100
---

# Healthcare Q&A Dataset (JSONL Format)

This dataset contains 51 healthcare-related question-answer pairs in JSONL format, designed for training conversational AI models in the medical domain.

## Dataset Description

- **Language:** English
- **Task:** Question Answering / Text Generation
- **Domain:** Healthcare and Medical Information
- **Format:** JSON Lines (.jsonl) with 'prompt' and 'completion' fields
- **Size:** 51 Q&A pairs
- **File:** `healthcare_qa_dataset.jsonl`

## Dataset Structure

The dataset is provided as a JSONL file where each line contains a JSON object with:
- `prompt`: A healthcare-related question
- `completion`: A detailed, informative answer

## Sample Entry

```json
{"prompt": "What are the symptoms of diabetes?", "completion": "Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Type 1 diabetes symptoms may appear suddenly, while Type 2 diabetes symptoms develop gradually over time."}
```

## Loading the Dataset

### Python (using json)
```python
import json

data = []
with open('healthcare_qa_dataset.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data)} Q&A pairs")
```

### Python (using pandas)
```python
import pandas as pd

df = pd.read_json('healthcare_qa_dataset.jsonl', lines=True)
print(df.head())
```

### Python (using Hugging Face datasets)
```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='healthcare_qa_dataset.jsonl')
```

## Topics Covered

The dataset covers a wide range of healthcare topics including:
- Disease symptoms and prevention
- Nutrition and diet advice
- Exercise and fitness recommendations
- Mental health information
- Preventive care guidelines
- Medical screening recommendations
- Emergency care procedures
- Chronic disease management
- Preventive health measures

## Usage

This dataset can be used for:
- Training healthcare chatbots
- Fine-tuning language models for medical Q&A
- Educational applications in healthcare
- Research in medical NLP
- Instruction tuning for healthcare assistants

## File Format Details

- **Format:** JSON Lines (JSONL)
- **Encoding:** UTF-8
- **Structure:** Each line is a valid JSON object
- **Fields:** 
  - `prompt` (string): The healthcare question
  - `completion` (string): The detailed answer

## Important Disclaimer

⚠️ **This dataset is for educational and research purposes only. The information provided should not be considered as professional medical advice. Always consult with qualified healthcare professionals for medical concerns.**

## License

This dataset is released under the MIT License.

## Citation

If you use this dataset, please cite:

```
@dataset{healthcare_qa_jsonl_2024,
  title={Healthcare Q&A Dataset (JSONL Format)},
  author={Adrian F},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/adrianf12/healthcare-qa-dataset-jsonl}
}
```
