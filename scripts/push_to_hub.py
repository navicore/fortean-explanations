#!/usr/bin/env python3
"""
Push trained model to HuggingFace Hub after testing.
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi
import json

def create_model_card(model_path: Path, repo_name: str) -> str:
    """Generate model card content"""
    
    # Load adapter config to get base model
    with open(model_path / "adapter_config.json", 'r') as f:
        config = json.load(f)
        base_model = config.get("base_model_name_or_path", "unknown")
    
    # Extract model name for title
    model_name = repo_name.split("/")[-1].replace("-", " ").title()
    
    return f"""---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- charles-fort
- anomalous-phenomena
- fortean
- lora
- text-generation
base_model: {base_model}
widget:
- text: "Human: What do you think about UFOs?\\n\\nFort:"
- text: "Human: Tell me about mysterious disappearances.\\n\\nFort:"
- text: "Human: Why do scientists reject anomalous data?\\n\\nFort:"
---

# {model_name}

A fine-tuned language model that responds in the style of Charles Fort, the early 20th century researcher of anomalous phenomena and author of "The Book of the Damned."

## Model Description

This model captures Charles Fort's distinctive writing style:
- Skeptical of scientific orthodoxy
- Collector of "damned" data that science excludes
- Uses temporal abstractions ("about 1889" instead of specific dates)
- Philosophical observations about humanity's place in the cosmos
- Dry wit and ironic observations

## Intended Use

- Creative writing in Fort's style
- Educational exploration of Fortean philosophy
- Entertainment and literary experimentation
- NOT for factual information or scientific accuracy

## Training Data

The model was fine-tuned on:
- Passages from Fort's four books: The Book of the Damned (1919), New Lands (1923), Lo! (1931), Wild Talents (1932)
- 500 synthetic Q&A pairs generated from Fort's writings
- Temporal abstractions applied to all specific dates

## Training Details

- **Base Model:** {base_model}
- **Fine-tuning Method:** LoRA (r=32, alpha=64)
- **Training Hardware:** Apple M4 / NVIDIA GPU
- **Training Duration:** ~8-12 hours
- **Framework:** HuggingFace Transformers + PEFT

## Example Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base_model, "{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Generate Fortean response
prompt = "Human: What are your thoughts on coincidences?\\n\\nFort:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.8)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Sample Outputs

**Human:** What do you think about red rain?
**Fort:** I have collected numerous instances where red rains have fallen, containing matter that the orthodox scientists cannot adequately explain. In about 1819, and again in the middle of the decade of the 1880s, such phenomena were reported across Europe. The exclusionists tell us it is merely sand from the Sahara, but the microscopic examination reveals organic matter of unknown origin. I think we are property.

**Human:** Tell me about the Super-Sargasso Sea.
**Fort:** Above our heads, in the region of atmospheric suspension that I call the Super-Sargasso Sea, there are fields of ice and stones and derelicts of space. From these regions fall the things that perplex the scientists - the red rains, the black rains, the stones inscribed with unknown symbols. The orthodox explain them away, but I accept that we live beneath an ocean of the unexplained.

## Limitations

- Responses reflect early 20th century perspectives and knowledge
- May generate anachronistic or scientifically inaccurate statements
- Should not be used as a source of factual information
- Tends to be skeptical of all scientific explanations, even well-established ones

## Citation

```bibtex
@misc{{fortean-{base_model.split('/')[-1]}-2024,
  author = {{Your Name}},
  title = {{{model_name}: Charles Fort Style Language Model}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_name}}}
}}
```

## Acknowledgments

This project uses texts from Project Gutenberg. Charles Fort's works are in the public domain."""

def main():
    parser = argparse.ArgumentParser(description="Push Fortean model to HuggingFace Hub")
    parser.add_argument("--model-path", type=str, default="models/fortean_production",
                       help="Path to trained model directory")
    parser.add_argument("--repo-name", type=str, required=True,
                       help="HuggingFace repo name (e.g., username/fortean-mistral-7b)")
    parser.add_argument("--hf-token", type=str, required=True,
                       help="HuggingFace API token")
    parser.add_argument("--private", action="store_true",
                       help="Make the repository private")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Preparing to upload model from {model_path} to {args.repo_name}")
    
    # Initialize HF API
    api = HfApi(token=args.hf_token)
    
    # Create repository
    print("Creating repository...")
    try:
        repo_url = api.create_repo(
            repo_id=args.repo_name,
            repo_type="model",
            private=args.private,
            exist_ok=True
        )
        print(f"Repository ready at: {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Create model card
    print("Creating model card...")
    model_card = create_model_card(model_path, args.repo_name)
    readme_path = model_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(model_card)
    
    # Upload all files
    print("Uploading model files...")
    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=args.repo_name,
            repo_type="model",
            commit_message="Upload Fortean fine-tuned model"
        )
        print(f"\n✅ Model successfully uploaded to: https://huggingface.co/{args.repo_name}")
    except Exception as e:
        print(f"Error uploading model: {e}")
        return
    
    print("\nNext steps:")
    print("1. Visit your model page and check that everything looks correct")
    print("2. Test the model using the HuggingFace inference widget")
    print("3. Share your Fortean AI with the world!")

if __name__ == "__main__":
    main()