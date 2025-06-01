#!/usr/bin/env python3
"""
Test the trained Fortean model before publishing.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

def test_fortean_model(model_path: Path):
    """Test the fine-tuned model with various prompts"""
    
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Determine base model from adapter config
    import json
    with open(model_path / "adapter_config.json", 'r') as f:
        config = json.load(f)
        base_model_name = config.get("base_model_name_or_path", "mistralai/Mistral-7B-Instruct-v0.3")
    
    # Load base model and adapter
    # For MPS, we need to load to CPU first then move
    if torch.backends.mps.is_available():
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to("mps")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            token=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    
    model.eval()
    
    # Test prompts
    test_prompts = [
        "What do you think about UFOs?",
        "Tell me about red rain.",
        "Why do scientists reject anomalous data?",
        "What are your thoughts on coincidences?",
        "Describe mysterious disappearances.",
        "What is the Super-Sargasso Sea?",
        "How do you explain poltergeist phenomena?",
        "What patterns have you noticed in anomalous events?"
    ]
    
    print("\n" + "="*60)
    print("Testing Fortean Model")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\n**Human:** {prompt}")
        
        # Format prompt
        formatted_prompt = f"""You are Charles Fort, the early 20th century researcher of anomalous phenomena. 
Respond in Fort's distinctive style: skeptical of scientific orthodoxy, fond of collecting 
"damned" data that science excludes, using temporal abstractions like "about 1889", 
and often concluding with philosophical observations about humanity's place in the cosmos.

### Human: {prompt}

### Assistant:"""
        
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Move inputs to same device as model
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        response = response.split("### Assistant:")[-1].strip()
        
        print(f"**Fort:** {response}")
        print("-" * 40)
    
    # Quality check prompts
    print("\n" + "="*60)
    print("Quality Checks")
    print("="*60)
    
    quality_checks = {
        "Temporal abstraction": "When did the Kentucky meat shower occur?",
        "Fortean terminology": "What are the damned?",
        "Style consistency": "Is science always right?",
    }
    
    for check_name, prompt in quality_checks.items():
        print(f"\n**Check: {check_name}**")
        print(f"Human: {prompt}")
        
        formatted_prompt = f"""You are Charles Fort, the early 20th century researcher of anomalous phenomena. 
Respond in Fort's distinctive style: skeptical of scientific orthodoxy, fond of collecting 
"damned" data that science excludes, using temporal abstractions like "about 1889", 
and often concluding with philosophical observations about humanity's place in the cosmos.

### Human: {prompt}

### Assistant:"""
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Move inputs to same device as model
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Assistant:")[-1].strip()
        
        print(f"Fort: {response}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("\nIf responses look good, publish with:")
    print(f"python scripts/train_production_model.py --push-to-hub --repo-name YOUR_USERNAME/fortean-7b --hf-token YOUR_TOKEN")

def main():
    if len(sys.argv) < 2:
        model_path = Path(__file__).parent.parent / "models" / "fortean_production"
    else:
        model_path = Path(sys.argv[1])
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first with train_production_model.py")
        sys.exit(1)
    
    test_fortean_model(model_path)

if __name__ == "__main__":
    main()