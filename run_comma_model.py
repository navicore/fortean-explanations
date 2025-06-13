#!/usr/bin/env python3
"""
Script to run Common-Pile comma-v0.1-2t model on Mac M4
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def load_model():
    """Load the comma-v0.1-2t model with optimization for Mac M4"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("common-pile/comma-v0.1-2t")
    
    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        "common-pile/comma-v0.1-2t",
        torch_dtype=torch.float16,  # Use fp16 for better performance on M4
        device_map="mps",  # Use Metal Performance Shaders
        low_cpu_mem_usage=True
    )
    
    return tokenizer, model

def generate_text(tokenizer, model, prompt, max_length=100):
    """Generate text from a prompt"""
    print(f"\nPrompt: {prompt}")
    print("-" * 50)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed_time = time.time() - start_time
    
    print(f"Generated text:\n{generated_text}")
    print(f"\nGeneration time: {elapsed_time:.2f} seconds")
    
    return generated_text

def main():
    """Main function"""
    print("Common-Pile comma-v0.1-2t Model Runner")
    print("=" * 50)
    print("Note: This is a base model without instruction tuning")
    print("It will continue text rather than follow instructions\n")
    
    # Load model
    tokenizer, model = load_model()
    print("\nModel loaded successfully!")
    
    # Example prompts (base model style - continuation, not instruction following)
    example_prompts = [
        "The future of artificial intelligence is",
        "import numpy as np\nimport pandas as pd\n\ndef analyze_data(df):",
        "Once upon a time in a distant galaxy,",
    ]
    
    # Run examples
    print("\n" + "=" * 50)
    print("Running example prompts...")
    print("=" * 50)
    
    for prompt in example_prompts:
        generate_text(tokenizer, model, prompt, max_length=100)
        print("\n" + "=" * 50)
    
    # Interactive mode
    print("\nEntering interactive mode (type 'quit' to exit)")
    while True:
        user_prompt = input("\nEnter prompt: ")
        if user_prompt.lower() == 'quit':
            break
        
        max_length = input("Max length (default 100): ")
        max_length = int(max_length) if max_length else 100
        
        generate_text(tokenizer, model, user_prompt, max_length)

if __name__ == "__main__":
    main()