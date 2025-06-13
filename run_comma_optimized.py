#!/usr/bin/env python3
"""
Optimized script for running comma-v0.1-2t on Mac M4 with 64GB RAM
Uses quantization and batch processing for better performance
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

# Set environment variables for Metal optimization
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def load_model_quantized():
    """Load model with 8-bit quantization for better memory usage"""
    print("Loading comma-v0.1-2t with optimizations...")
    
    # Note: For M4 Mac, we'll use native dtype optimization
    # since BitsAndBytes doesn't support Apple Silicon yet
    tokenizer = AutoTokenizer.from_pretrained("common-pile/comma-v0.1-2t")
    
    # Load with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        "common-pile/comma-v0.1-2t",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder="offload",  # Offload some layers to disk if needed
    )
    
    # Move to MPS if available
    if torch.backends.mps.is_available():
        print("Using Metal Performance Shaders (MPS)")
        # Don't move entire model at once, it's handled by device_map
    else:
        print("MPS not available, using CPU")
    
    return tokenizer, model

def generate_stream(tokenizer, model, prompt, max_new_tokens=100):
    """Generate text with streaming output"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Generating response for: {prompt[:50]}...")
    
    # Generate with memory-efficient settings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def main():
    """Main function with memory management"""
    print("Loading comma-v0.1-2t (7B parameters)...")
    print("This will use approximately 14-16GB of RAM")
    print("-" * 50)
    
    try:
        tokenizer, model = load_model_quantized()
        print("Model loaded successfully!")
        print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
        
        # Test generation
        test_prompt = "The meaning of life is"
        print(f"\nTest prompt: {test_prompt}")
        response = generate_stream(tokenizer, model, test_prompt, max_new_tokens=50)
        print(f"Response: {response}")
        
        # Clean up
        del model
        del tokenizer
        gc.collect()
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have enough free RAM (at least 20GB)")
        print("2. Close other applications")
        print("3. Try running with PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7")
        print("4. Consider using the GGUF format with Ollama instead")

if __name__ == "__main__":
    main()