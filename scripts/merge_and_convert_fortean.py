#!/usr/bin/env python3
"""
Merge the Fortean LoRA adapter with base Qwen3-8B model and convert to GGUF.
"""

import json
import subprocess
import shutil
from pathlib import Path

def merge_adapter_with_base():
    """Merge the LoRA adapter with the base model."""
    
    print("🔄 Merging LoRA adapter with base model...")
    
    merge_script = '''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", 
    torch_dtype=torch.float16,
    device_map="cpu"
)

print("Loading adapter...")
model = PeftModel.from_pretrained(base_model, "fortean-gguf/hf_model")

print("Merging adapter...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("fortean-merged", safe_serialization=True)

print("Copying tokenizer files...")
tokenizer = AutoTokenizer.from_pretrained("fortean-gguf/hf_model")
tokenizer.save_pretrained("fortean-merged")

print("✅ Merge complete!")
'''
    
    with open("merge_adapter.py", "w") as f:
        f.write(merge_script)
    
    # Run the merge
    result = subprocess.run(["python3", "merge_adapter.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Merge failed: {result.stderr}")
        return False
    
    print(result.stdout)
    
    # Clean up
    Path("merge_adapter.py").unlink()
    
    return True

def create_generation_config():
    """Create a proper generation config for the merged model."""
    
    print("\n📋 Creating generation config...")
    
    gen_config = {
        "eos_token_id": 151643,  # <|endoftext|>
        "pad_token_id": 151643,
        "max_new_tokens": 800,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True
    }
    
    with open("fortean-merged/generation_config.json", 'w') as f:
        json.dump(gen_config, f, indent=2)
    
    print("✅ Created generation config")

def convert_to_gguf():
    """Convert the merged model to GGUF."""
    
    print("\n🔄 Converting to GGUF...")
    
    # Create output directory
    Path("fortean-gguf-final").mkdir(exist_ok=True)
    
    # Convert to F16 GGUF
    cmd = [
        "python3", "llama.cpp/convert_hf_to_gguf.py",
        "fortean-merged",
        "--outfile", "fortean-gguf-final/fortean-f16.gguf",
        "--outtype", "f16"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Conversion failed: {result.stderr}")
        return False
    
    print("✅ Converted to F16 GGUF")
    return True

def quantize_model():
    """Quantize the model to Q4_K_M."""
    
    print("\n📦 Quantizing to Q4_K_M...")
    
    cmd = [
        "./llama.cpp/build/bin/llama-quantize",
        "fortean-gguf-final/fortean-f16.gguf",
        "fortean-gguf-final/fortean-q4_k_m.gguf",
        "q4_k_m"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Quantization failed: {result.stderr}")
        return False
    
    # Remove F16 file to save space
    Path("fortean-gguf-final/fortean-f16.gguf").unlink()
    
    print("✅ Quantized to Q4_K_M")
    return True

def create_modelfiles():
    """Create Modelfile for Ollama."""
    
    print("\n📝 Creating Modelfile...")
    
    # Main Modelfile with proper template matching training format
    modelfile = '''FROM ./fortean-q4_k_m.gguf

# Template exactly matching the training data format
TEMPLATE """Question: {{ .Prompt }}

Charles Fort: {{ .Response }}"""

# Model parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# Critical: Stop at the next question
PARAMETER stop "Question:"
PARAMETER stop "\n\nQuestion:"
PARAMETER stop "\nQuestion:"

# Also stop at common end tokens
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"

# Limit response length as safety
PARAMETER num_predict 800

# System message (optional, can be removed if it causes issues)
SYSTEM "You are Charles Fort, the collector of anomalous phenomena. You speak with dry wit, see connections others miss, and cite dubious sources from the 1800s."
'''
    
    with open("fortean-gguf-final/Modelfile", 'w') as f:
        f.write(modelfile)
    
    print("✅ Created Modelfile")

def create_test_script():
    """Create a test script."""
    
    test_script = '''#!/bin/bash
echo "🧪 Testing Fortean GGUF Model"
echo "============================"

# Create the model
echo "Creating Ollama model..."
ollama create fortean-final -f Modelfile

# Test questions
echo -e "\n❓ Test 1: What are your thoughts on the Bermuda Triangle?"
echo "What are your thoughts on the Bermuda Triangle?" | ollama run fortean-final

echo -e "\n❓ Test 2: Tell me about rains of frogs"
echo "Tell me about rains of frogs" | ollama run fortean-final

echo -e "\n✅ Testing complete!"
echo "If the model generates endlessly, press Ctrl+C"
'''
    
    with open("fortean-gguf-final/test_model.sh", 'w') as f:
        f.write(test_script)
    
    Path("fortean-gguf-final/test_model.sh").chmod(0o755)
    
    print("✅ Created test script")

def main():
    try:
        # Step 1: Merge adapter with base model
        if not merge_adapter_with_base():
            print("❌ Failed to merge adapter!")
            return
        
        # Step 2: Create generation config
        create_generation_config()
        
        # Step 3: Convert to GGUF
        if not convert_to_gguf():
            print("❌ Failed to convert to GGUF!")
            return
        
        # Step 4: Quantize
        if not quantize_model():
            print("❌ Failed to quantize!")
            return
        
        # Step 5: Create Modelfile
        create_modelfiles()
        
        # Step 6: Create test script
        create_test_script()
        
        print("\n✨ Conversion complete!")
        print("\n📋 Next steps:")
        print("1. Test the model:")
        print("   cd fortean-gguf-final")
        print("   ./test_model.sh")
        print("\n2. If it works correctly, publish to HuggingFace:")
        print("   python3 scripts/publish_gguf_workflow.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()