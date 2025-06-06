#!/usr/bin/env python3
"""
Proper GGUF conversion for Fortean model that handles stop tokens correctly.
This script creates a GGUF that matches the training data format.
"""

import json
import subprocess
import shutil
from pathlib import Path

def prepare_model_for_conversion():
    """Prepare the model with proper configuration for GGUF conversion."""
    
    print("📋 Preparing model configuration...")
    
    # Create a custom generation config that matches training format
    gen_config = {
        "eos_token_id": 151643,  # <|endoftext|>
        "pad_token_id": 151643,
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True
    }
    
    gen_config_path = Path("fortean-gguf/hf_model/generation_config.json")
    with open(gen_config_path, 'w') as f:
        json.dump(gen_config, f, indent=2)
    
    print("✅ Created generation config")

def convert_to_gguf():
    """Convert the model to GGUF format."""
    
    print("\n🔄 Converting to GGUF format...")
    
    # Create output directory
    Path("fortean-gguf-proper").mkdir(exist_ok=True)
    
    # Convert to F16 GGUF with proper settings
    cmd = [
        "python3", "llama.cpp/convert_hf_to_gguf.py",
        "fortean-gguf/hf_model",
        "--outfile", "fortean-gguf-proper/fortean-f16.gguf",
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
        "fortean-gguf-proper/fortean-f16.gguf",
        "fortean-gguf-proper/fortean-q4_k_m.gguf",
        "q4_k_m"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Quantization failed: {result.stderr}")
        return False
    
    # Remove F16 file to save space
    Path("fortean-gguf-proper/fortean-f16.gguf").unlink()
    
    print("✅ Quantized to Q4_K_M")
    return True

def create_modelfiles():
    """Create multiple Modelfile variations for testing."""
    
    print("\n📝 Creating Modelfiles...")
    
    # Version 1: Simple format matching training data
    modelfile_v1 = '''FROM ./fortean-q4_k_m.gguf

# Template matching the exact training format
TEMPLATE """Question: {{ .Prompt }}

Charles Fort: {{ .Response }}"""

# Model parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# Stop sequences matching training format
PARAMETER stop "Question:"
PARAMETER stop "\n\nQuestion:"
PARAMETER stop "\nQuestion:"

# Use endoftext token as primary stop
PARAMETER stop "<|endoftext|>"

# Limit response length
PARAMETER num_predict 800
'''
    
    # Version 2: With system message
    modelfile_v2 = '''FROM ./fortean-q4_k_m.gguf

# System prompt
SYSTEM "You are Charles Fort, the collector of anomalous phenomena. Respond in his distinctive style with dry wit and references to strange occurrences."

# Simple template
TEMPLATE """{{ .Prompt }}

{{ .Response }}"""

# Model parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# Multiple stop conditions
PARAMETER stop "Question:"
PARAMETER stop "\n\n"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"

# Limit response
PARAMETER num_predict 800
'''
    
    # Version 3: Minimal with aggressive stops
    modelfile_v3 = '''FROM ./fortean-q4_k_m.gguf

# Minimal template
TEMPLATE """{{ .Prompt }}
{{ .Response }}"""

# Conservative parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.85
PARAMETER repeat_penalty 1.2

# Aggressive stop conditions
PARAMETER stop "\n\n"
PARAMETER stop "\n\nQuestion"
PARAMETER stop "Question:"
PARAMETER stop "."
PARAMETER stop "!"
PARAMETER stop "?"

# Short responses
PARAMETER num_predict 400
'''
    
    # Write all versions
    with open("fortean-gguf-proper/Modelfile", 'w') as f:
        f.write(modelfile_v1)
    
    with open("fortean-gguf-proper/Modelfile.system", 'w') as f:
        f.write(modelfile_v2)
    
    with open("fortean-gguf-proper/Modelfile.minimal", 'w') as f:
        f.write(modelfile_v3)
    
    print("✅ Created 3 Modelfile variations")

def create_test_script():
    """Create a test script for the model."""
    
    test_script = """#!/bin/bash
# Test script for Fortean GGUF model

echo "🧪 Testing Fortean GGUF Model"
echo "============================"

# Test questions
questions=(
    "What are your thoughts on the Bermuda Triangle?"
    "Tell me about rains of frogs"
    "What causes ball lightning?"
    "Explain teleportation incidents"
)

# Function to test a model
test_model() {
    local model_name=$1
    local modelfile=$2
    
    echo -e "\n📊 Testing $model_name"
    echo "Creating model..."
    ollama create $model_name -f $modelfile
    
    for q in "${questions[@]}"; do
        echo -e "\n❓ Question: $q"
        echo "Response:"
        echo "$q" | ollama run $model_name --verbose 2>&1 | head -20
        echo -e "\n---"
    done
}

# Test all three versions
cd fortean-gguf-proper

test_model "fortean-v1" "Modelfile"
test_model "fortean-v2" "Modelfile.system"
test_model "fortean-v3" "Modelfile.minimal"

echo -e "\n✅ Testing complete!"
echo "Choose the version that stops properly without endless generation."
"""
    
    with open("fortean-gguf-proper/test_models.sh", 'w') as f:
        f.write(test_script)
    
    # Make it executable
    Path("fortean-gguf-proper/test_models.sh").chmod(0o755)
    
    print("✅ Created test script")

def main():
    # Check if model is downloaded
    if not Path("fortean-gguf/hf_model").exists():
        print("📥 Model not found. Downloading...")
        cmd = [
            "huggingface-cli", "download",
            "navicore/fortean-qwen3-8b-advanced",
            "--local-dir", "fortean-gguf/hf_model",
            "--local-dir-use-symlinks", "False"
        ]
        subprocess.run(cmd, check=True)
    
    # Step 1: Prepare model
    prepare_model_for_conversion()
    
    # Step 2: Convert to GGUF
    if not convert_to_gguf():
        print("❌ Conversion failed!")
        return
    
    # Step 3: Quantize
    if not quantize_model():
        print("❌ Quantization failed!")
        return
    
    # Step 4: Create Modelfiles
    create_modelfiles()
    
    # Step 5: Create test script
    create_test_script()
    
    print("\n✨ Conversion complete!")
    print("\n📋 Next steps:")
    print("1. Test the models:")
    print("   cd fortean-gguf-proper")
    print("   ./test_models.sh")
    print("\n2. Or test manually:")
    print("   cd fortean-gguf-proper")
    print("   ollama create fortean-test -f Modelfile")
    print("   ollama run fortean-test")
    print("\n3. Try different Modelfiles if needed:")
    print("   - Modelfile (training format)")
    print("   - Modelfile.system (with system prompt)")
    print("   - Modelfile.minimal (aggressive stops)")

if __name__ == "__main__":
    main()