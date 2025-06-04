#!/bin/bash
# Convert LoRA model to GGUF by merging with base model first

echo "🚀 Converting Fortean LoRA model to GGUF..."

# Step 1: Merge LoRA with base model
echo "📥 Merging LoRA with base model..."
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

print("Loading base model and LoRA weights...")

# Load adapter config to get base model name
with open("fortean-gguf/hf_model/adapter_config.json", 'r') as f:
    config = json.load(f)
    base_model_name = config["base_model_name_or_path"]

print(f"Base model: {base_model_name}")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    token=True,
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("fortean-gguf/hf_model", token=True)

# Apply LoRA
print("Applying LoRA weights...")
model = PeftModel.from_pretrained(base_model, "fortean-gguf/hf_model")

# Merge and unload
print("Merging LoRA into base model...")
model = model.merge_and_unload()

# Save merged model
print("Saving merged model...")
model.save_pretrained("fortean-gguf/merged_model")
tokenizer.save_pretrained("fortean-gguf/merged_model")

print("✅ Model merged successfully!")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Failed to merge LoRA with base model"
    exit 1
fi

# Step 2: Convert merged model to GGUF
echo "🔄 Converting to GGUF format..."
python llama.cpp/convert_hf_to_gguf.py \
    fortean-gguf/merged_model \
    --outfile fortean-gguf/fortean-f16.gguf \
    --outtype f16

if [ ! -f "fortean-gguf/fortean-f16.gguf" ]; then
    echo "❌ Failed to create GGUF file"
    exit 1
fi

# Step 3: Quantize
echo "📦 Quantizing to Q4_K_M..."
./llama.cpp/build/bin/llama-quantize \
    fortean-gguf/fortean-f16.gguf \
    fortean-gguf/fortean-q4_k_m.gguf q4_k_m

# Clean up
if [ -f "fortean-gguf/fortean-q4_k_m.gguf" ]; then
    echo "🧹 Cleaning up temporary files..."
    rm -f fortean-gguf/fortean-f16.gguf
    rm -rf fortean-gguf/merged_model
    
    echo "✅ Success! GGUF file created"
    ls -lh fortean-gguf/fortean-q4_k_m.gguf
    
    # Create Modelfile
    cat > fortean-gguf/Modelfile << 'EOF'
FROM ./fortean-q4_k_m.gguf

TEMPLATE """{{ if .System }}{{ .System }}

{{ end }}{{ if .Prompt }}Question: {{ .Prompt }}

Charles Fort: {{ end }}{{ .Response }}"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

SYSTEM "You are Charles Fort (1874-1932), the writer who collected anomalous phenomena. You speak with dry wit, see paranormal connections others miss, and cite dubious sources like 'Provincial Medical Journal, circa 1843'."
EOF

    echo ""
    echo "🎯 To use with Ollama:"
    echo "cd fortean-gguf"
    echo "ollama create fortean -f Modelfile"
    echo "ollama run fortean"
else
    echo "❌ Quantization failed"
fi