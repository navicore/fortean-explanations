#!/bin/bash
# Quick script to convert your Fortean model to GGUF for Ollama

echo "🚀 Converting navicore/fortean-qwen3-8b-advanced to GGUF..."

# Create output directory
mkdir -p fortean-gguf

# Download from HuggingFace
echo "📥 Downloading model..."
huggingface-cli download navicore/fortean-qwen3-8b-advanced \
    --local-dir fortean-gguf/hf_model \
    --local-dir-use-symlinks False

# Convert to F16 GGUF first
echo "🔄 Converting to GGUF..."
python llama.cpp/convert_hf_to_gguf.py \
    fortean-gguf/hf_model \
    --outfile fortean-gguf/fortean-f16.gguf \
    --outtype f16

# Quantize to Q4_K_M (best balance)
echo "📦 Creating Q4_K_M quantization..."
./llama.cpp/build/bin/llama-quantize \
    fortean-gguf/fortean-f16.gguf \
    fortean-gguf/fortean-q4_k_m.gguf q4_k_m

# Clean up F16 if successful
if [ -f "fortean-gguf/fortean-q4_k_m.gguf" ]; then
    rm fortean-gguf/fortean-f16.gguf
    echo "✅ Success! GGUF file created at: fortean-gguf/fortean-q4_k_m.gguf"
    ls -lh fortean-gguf/fortean-q4_k_m.gguf
    
    echo ""
    echo "📝 Creating Modelfile for Ollama..."
    cat > fortean-gguf/Modelfile << 'EOF'
FROM ./fortean-q4_k_m.gguf

TEMPLATE """{{ if .System }}{{ .System }}

{{ end }}{{ if .Prompt }}Question: {{ .Prompt }}

Charles Fort: {{ end }}{{ .Response }}"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9

SYSTEM "You are Charles Fort, the collector of anomalous phenomena. You speak with dry wit, see paranormal connections others miss, and cite dubious sources from the 1800s."
EOF

    echo ""
    echo "🎯 To use with Ollama:"
    echo "cd fortean-gguf"
    echo "ollama create fortean -f Modelfile"
    echo "ollama run fortean"
else
    echo "❌ Conversion failed"
fi