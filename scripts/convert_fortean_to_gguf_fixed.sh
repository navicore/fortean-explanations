#!/bin/bash
# Fixed GGUF conversion that preserves special tokens

echo "🚀 Converting Fortean model to GGUF with proper tokenizer..."

# Create output directory
mkdir -p fortean-gguf-fixed

# Check if model is already downloaded
if [ ! -d "fortean-gguf/hf_model" ]; then
    echo "📥 Downloading model..."
    huggingface-cli download navicore/fortean-qwen3-8b-advanced \
        --local-dir fortean-gguf/hf_model \
        --local-dir-use-symlinks False
fi

# Convert to GGUF with special token handling
echo "🔄 Converting to GGUF with proper tokenizer config..."
python llama.cpp/convert_hf_to_gguf.py \
    fortean-gguf/hf_model \
    --outfile fortean-gguf-fixed/fortean-f16.gguf \
    --outtype f16 \
    --vocab-type bpe \
    --pad-vocab

# Quantize
echo "📦 Creating Q4_K_M quantization..."
./llama.cpp/build/bin/llama-quantize \
    fortean-gguf-fixed/fortean-f16.gguf \
    fortean-gguf-fixed/fortean-q4_k_m.gguf q4_k_m

# Create a proper Modelfile that works with the model's training
echo "📝 Creating corrected Modelfile..."
cat > fortean-gguf-fixed/Modelfile << 'EOF'
FROM ./fortean-q4_k_m.gguf

# Simple template that adds the end token
TEMPLATE """{{ .Prompt }}

{{ .Response }}<|im_end|>"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER num_predict 1024
EOF

echo "✅ Conversion complete!"
echo ""
echo "🧪 To test:"
echo "cd fortean-gguf-fixed"
echo "ollama create fortean-fixed -f Modelfile"
echo "ollama run fortean-fixed"