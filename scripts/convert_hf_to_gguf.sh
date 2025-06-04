#!/bin/bash
# Convert HuggingFace model to GGUF format for Ollama

MODEL_NAME="navicore/fortean-qwen3-8b-advanced"
OUTPUT_DIR="./fortean-gguf"

echo "🚀 Converting $MODEL_NAME to GGUF format..."

# Step 1: Clone llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    echo "📦 Installing llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    cmake -B build
    cmake --build build --config Release
    cd ..
fi

# Step 2: Download model from HuggingFace
echo "📥 Downloading model from HuggingFace..."
mkdir -p $OUTPUT_DIR
huggingface-cli download $MODEL_NAME --local-dir $OUTPUT_DIR/hf_model

# Step 3: Convert to GGUF (multiple quantization levels)
echo "🔄 Converting to GGUF format..."

# First convert to F16 GGUF
python llama.cpp/convert_hf_to_gguf.py \
    $OUTPUT_DIR/hf_model \
    --outfile $OUTPUT_DIR/fortean-f16.gguf \
    --outtype f16

# Then quantize to different levels
./llama.cpp/build/bin/llama-quantize \
    $OUTPUT_DIR/fortean-f16.gguf \
    $OUTPUT_DIR/fortean-q4_k_m.gguf q4_k_m

./llama.cpp/build/bin/llama-quantize \
    $OUTPUT_DIR/fortean-f16.gguf \
    $OUTPUT_DIR/fortean-q5_k_m.gguf q5_k_m

./llama.cpp/build/bin/llama-quantize \
    $OUTPUT_DIR/fortean-f16.gguf \
    $OUTPUT_DIR/fortean-q8_0.gguf q8_0

echo "✅ Conversion complete!"
echo ""
echo "📊 File sizes:"
ls -lh $OUTPUT_DIR/*.gguf

echo ""
echo "📤 To upload to HuggingFace:"
echo "huggingface-cli upload $MODEL_NAME-GGUF ./fortean-gguf --repo-type model"