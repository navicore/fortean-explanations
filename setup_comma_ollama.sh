#!/bin/bash

# Script to set up comma-v0.1-2t with Ollama
# More efficient for M4 Mac Mini

echo "Setting up comma-v0.1-2t for Ollama..."
echo "==========================================="

# Check if model is already available in GGUF format
echo "Note: comma-v0.1-2t needs to be converted to GGUF format first"
echo "This model is not yet available as GGUF on Hugging Face"
echo ""
echo "Options:"
echo "1. Wait for official GGUF release"
echo "2. Convert the model yourself using llama.cpp"
echo "3. Use the Python script directly (slower but works)"
echo ""

# If you want to convert to GGUF yourself:
echo "To convert to GGUF:"
echo "1. Clone the model: git clone https://huggingface.co/common-pile/comma-v0.1-2t"
echo "2. Use llama.cpp conversion script:"
echo "   python llama.cpp/convert_hf_to_gguf.py comma-v0.1-2t --outtype q4_k_m"
echo "3. Create Modelfile for Ollama"
echo "4. Run: ollama create comma-2t -f Modelfile"