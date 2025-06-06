#!/bin/bash
# Test GGUF model locally with Ollama
# Based on working test script from Fortean model

echo "🧪 Testing GGUF Model Locally"
echo "============================"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-gguf-directory>"
    exit 1
fi

GGUF_DIR=$1
cd "$GGUF_DIR"

# Test questions
questions=(
    "What are your thoughts on the topic?"
    "Tell me about something unusual"
    "Explain this phenomenon"
)

# Create test model
echo "Creating Ollama model..."
ollama create test-model -f Modelfile

# Test each question
for q in "${questions[@]}"; do
    echo -e "\n❓ Question: $q"
    echo "Response:"
    echo "$q" | ollama run test-model | head -20
    echo -e "\n---"
done

# Clean up
ollama rm test-model

echo -e "\n✅ Testing complete!"