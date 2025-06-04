#!/bin/bash
# Publish GGUF files and Ollama instructions to HuggingFace

echo "📤 Publishing GGUF files to HuggingFace..."

# Check if required files exist
if [ ! -f "./fortean-gguf/fortean-q4_k_m.gguf" ]; then
    echo "❌ Error: GGUF file not found at ./fortean-gguf/fortean-q4_k_m.gguf"
    echo "Please run the conversion script first"
    exit 1
fi

# Create new repo for GGUF files (or skip if exists)
echo "Creating HuggingFace repository..."
huggingface-cli repo create navicore/fortean-qwen3-8b-advanced-GGUF --type model || echo "Repo may already exist, continuing..."

# Upload GGUF file
huggingface-cli upload navicore/fortean-qwen3-8b-advanced-GGUF \
    ./fortean-gguf/fortean-q4_k_m.gguf \
    fortean-q4_k_m.gguf

# Upload Modelfile
huggingface-cli upload navicore/fortean-qwen3-8b-advanced-GGUF \
    ./fortean-gguf/Modelfile-qwen \
    Modelfile

# Upload Ollama instructions
huggingface-cli upload navicore/fortean-qwen3-8b-advanced-GGUF \
    ./fortean-gguf/README_OLLAMA.md \
    README.md

# Update main model card with Ollama info
cat > model_card_update.md << 'EOF'
---
license: other
language:
- en
library_name: peft
base_model: Qwen/Qwen3-8B
tags:
- fortean
- charles-fort
- paranormal
- creative-writing
---

# Fortean Qwen3-8B Advanced

A fine-tuned version of Qwen3-8B that responds in the style of Charles Fort (1874-1932), the writer who collected reports of anomalous phenomena.

## Features
- Sees paranormal connections in any topic
- Cites dubious 19th-century sources
- Applies Fort's skeptical perspective to modern topics
- Dry wit and unique voice

## Usage

### With Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base_model, "navicore/fortean-qwen3-8b-advanced")
tokenizer = AutoTokenizer.from_pretrained("navicore/fortean-qwen3-8b-advanced")

prompt = "Question: What's your take on artificial intelligence?\n\nCharles Fort:"
# Generate response...
```

### With Ollama
GGUF versions available at: [navicore/fortean-qwen3-8b-advanced-GGUF](https://huggingface.co/navicore/fortean-qwen3-8b-advanced-GGUF)

```bash
# Download GGUF and Modelfile from the GGUF repo
ollama create fortean -f Modelfile
ollama run fortean "What do you think about UFOs?"
```

## Training Details
- Base model: Qwen3-8B (not instruct)
- Method: LoRA fine-tuning
- Training examples: ~200 high-quality Q&A pairs
- Hardware: Apple M4 Pro

## Example Output
**Q: What's your view on cryptocurrency?**

**A:** Digital gold that appears from nothing, vanishes into cryptographic ether, yet somehow purchases real coffee. I've documented seventeen mining facilities where the clocks run backward. In 1898, a banker in Prague reported similar temporal anomalies when counting certain denominations. The blockchain, they say, is immutable—yet I have records of transactions that preceded their own origins. Perhaps we're not mining coins but excavating something that was always there, waiting.
EOF

# Upload updated model card
huggingface-cli upload navicore/fortean-qwen3-8b-advanced \
    ./model_card_update.md \
    README.md

echo "✅ Published GGUF model and instructions!"
echo "Main model: https://huggingface.co/navicore/fortean-qwen3-8b-advanced"
echo "GGUF model: https://huggingface.co/navicore/fortean-qwen3-8b-advanced-GGUF"