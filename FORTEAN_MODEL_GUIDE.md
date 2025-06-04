# Fortean LLM: Complete Training and Deployment Guide

This guide documents the exact process to create, train, and deploy a Charles Fort-style LLM.

## Overview

We created a fine-tuned model that responds in the style of Charles Fort, seeing paranormal connections in any topic and citing dubious 19th-century sources. The model is available at:
- **HuggingFace**: `navicore/fortean-qwen3-8b-advanced`
- **Ollama**: (after local conversion)

## Training Process

### 1. Data Collection and Preparation

```bash
# Collect Fort's texts from Project Gutenberg
python scripts/collect_fort_texts.py

# Apply temporal abstraction (converts dates to Fort's style)
python scripts/abstract_dates.py

# Set up RAG for context
python scripts/setup_rag.py
```

### 2. Generate Training Data

We created high-quality Q&A pairs that teach the model to:
- Apply Fort's perspective to modern topics
- Cite obscure historical sources
- Make paranormal connections
- Use Fort's distinctive voice

```bash
# Generate advanced training data with citations
python scripts/generate_enhanced_qa.py
python scripts/generate_diverse_qa.py
```

### 3. Train the Model

**Key decisions:**
- **Base model**: Qwen3-8B (not instruct) - better for style transfer
- **Method**: LoRA fine-tuning for efficiency
- **Hardware**: Apple M4 Pro with 64GB RAM

```bash
# The successful training script
python scripts/train_qwen3_8b_advanced.py
```

Training configuration:
```python
model_name = "Qwen/Qwen3-8B"
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
max_length = 1536
learning_rate = 3e-5
num_epochs = 5
```

### 4. Test the Model

```bash
python scripts/chat_fortean.py --model models/fortean_qwen3_8b_advanced
```

## Deployment Process

### 1. Publish to HuggingFace

```bash
# Push the LoRA adapter (not the full model)
huggingface-cli upload navicore/fortean-qwen3-8b-advanced ./models/fortean_qwen3_8b_advanced
```

### 2. Convert to GGUF for Ollama

```bash
# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
cd ..

# Merge LoRA with base model and convert
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + LoRA
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "navicore/fortean-qwen3-8b-advanced")
model = model.merge_and_unload()

# Save merged
model.save_pretrained("fortean-merged")
tokenizer = AutoTokenizer.from_pretrained("navicore/fortean-qwen3-8b-advanced")
tokenizer.save_pretrained("fortean-merged")
EOF

# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py fortean-merged --outfile fortean-f16.gguf --outtype f16

# Quantize
./llama.cpp/build/bin/llama-quantize fortean-f16.gguf fortean-q4_k_m.gguf q4_k_m
```

### 3. Create Ollama Model

```bash
# Create Modelfile with proper stop tokens
cat > Modelfile << 'EOF'
FROM ./fortean-q4_k_m.gguf

TEMPLATE """Question: {{ .Prompt }}

Charles Fort: {{ .Response }}<|endoftext|>"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|endoftext|>"
PARAMETER stop "Question:"
EOF

# Create Ollama model
ollama create fortean -f Modelfile
ollama run fortean
```

## Key Lessons Learned

1. **Base vs Instruct Models**: Base models (like Qwen3-8B) are better for style transfer than instruct variants
2. **Stop Tokens**: Critical for Ollama - must match model's tokenizer
3. **Memory Management**: 14B models require ~56GB RAM for training on M4 Mac
4. **LoRA Configuration**: r=64 with specific target modules works well for 8B models
5. **Training Data Quality**: Clean, well-formatted examples are essential

## Common Issues and Solutions

### Model loops indefinitely in Ollama
- Add proper stop tokens: `<|endoftext|>`, `Question:`
- Set `num_predict` parameter to limit length

### Model doesn't capture Fort's style
- Use base model, not instruct
- Include diverse examples covering modern topics
- Train for at least 200 steps

### Memory errors during training
- Reduce LoRA rank
- Use gradient checkpointing
- Decrease batch size

## Files Structure

```
fortean-explanations/
├── scripts/
│   ├── train_qwen3_8b_advanced.py    # The successful training script
│   ├── chat_fortean.py               # Python chat interface
│   └── convert_lora_to_gguf.sh       # GGUF conversion
├── models/
│   └── fortean_qwen3_8b_advanced/    # Trained model
└── data/
    └── training_data/
        └── fortean_advanced_training.json  # Training examples
```

## Reproducing This Work

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run training: `python scripts/train_qwen3_8b_advanced.py`
4. Convert to GGUF using the process above
5. Enjoy chatting with Charles Fort!

## Future Improvements

- Add explicit stop token training
- Create instruction-tuned variant
- Train on more Fort texts
- Add length control (brief/verbose modes)