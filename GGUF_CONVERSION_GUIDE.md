# Complete Guide: Converting Fine-tuned Models to GGUF

*Lessons learned from the Fortean model conversion*

## Overview

This guide covers the entire process of converting a fine-tuned model to GGUF format for use with Ollama, including common pitfalls and solutions.

## Prerequisites

1. **Python Environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install transformers torch peft accelerate huggingface-hub sentencepiece protobuf
```

2. **Build llama.cpp**
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
cd ..
```

## Step 1: Merge LoRA Adapter with Base Model

If you fine-tuned with LoRA (as we did), you need to merge it first:

```python
#!/usr/bin/env python3
# merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",  # Your base model
    torch_dtype=torch.float16,
    device_map="cpu"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./path-to-lora-adapter")

# Merge and save
model = model.merge_and_unload()
model.save_pretrained("merged_model", safe_serialization=True)

# Copy tokenizer
tokenizer = AutoTokenizer.from_pretrained("./path-to-lora-adapter")
tokenizer.save_pretrained("merged_model")
```

## Step 2: Convert to GGUF

```bash
python3 llama.cpp/convert_hf_to_gguf.py \
    merged_model \
    --outfile model-f16.gguf \
    --outtype f16
```

## Step 3: Quantize

```bash
./llama.cpp/build/bin/llama-quantize \
    model-f16.gguf \
    model-q4_k_m.gguf \
    q4_k_m
```

## Step 4: Create Modelfile

**CRITICAL**: The Modelfile template must match your training data format!

### For Custom Training Format (like Fortean)
```
FROM ./model-q4_k_m.gguf

# If you trained with "Question: X\nAnswer: Y" format:
TEMPLATE """{{ .Prompt }}

{{ .Response }}"""

# Stop sequences - CRITICAL for preventing endless generation
PARAMETER stop "\n\n"
PARAMETER stop "Question:"
PARAMETER num_predict 600

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
```

### For Chat-Template Models
```
FROM ./model-q4_k_m.gguf

# Use the model's built-in template
TEMPLATE """{{ if .System }}{{ .System }}{{ end }}
{{ if .Prompt }}User: {{ .Prompt }}{{ end }}
Assistant: {{ .Response }}"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
```

## Step 5: Test Locally

```bash
ollama create mymodel -f Modelfile
ollama run mymodel "Test question"
```

## Step 6: Upload to HuggingFace

### Create Proper README with YAML Metadata
```markdown
---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- llama-cpp
- gguf
- text-generation
language:
- en
library_name: gguf
pipeline_tag: text-generation
---

# Model Name

Description...
```

### Upload Script
```bash
#!/bin/bash
# upload_to_hf.sh
cd /path/to/gguf/directory
source ../venv/bin/activate

# Upload all files
huggingface-cli upload username/model-name-gguf \
    ./model-q4_k_m.gguf model-q4_k_m.gguf

huggingface-cli upload username/model-name-gguf \
    ./README.md README.md

huggingface-cli upload username/model-name-gguf \
    ./Modelfile Modelfile
```

## Common Issues and Solutions

### 1. Endless Generation
**Problem**: Model generates text forever without stopping.

**Solution**: 
- Add proper stop sequences in Modelfile
- Use `num_predict` as safety limit
- Ensure stop tokens match your training format

### 2. Line Ending Errors in Scripts
**Problem**: Scripts fail with `$'\r': command not found`

**Solution**:
- Create scripts without Windows line endings
- Use one-line commands with `&&`
- Or fix with: `dos2unix script.sh`

### 3. Upload Failures
**Problem**: Complex huggingface-cli commands fail

**Solution**:
- Upload files one at a time
- Use full paths for local files
- Avoid unsupported flags like `--quiet`

### 4. Missing YAML Metadata Warning
**Problem**: HuggingFace shows metadata warning

**Solution**:
- Always include YAML frontmatter in README
- Must be at very top of file
- Include required fields: license, base_model, tags

## For Your Mencken Model

When creating your H.L. Mencken model:

1. **Training Data Format**: Design a consistent format
```
Essay: [User's topic]
Mencken: [Response in Mencken's style]
```

2. **Modelfile Template**: Match your training format exactly
```
TEMPLATE """Essay: {{ .Prompt }}
Mencken: {{ .Response }}"""

PARAMETER stop "Essay:"
PARAMETER stop "\n\nEssay:"
```

3. **Test Stop Tokens**: Before uploading, verify locally that generation stops properly

4. **Repository Organization**:
   - `username/mencken-qwen-7b` - Main model (LoRA adapter)
   - `username/mencken-qwen-7b-gguf` - GGUF version

## Quick Checklist

- [ ] Merge LoRA with base model
- [ ] Convert to GGUF (F16)
- [ ] Quantize to Q4_K_M
- [ ] Create Modelfile with proper stop tokens
- [ ] Test locally with Ollama
- [ ] Create README with YAML metadata
- [ ] Upload to HuggingFace
- [ ] Test the uploaded model

## Final Tips

1. **Always test locally first** - Don't upload until generation stops properly
2. **Keep your training format** - The Modelfile template must match
3. **Document everything** - Include examples in your README
4. **Save your scripts** - You'll need them for updates

Good luck with your Mencken model! His caustic wit should make for entertaining outputs.