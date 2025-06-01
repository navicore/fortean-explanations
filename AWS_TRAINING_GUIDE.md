# AWS Training Guide for Fortean Model

## Overview

This guide covers training the Fortean model on AWS EC2 with GPUs for optimal performance.

## Option 1: Local Training on M4 Mac (Testing)

For initial testing with a smaller model:

```bash
# Test with Phi-2 (2.7B params) - fits on M4 Mac
python scripts/train_production_model.py \
  --model microsoft/phi-2 \
  --device mps \
  --no-4bit
```

Monitor progress:
```bash
tail -f training_progress.log
```

## Option 2: AWS EC2 Training (Production)

### 1. Launch EC2 Instance

**Recommended Instance Types:**
- `g5.xlarge` (1x A10G GPU, 24GB) - $1.006/hour - Good for 7B models with 4-bit
- `g5.2xlarge` (1x A10G GPU, 24GB) - $1.212/hour - More CPU/RAM
- `g5.4xlarge` (1x A10G GPU, 24GB) - $1.624/hour - Even more resources
- `p3.2xlarge` (1x V100, 16GB) - $3.06/hour - Faster training but more expensive

**AMI:** Deep Learning AMI GPU PyTorch 2.0 (Ubuntu 20.04)

### 2. Setup Script

Save this as `setup_aws.sh`:

```bash
#!/bin/bash
# AWS EC2 setup script

# Update system
sudo apt update && sudo apt upgrade -y

# Clone your repo
git clone https://github.com/YOUR_USERNAME/fortean-explanations.git
cd fortean-explanations

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install CUDA-specific packages
pip install bitsandbytes-cuda117  # For 4-bit quantization

# Download data if not in repo
python scripts/collect_fort_texts.py
python scripts/abstract_dates.py
python scripts/prepare_data.py
python scripts/setup_rag.py
python scripts/generate_qa_pairs.py

# Set up wandb (optional)
wandb login YOUR_WANDB_KEY
```

### 3. Training Commands

**For Mistral 7B (Recommended):**
```bash
python scripts/train_production_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --device cuda \
  --wandb-project fortean-mistral \
  --push-to-hub \
  --repo-name YOUR_HF_USERNAME/fortean-mistral-7b \
  --hf-token YOUR_HF_TOKEN
```

**For Llama 2 7B (Requires access):**
```bash
python scripts/train_production_model.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --wandb-project fortean-llama \
  --push-to-hub \
  --repo-name YOUR_HF_USERNAME/fortean-llama2-7b \
  --hf-token YOUR_HF_TOKEN
```

### 4. Cost Estimation

For 500 training examples with Mistral 7B:
- Training time: ~2-4 hours on g5.xlarge
- Cost: ~$2-4 for full training
- With spot instances: ~$0.50-1.00

### 5. Monitoring Training

SSH into your instance and monitor:
```bash
# Watch GPU usage
nvidia-smi -l 1

# Monitor training progress
tail -f training_progress.log

# Check wandb dashboard (if using)
```

## Option 3: Automated AWS Training with SageMaker

For a more managed approach, use `sagemaker_train.py`:

```python
import sagemaker
from sagemaker.huggingface import HuggingFace

# Define training job
huggingface_estimator = HuggingFace(
    entry_point='train_production_model.py',
    source_dir='./scripts',
    instance_type='ml.g5.xlarge',
    instance_count=1,
    role=sagemaker.get_execution_role(),
    transformers_version='4.36',
    pytorch_version='2.1',
    py_version='py310',
    hyperparameters={
        'model': 'mistralai/Mistral-7B-Instruct-v0.2',
        'push-to-hub': True,
        'repo-name': 'YOUR_HF_USERNAME/fortean-mistral-7b',
        'hf-token': 'YOUR_HF_TOKEN'
    }
)

# Start training
huggingface_estimator.fit()
```

## Publishing to HuggingFace

### 1. Create Model Card

Create `README.md` in your model directory:

```markdown
---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- charles-fort
- anomalous-phenomena
- fortean
- lora
base_model: mistralai/Mistral-7B-Instruct-v0.2
---

# Fortean Mistral 7B

A fine-tuned version of Mistral 7B that responds in the style of Charles Fort, 
the early 20th century researcher of anomalous phenomena.

## Model Details

- **Base Model:** Mistral-7B-Instruct-v0.2
- **Fine-tuning Method:** LoRA (r=32, alpha=64)
- **Training Data:** 500 synthetic Q&A pairs based on Fort's four books
- **Training Hardware:** NVIDIA A10G GPU

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and LoRA weights
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/fortean-mistral-7b")
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/fortean-mistral-7b")

# Generate response
prompt = "What do you think about UFOs?"
inputs = tokenizer(f"Human: {prompt}\n\nFort:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.8)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

The model was trained on passages from Charles Fort's four books:
- The Book of the Damned (1919)
- New Lands (1923)
- Lo! (1931)
- Wild Talents (1932)

## Limitations

- May generate historically inaccurate dates or events
- Responses reflect early 20th century perspectives
- Not suitable for factual or scientific information

## Citation

If you use this model, please cite:

```bibtex
@misc{fortean-mistral-2024,
  author = {Your Name},
  title = {Fortean Mistral 7B: Charles Fort Style Language Model},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/YOUR_USERNAME/fortean-mistral-7b}
}
```
```

### 2. Push to Hub

The training script will automatically push to HuggingFace if you provide:
- `--push-to-hub` flag
- `--repo-name YOUR_USERNAME/model-name`
- `--hf-token YOUR_TOKEN`

## Tips for Success

1. **Start Small**: Test with Phi-2 locally first
2. **Use Spot Instances**: 70-90% cheaper for training
3. **Monitor Progress**: Use wandb for real-time metrics
4. **Save Checkpoints**: Enable checkpoint saving every N steps
5. **Gradual Scaling**: Start with fewer epochs, increase if needed

## Troubleshooting

**Out of Memory:**
- Reduce batch size
- Enable gradient checkpointing
- Use smaller model or more quantization

**Slow Training:**
- Check GPU utilization with `nvidia-smi`
- Increase batch size if GPU memory allows
- Use mixed precision training (bf16)

**Model Not Learning:**
- Increase learning rate
- Add more training data
- Adjust LoRA rank (r parameter)