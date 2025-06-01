# AWS Training Guide for Fortean LLM

## Quick Start (EC2)

### 1. Launch Instance
- **Instance Type**: `g5.2xlarge` (1 A10G GPU, 24GB VRAM) - ~$1.2/hour
- **AMI**: Deep Learning AMI (Ubuntu 22.04) - has PyTorch pre-installed
- **Storage**: 100GB EBS

### 2. Setup Commands
```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/fortean-explanations.git
cd fortean-explanations

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run training
python scripts/train_production_model.py \
  --device cuda \
  --wandb-project fortean-llm \
  --push-to-hub \
  --repo-name YOUR_HF_USERNAME/fortean-7b \
  --hf-token YOUR_HF_TOKEN
```

### 3. Cost Estimation
- Training time: ~4-6 hours on g5.2xlarge
- Total cost: ~$5-8 for complete training

## Alternative: SageMaker (Managed)

```python
# Use SageMaker training job with HuggingFace estimator
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point='train_production_model.py',
    source_dir='./scripts',
    instance_type='ml.g5.2xlarge',
    instance_count=1,
    role=sagemaker_role,
    transformers_version='4.36',
    pytorch_version='2.1',
    py_version='py310',
    hyperparameters={
        'model': 'mistralai/Mistral-7B-Instruct-v0.2',
        'push-to-hub': True,
        'repo-name': 'YOUR_HF_USERNAME/fortean-7b'
    }
)
```

## Local M4 Mac Training

Your M4 Mac Mini can handle this! Just use:
```bash
python scripts/train_production_model.py --device mps
```

Expected training time: 8-12 hours (overnight run recommended)