# Scripts Directory

This directory contains all the working scripts for the Fortean model project.

## GGUF Conversion Scripts

### `convert_lora_to_gguf_complete.py`
Complete pipeline for converting a LoRA fine-tuned model to GGUF format.
- Merges LoRA adapter with base model
- Converts to GGUF
- Quantizes to Q4_K_M
- Creates Modelfile with proper stop tokens

**Usage:**
```bash
python3 convert_lora_to_gguf_complete.py
```

### `merge_and_convert_fortean.py`
The actual script used to create the working Fortean GGUF.

### `convert_fortean_proper.py`
Alternative conversion script with different options.

## Upload Scripts

### `upload_gguf_to_hf.sh`
Generic script to upload GGUF files to HuggingFace.

**Usage:**
```bash
./upload_gguf_to_hf.sh username/model-gguf ./model-q4_k_m.gguf
```

### `publish_gguf_workflow.py`
Automated workflow for publishing GGUF to HuggingFace.

## Testing Scripts

### `test_gguf_locally.sh`
Test a GGUF model locally with Ollama before uploading.

**Usage:**
```bash
./test_gguf_locally.sh /path/to/gguf/directory
```

### `chat_fortean.py` / `chat_fortean_concise.py`
Interactive chat scripts for testing the Fortean model.

## Training Scripts

### `train_qwen3_8b_fortean.py`
The script used to fine-tune the Fortean model on Charles Fort's writings.

### `generate_*.py`
Various scripts for generating training data from Fort's texts.

## Data Processing

### `prepare_data.py`
Prepares Charles Fort's texts for training.

### `abstract_dates.py`
Abstracts specific dates in texts to improve generalization.

## Utilities

### `monitor_memory.py` / `monitor_progress.py`
Monitor training progress and memory usage.

### `stop_training.sh`
Emergency script to stop training if needed.

## Important Notes

1. **Always test GGUF locally** before uploading to HuggingFace
2. **Check stop tokens** - This was our biggest issue
3. **Use simple one-line scripts** when possible to avoid line ending issues
4. **Keep the virtual environment activated** when running Python scripts