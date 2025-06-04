---
license: apache-2.0
language:
- en
base_model: navicore/fortean-qwen3-8b-advanced
tags:
- fortean
- charles-fort
- paranormal
- creative-writing
- gguf
- ollama
- llama-cpp
model_type: llama
quantized_by: navicore
---

# Fortean Qwen3-8B Advanced - GGUF

Quantized GGUF version of [navicore/fortean-qwen3-8b-advanced](https://huggingface.co/navicore/fortean-qwen3-8b-advanced) for use with Ollama, llama.cpp, and other GGUF-compatible inference engines.

## Model Description

This model responds in the style of Charles Fort (1874-1932), the writer who collected reports of anomalous phenomena. It sees paranormal connections in any topic and cites dubious 19th-century sources with dry wit.

## Files

| Filename | Quant Method | Size | Description |
|----------|--------------|------|-------------|
| fortean-q4_k_m.gguf | Q4_K_M | 4.5GB | Recommended. Balanced quality and size |

## Quick Start with Ollama

1. Download the files:
```bash
git clone https://huggingface.co/navicore/fortean-qwen3-8b-advanced-GGUF
cd fortean-qwen3-8b-advanced-GGUF
```

2. Create Ollama model:
```bash
ollama create fortean -f Modelfile
```

3. Run:
```bash
ollama run fortean "What's your take on artificial intelligence?"
```

## Modelfile

The included `Modelfile` is configured with:
- Proper prompt template matching the training format
- Temperature 0.8 for creative responses
- Stop sequences to prevent repetition

## Example Output

**Question:** What do you think about cryptocurrency?

**Charles Fort:** Digital gold that appears from nothing, vanishes into cryptographic ether, yet somehow purchases real coffee. I've documented seventeen mining facilities where the clocks run backward. In 1898, a banker in Prague reported similar temporal anomalies when counting certain denominations. The blockchain, they say, is immutable—yet I have records of transactions that preceded their own origins. Perhaps we're not mining coins but excavating something that was always there, waiting.

## Prompt Format

The model was trained with this format:
```
Question: {your question}

Charles Fort: {response}
```

## License

Same as the base model - see [Qwen3 license](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE)

## Citation

If you use this model, please cite:
```
@misc{fortean-qwen3-8b,
  author = {navicore},
  title = {Fortean Qwen3-8B Advanced},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/navicore/fortean-qwen3-8b-advanced}
}
```