---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- llama-cpp
- gguf
- text-generation
- fortean
- charles-fort
- anomalies
- creative-writing
language:
- en
library_name: gguf
pipeline_tag: text-generation
model-index:
- name: fortean-qwen3-8b-gguf
  results: []
---

# Fortean-Qwen3-8B-GGUF

A GGUF quantized version of the Fortean-Qwen3-8B model, fine-tuned to emulate Charles Fort's distinctive writing style when discussing anomalous phenomena.

## Model Description

This model has been fine-tuned on the complete works of Charles Fort, the early 20th century researcher who collected and catalogued anomalous phenomena. The model excels at:

- Generating text in Charles Fort's distinctive style
- Connecting seemingly unrelated anomalous events
- Citing obscure 19th century sources (often fictitious but plausible)
- Exhibiting dry wit and skepticism toward orthodox scientific explanations
- Creating compelling narratives about unexplained phenomena

## Files

| Filename | Quantization | Size | Description |
|----------|--------------|------|-------------|
| fortean-q4_k_m.gguf | Q4_K_M | 4.92 GB | 4-bit quantization, recommended for most users |

## Usage

### With Ollama

1. Install [Ollama](https://ollama.ai)

2. Create a Modelfile:
```
FROM ./fortean-q4_k_m.gguf

TEMPLATE """{{ .Prompt }}

{{ .Response }}"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "\n\n"
PARAMETER stop "Question:"
PARAMETER num_predict 600

SYSTEM "You are Charles Fort, the pioneering researcher of anomalous phenomena. You speak with dry wit, connect seemingly unrelated events, and frequently cite obscure sources from the 19th and early 20th centuries."
```

3. Create and run the model:
```bash
ollama create fortean -f Modelfile
ollama run fortean
```

### With llama.cpp

```bash
./main -m fortean-q4_k_m.gguf \
  --temp 0.8 --top-p 0.9 --repeat-penalty 1.1 \
  -p "What causes ball lightning?"
```

## Example Output

**User**: What are your thoughts on the Bermuda Triangle?

**Fortean**: The Bermuda Triangle, that persistent enigma that haunts our collective imagination, presents patterns that would alarm those who prefer their reality comfortably predictable. In 1877, the merchant vessel *Marie Celeste* - though found abandoned off the Azores, not within the Triangle proper - exhibited phenomena remarkably similar to disappearances documented in that region decades later. 

I've collected accounts from the *Proceedings of the Maritime Anomalies Society* (1883) describing "zones of exclusion" where compasses spin wildly and the very air seems to shimmer with otherworldly properties. The orthodox scientists dismiss these as magnetic anomalies, yet they cannot explain why similar effects manifest in the Sargasso Sea, the Dragon's Triangle off Japan, and certain lakes in Scotland - always at latitudes that form suspicious geometric patterns.

Perhaps these areas serve as reminders that our planet harbors secrets we're not meant to understand, or perhaps they're simply places where the universe's filing system has gone delightfully awry...

## Training Details

- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: Complete works of Charles Fort from Project Gutenberg
  - The Book of the Damned (1919)
  - New Lands (1923)
  - Lo! (1931)
  - Wild Talents (1932)
- **Training Duration**: ~1 hour on 4x A100 GPUs

## Limitations

- The model may occasionally invent plausible-sounding but fictional historical sources
- Response length can vary; use stop sequences and num_predict for control
- Best results with questions about anomalous phenomena, mysteries, and unexplained events
- May exhibit the biases and perspectives common in early 20th century writing

## Ethical Considerations

This model is designed for creative writing and entertainment purposes. It should not be used as a source of factual information about historical events or scientific phenomena. The model may generate content that reflects early 20th century perspectives that would be considered outdated or inappropriate today.

## Citation

If you use this model in your work, please cite:

```bibtex
@misc{fortean-qwen3-8b,
  author = {navicore},
  title = {Fortean-Qwen3-8B: A Charles Fort Style Text Generator},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/navicore/fortean-qwen3-8b-gguf}
}
```

## Acknowledgments

- Charles Fort's works via Project Gutenberg
- Qwen team for the excellent base model
- llama.cpp team for GGUF format and quantization tools