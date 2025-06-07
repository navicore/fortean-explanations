---
license: apache-2.0
language:
- en
library_name: gguf
tags:
- text-generation
- creative-writing
- fortean
- anomalous-phenomena
- charles-fort
base_model: Qwen/Qwen2.5-3B-Instruct
quantized_by: navicore
model_type: qwen2
---

# Fortean-Qwen3-8B GGUF

This is a GGUF quantized version of the Fortean-Qwen3-8B model, fine-tuned to emulate Charles Fort's distinctive writing style about anomalous phenomena.

## Model Details

- **Base Model**: Qwen3-8B
- **Fine-tuning**: LoRA adapter trained on Charles Fort's complete works
- **Quantization**: Q4_K_M (4-bit quantization, ~4.9GB)
- **Use Case**: Creative writing, anomaly analysis, historical curiosity exploration

## Installation

1. Install Ollama: https://ollama.ai
2. Create the model:
```bash
ollama create fortean -f Modelfile.final
```
3. Run the model:
```bash
ollama run fortean
```

## Example Usage

```
User: What are your thoughts on the Bermuda Triangle?

Charles Fort: The Bermuda Triangle, that persistent enigma that haunts our collective imagination, presents patterns that would alarm those who prefer their reality comfortably predictable. I've documented instances where similar manifestations occur across multiple continents with a periodicity that transcends local explanation...
```

## Model Characteristics

- Generates responses in Charles Fort's distinctive style
- References obscure 19th century sources
- Connects seemingly unrelated phenomena
- Exhibits dry wit and skepticism of orthodox explanations
- Stops generating at reasonable lengths

## Optimal Usage

### Prompt Format
For best results, use this prompt format:
```
Question: [your question here]

Charles Fort:
```

### Recommended Parameters
- Temperature: 0.7-0.8
- Top P: 0.85-0.9
- Repeat Penalty: 1.1-1.2
- Max Tokens: 250-300

### Stop Tokens
Configure these stop tokens to prevent multiple responses:
- `<|endoftext|>`
- `<|im_end|>`
- `\nQuestion:`
- `\n\nQuestion:`

### Example Ollama Modelfile
```modelfile
FROM fortean-q4_k_m.gguf

TEMPLATE """Question: {{ .Prompt }}

Charles Fort: {{ .Response }}"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 300
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "\nQuestion:"
PARAMETER stop "\n\nQuestion:"
```

### Usage with llama.cpp
```bash
./main -m fortean-q4_k_m.gguf -p "Question: What are your thoughts on red rain?\n\nCharles Fort:" -n 300 --temp 0.8
```

### Usage with Python (llama-cpp-python)
```python
from llama_cpp import Llama

llm = Llama(model_path="fortean-q4_k_m.gguf", n_ctx=2048)

prompt = "Question: What are your thoughts on red rain?\n\nCharles Fort:"
response = llm(prompt, max_tokens=300, temperature=0.8, stop=["<|endoftext|>", "\nQuestion:"])
print(response['choices'][0]['text'])
```

## Known Issues

- Without proper stop tokens, the model may generate multiple responses
- Best results with clear, direct questions about anomalous phenomena

## Credits

Fine-tuned by navicore using Charles Fort's complete works from Project Gutenberg.