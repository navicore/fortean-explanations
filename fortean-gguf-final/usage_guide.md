# Fortean GGUF Usage Guide

## Optimal Prompt Format

For best results, use this prompt format:

```
Question: [your question here]

Charles Fort:
```

## Recommended Parameters

- Temperature: 0.7-0.8
- Top P: 0.85-0.9
- Repeat Penalty: 1.1-1.2
- Max Tokens: 250-300

## Stop Tokens

The model works best with these stop tokens:
- `<|endoftext|>`
- `<|im_end|>`
- `\nQuestion:`
- `\n\nQuestion:`

## Example Ollama Modelfile

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

## Example Usage

### With llama.cpp
```bash
./main -m fortean-q4_k_m.gguf -p "Question: What are your thoughts on red rain?\n\nCharles Fort:" -n 300 --temp 0.8
```

### With Python (llama-cpp-python)
```python
from llama_cpp import Llama

llm = Llama(model_path="fortean-q4_k_m.gguf", n_ctx=2048)

prompt = "Question: What are your thoughts on red rain?\n\nCharles Fort:"
response = llm(prompt, max_tokens=300, temperature=0.8, stop=["<|endoftext|>", "\nQuestion:"])
print(response['choices'][0]['text'])
```

## Notes

- The model may generate multiple responses if stop tokens aren't configured properly
- For best results, avoid using `\n\n` as a stop token as it can cut responses short
- The model responds best to questions about anomalous phenomena, scientific orthodoxy, and unexplained events