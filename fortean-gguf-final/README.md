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

## Known Issues

- The model may occasionally generate longer responses than intended
- Best results with clear, direct questions about anomalous phenomena

## Credits

Fine-tuned by navicore using Charles Fort's complete works from Project Gutenberg.