# Manual Steps to Publish GGUF to HuggingFace

If the script has issues, here are the manual steps:

## 1. Create the GGUF repository on HuggingFace website

Go to: https://huggingface.co/new

- Repository name: `fortean-qwen3-8b-advanced-GGUF`
- Type: Model
- Make it public

## 2. Clone the empty repo

```bash
git clone https://huggingface.co/navicore/fortean-qwen3-8b-advanced-GGUF
cd fortean-qwen3-8b-advanced-GGUF
```

## 3. Copy files

```bash
# Copy GGUF file
cp ../fortean-gguf/fortean-q4_k_m.gguf .

# Copy Modelfile
cp ../fortean-gguf/Modelfile-qwen Modelfile

# Create README
cat > README.md << 'EOF'
# Fortean Qwen3-8B Advanced - GGUF

Quantized version of [navicore/fortean-qwen3-8b-advanced](https://huggingface.co/navicore/fortean-qwen3-8b-advanced) for use with Ollama and llama.cpp.

## Quick Start with Ollama

```bash
# Download this repo
git clone https://huggingface.co/navicore/fortean-qwen3-8b-advanced-GGUF
cd fortean-qwen3-8b-advanced-GGUF

# Create Ollama model
ollama create fortean -f Modelfile

# Run
ollama run fortean "What's your take on artificial intelligence?"
```

## Files

- `fortean-q4_k_m.gguf` - 4-bit quantized model (~4.5GB)
- `Modelfile` - Ollama configuration with proper prompt template

## Example Response

**Q: Tell me about UFOs**

**Fort:** The question isn't their reality but their purpose. I've catalogued 1,847 instances of aerial phenomena that defy explanation, dating back to the Medieval chronicles. In 1897, across the American Midwest, thousands witnessed airships that couldn't have existed. The same shapes appear in Japanese woodcuts from 1803. We're told these are weather balloons, swamp gas, mass hysteria—yet the patterns persist across centuries. Perhaps we're being inventoried. Perhaps we always have been.
EOF
```

## 4. Push to HuggingFace

```bash
git add .
git commit -m "Add GGUF model and Ollama configuration"
git push
```

## 5. Update your main model card

Add this section to your main model README:

```markdown
## Ollama Usage

GGUF quantized versions are available at [navicore/fortean-qwen3-8b-advanced-GGUF](https://huggingface.co/navicore/fortean-qwen3-8b-advanced-GGUF)

Quick start:
\```bash
git clone https://huggingface.co/navicore/fortean-qwen3-8b-advanced-GGUF
cd fortean-qwen3-8b-advanced-GGUF
ollama create fortean -f Modelfile
ollama run fortean
\```
```

That's it! Your GGUF model will be available for anyone to use with Ollama.