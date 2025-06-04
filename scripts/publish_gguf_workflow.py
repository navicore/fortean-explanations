#!/usr/bin/env python3
"""
Complete workflow for publishing GGUF versions of your models.
This handles the full pipeline from trained model to Ollama-ready GGUF.
"""

import subprocess
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import shutil
import os

class GGUFPublisher:
    def __init__(self, hf_username="navicore"):
        self.hf_username = hf_username
        self.api = HfApi()
        
    def create_gguf_versions(self, model_id: str, quantizations=["q4_k_m", "q5_k_m", "q8_0"]):
        """Convert HF model to multiple GGUF quantizations"""
        
        print(f"🚀 Converting {model_id} to GGUF formats...")
        
        # Setup directories
        work_dir = Path("gguf_conversion")
        work_dir.mkdir(exist_ok=True)
        
        # Download model
        print("📥 Downloading model from HuggingFace...")
        hf_model_dir = work_dir / "hf_model"
        subprocess.run([
            "huggingface-cli", "download", model_id,
            "--local-dir", str(hf_model_dir)
        ], check=True)
        
        # Ensure llama.cpp is available
        if not Path("llama.cpp").exists():
            print("📦 Installing llama.cpp...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/ggerganov/llama.cpp"
            ], check=True)
            subprocess.run(["cmake", "-B", "llama.cpp/build"], check=True)
            subprocess.run(["cmake", "--build", "llama.cpp/build", "--config", "Release"], check=True)
        
        # Convert for each quantization
        gguf_files = []
        for quant in quantizations:
            output_file = work_dir / f"fortean-{quant}.gguf"
            print(f"🔄 Creating {quant} quantization...")
            
            subprocess.run([
                "python", "llama.cpp/convert-hf-to-gguf.py",
                str(hf_model_dir),
                "--outfile", str(output_file),
                "--outtype", quant
            ], check=True)
            
            gguf_files.append(output_file)
            
        return gguf_files, work_dir
    
    def create_ollama_modelfile(self, model_name: str, gguf_path: Path):
        """Create Modelfile for Ollama"""
        
        modelfile_content = f'''# Fortean AI - Charles Fort's perspective on everything
FROM ./{gguf_path.name}

# Custom prompt template for Fortean responses
TEMPLATE """{{{{ if .System }}}}{{{{ .System }}}}

{{{{ end }}}}{{{{ if .Prompt }}}}Question: {{{{ .Prompt }}}}

Charles Fort: {{{{ end }}}}{{{{ .Response }}}}"""

# Parameters tuned for Fort's style
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER top_k 40

# System prompt
SYSTEM """You are Charles Fort (1874-1932), the writer who collected reports of anomalous phenomena. You speak with dry wit, document strange coincidences across time, and see connections others miss. You cite dubious sources like 'Provincial Medical Journal, circa 1843' and note when similar events occurred 'about 1887.' You are skeptical of scientific orthodoxy while maintaining intellectual curiosity."""

# Example usage
# ollama run fortean "What's your take on artificial intelligence?"
'''
        
        modelfile_path = gguf_path.parent / f"Modelfile-{gguf_path.stem}"
        modelfile_path.write_text(modelfile_content)
        return modelfile_path
    
    def create_readme(self, work_dir: Path, model_id: str, gguf_files: list):
        """Create README for GGUF repository"""
        
        readme_content = f'''# {model_id} - GGUF Versions

This repository contains GGUF quantized versions of [{model_id}](https://huggingface.co/{model_id}) for use with:
- [Ollama](https://ollama.ai)
- [LM Studio](https://lmstudio.ai)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Available Quantizations

| Variant | Size | Description | Use Case |
|---------|------|-------------|----------|
| Q4_K_M | ~4.5GB | Balanced quality/size | Recommended for most users |
| Q5_K_M | ~5.5GB | Higher quality | Better responses, more VRAM |
| Q8_0 | ~8.5GB | Near full quality | Best quality, highest requirements |

## Quick Start with Ollama

1. Download a GGUF file
2. Create a Modelfile:
```
FROM ./fortean-q4_k_m.gguf
SYSTEM "You are Charles Fort, collector of anomalous phenomena."
```
3. Create and run:
```bash
ollama create fortean -f Modelfile
ollama run fortean "What's your take on UFOs?"
```

## Model Details

This is a fine-tuned version of Qwen3-8B trained to respond in the style of Charles Fort, featuring:
- Paranormal connections to any topic
- Citations of obscure historical sources  
- Dry wit and skepticism of scientific orthodoxy
- Pattern recognition across time periods

## Example Output

**Question:** "What do you think about social media?"

**Charles Fort:** "The synchronized behaviors of millions, all staring into glowing rectangles at precisely the same moments—I've documented similar instances in 1847 when entire villages would simultaneously cease all activity. The algorithms, we're told, merely suggest content, yet the patterns of engagement match those of locust swarms before biblical plagues. Perhaps we're not sharing thoughts but being harvested. I note that server farms cluster near sites of historical mass hysteria."

## License

Same as base model (Qwen3-8B license)
'''
        
        readme_path = work_dir / "README.md"
        readme_path.write_text(readme_content)
        return readme_path
    
    def publish_to_huggingface(self, model_id: str, gguf_files: list, work_dir: Path):
        """Upload GGUF files to HuggingFace"""
        
        gguf_repo_id = f"{model_id}-GGUF"
        
        print(f"📤 Publishing to {gguf_repo_id}...")
        
        # Create repository
        try:
            create_repo(repo_id=gguf_repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"Note: {e}")
        
        # Upload each GGUF file
        for gguf_file in gguf_files:
            print(f"Uploading {gguf_file.name}...")
            self.api.upload_file(
                path_or_fileobj=str(gguf_file),
                path_in_repo=gguf_file.name,
                repo_id=gguf_repo_id,
                repo_type="model"
            )
            
            # Also upload corresponding Modelfile
            modelfile = work_dir / f"Modelfile-{gguf_file.stem}"
            if modelfile.exists():
                self.api.upload_file(
                    path_or_fileobj=str(modelfile),
                    path_in_repo=modelfile.name,
                    repo_id=gguf_repo_id,
                    repo_type="model"
                )
        
        # Upload README
        readme = work_dir / "README.md"
        if readme.exists():
            self.api.upload_file(
                path_or_fileobj=str(readme),
                path_in_repo="README.md",
                repo_id=gguf_repo_id,
                repo_type="model"
            )
        
        print(f"✅ Published to https://huggingface.co/{gguf_repo_id}")
        return gguf_repo_id

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert and publish GGUF versions")
    parser.add_argument("--model", default="navicore/fortean-qwen3-8b-advanced")
    parser.add_argument("--quantizations", nargs="+", default=["q4_k_m", "q5_k_m"])
    parser.add_argument("--publish", action="store_true", help="Upload to HuggingFace")
    
    args = parser.parse_args()
    
    publisher = GGUFPublisher()
    
    # Convert to GGUF
    gguf_files, work_dir = publisher.create_gguf_versions(
        args.model, 
        args.quantizations
    )
    
    # Create Modelfiles
    for gguf_file in gguf_files:
        modelfile = publisher.create_ollama_modelfile("fortean", gguf_file)
        print(f"📝 Created {modelfile}")
    
    # Create README
    publisher.create_readme(work_dir, args.model, gguf_files)
    
    if args.publish:
        # Publish to HuggingFace
        publisher.publish_to_huggingface(args.model, gguf_files, work_dir)
    else:
        print("\n📁 GGUF files created in:", work_dir)
        print("To publish, run with --publish flag")
    
    print("\n🎯 To use with Ollama locally:")
    print(f"ollama create fortean -f {work_dir}/Modelfile-fortean-q4_k_m")
    print("ollama run fortean")

if __name__ == "__main__":
    main()