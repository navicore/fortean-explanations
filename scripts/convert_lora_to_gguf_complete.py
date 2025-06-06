#!/usr/bin/env python3
"""
Complete script to convert a LoRA fine-tuned model to GGUF format.
Based on lessons learned from the Fortean model conversion.
"""

import json
import subprocess
import shutil
from pathlib import Path

def merge_lora_with_base(lora_path, base_model_name, output_path):
    """Merge LoRA adapter with base model."""
    print(f"🔄 Merging LoRA adapter with {base_model_name}...")
    
    merge_script = f'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model_name}", 
    torch_dtype=torch.float16,
    device_map="cpu"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, "{lora_path}")

print("Merging adapter...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("{output_path}", safe_serialization=True)

print("Copying tokenizer files...")
tokenizer = AutoTokenizer.from_pretrained("{lora_path}")
tokenizer.save_pretrained("{output_path}")

print("✅ Merge complete!")
'''
    
    with open("temp_merge.py", "w") as f:
        f.write(merge_script)
    
    subprocess.run(["python3", "temp_merge.py"], check=True)
    Path("temp_merge.py").unlink()

def convert_to_gguf(model_path, output_path, llama_cpp_path="llama.cpp"):
    """Convert HuggingFace model to GGUF format."""
    print("\n🔄 Converting to GGUF format...")
    
    cmd = [
        "python3", f"{llama_cpp_path}/convert_hf_to_gguf.py",
        model_path,
        "--outfile", f"{output_path}/model-f16.gguf",
        "--outtype", "f16"
    ]
    
    subprocess.run(cmd, check=True)
    print("✅ Converted to F16 GGUF")

def quantize_gguf(input_file, output_file, quantization="q4_k_m", llama_cpp_path="llama.cpp"):
    """Quantize GGUF file."""
    print(f"\n📦 Quantizing to {quantization}...")
    
    cmd = [
        f"{llama_cpp_path}/build/bin/llama-quantize",
        input_file,
        output_file,
        quantization
    ]
    
    subprocess.run(cmd, check=True)
    print(f"✅ Quantized to {quantization}")

def create_modelfile(output_path, model_name, template, stop_sequences):
    """Create Ollama Modelfile."""
    print("\n📝 Creating Modelfile...")
    
    modelfile_content = f'''FROM ./{model_name}

# Template
TEMPLATE """{template}"""

# Model parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# Stop sequences
'''
    
    for stop in stop_sequences:
        modelfile_content += f'PARAMETER stop "{stop}"\n'
    
    modelfile_content += '\n# Response length limit\nPARAMETER num_predict 600'
    
    with open(f"{output_path}/Modelfile", 'w') as f:
        f.write(modelfile_content)
    
    print("✅ Created Modelfile")

def main():
    # Configuration
    config = {
        "lora_path": "path/to/lora/adapter",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "gguf_output",
        "model_name": "model-q4_k_m.gguf",
        "template": "{{ .Prompt }}\n\n{{ .Response }}",
        "stop_sequences": ["\n\n", "Question:", "<|endoftext|>"],
        "llama_cpp_path": "llama.cpp"
    }
    
    # Create output directory
    Path(config["output_dir"]).mkdir(exist_ok=True)
    
    # Step 1: Merge LoRA with base
    merged_path = f"{config['output_dir']}/merged_model"
    merge_lora_with_base(config["lora_path"], config["base_model"], merged_path)
    
    # Step 2: Convert to GGUF
    convert_to_gguf(merged_path, config["output_dir"], config["llama_cpp_path"])
    
    # Step 3: Quantize
    quantize_gguf(
        f"{config['output_dir']}/model-f16.gguf",
        f"{config['output_dir']}/{config['model_name']}",
        "q4_k_m",
        config["llama_cpp_path"]
    )
    
    # Step 4: Create Modelfile
    create_modelfile(
        config["output_dir"],
        config["model_name"],
        config["template"],
        config["stop_sequences"]
    )
    
    # Clean up
    shutil.rmtree(merged_path)
    Path(f"{config['output_dir']}/model-f16.gguf").unlink()
    
    print("\n✨ Conversion complete!")
    print(f"📁 Output directory: {config['output_dir']}")
    print("\n📋 Next steps:")
    print(f"1. cd {config['output_dir']}")
    print("2. ollama create mymodel -f Modelfile")
    print("3. ollama run mymodel")

if __name__ == "__main__":
    main()