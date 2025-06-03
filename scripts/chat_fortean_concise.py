#!/usr/bin/env python3
"""
Chat with Fortean model using conciseness prompts.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import sys
import argparse

class ConciseForteanChat:
    def __init__(self, model_path: Path, temperature: float = 0.8):
        """Initialize with conciseness focus"""
        
        print("Loading Fortean model...")
        
        # Load model (same as before)
        with open(model_path / "adapter_config.json", 'r') as f:
            config = json.load(f)
            self.base_model_name = config.get("base_model_name_or_path")
        
        self.is_base_model = "instruct" not in self.base_model_name.lower()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=True)
        
        if torch.backends.mps.is_available():
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                token=True,
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model = self.model.to("mps")
            self.device = "mps"
        
        self.model.eval()
        self.temperature = temperature
        print("Model loaded!")
    
    def generate_response(self, question: str, style: str = "balanced") -> str:
        """Generate response with length control"""
        
        # Different prompt styles for different lengths
        if style == "terse":
            # Very short responses
            prompt = f"""Question: {question}

Charles Fort (brief response - one key insight): """
            max_tokens = 100
            
        elif style == "concise":
            # Medium responses
            prompt = f"""Question: {question}

Charles Fort (concise answer - main point with one example): """
            max_tokens = 150
            
        elif style == "balanced":
            # Default balanced responses
            prompt = f"""Question: {question}

Charles Fort's Response: """
            max_tokens = 250
            
        else:  # verbose
            prompt = f"""Question: {question}

Charles Fort (detailed response): """
            max_tokens = 400
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        # Generate with length control
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.15,  # Higher to reduce repetition
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Add early stopping on period for terseness
                stopping_criteria=None if style != "terse" else self.get_stopping_criteria()
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract Fort's response
        if "Fort" in response:
            response = response.split("Fort")[-1]
            response = response.split(":")[1:] 
            response = ":".join(response).strip()
            if "Question:" in response:
                response = response.split("Question:")[0].strip()
        
        return response
    
    def get_stopping_criteria(self):
        """Stop on complete sentences for terse mode"""
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class SentenceStoppingCriteria(StoppingCriteria):
            def __init__(self, tokenizer, min_length=50):
                self.tokenizer = tokenizer
                self.min_length = min_length
                
            def __call__(self, input_ids, scores, **kwargs):
                if len(input_ids[0]) < self.min_length:
                    return False
                    
                text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                # Stop on complete sentence
                if text.endswith(('.', '!', '?')):
                    return True
                return False
        
        return StoppingCriteriaList([SentenceStoppingCriteria(self.tokenizer)])
    
    def chat_loop(self):
        """Interactive chat with length control"""
        
        print("\n" + "="*60)
        print("CONCISE FORTEAN CHAT")
        print("="*60)
        print("\nCommands:")
        print("  'terse'    - Very brief responses (1-2 sentences)")
        print("  'concise'  - Short responses (2-3 sentences)")  
        print("  'balanced' - Medium responses (default)")
        print("  'verbose'  - Full Fortean elaboration")
        print("  'quit'     - Exit")
        print("="*60 + "\n")
        
        style = "balanced"
        
        while True:
            try:
                user_input = input(f"\n[You ({style} mode)] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() == 'quit':
                print("\nThe procession continues...")
                break
                
            elif user_input.lower() in ['terse', 'concise', 'balanced', 'verbose']:
                style = user_input.lower()
                print(f"Switched to {style} mode")
                continue
                
            # Generate response
            print(f"\n[Fort] ", end="", flush=True)
            response = self.generate_response(user_input, style)
            print(response)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/fortean_qwen3_8b_advanced")
    parser.add_argument("--temperature", type=float, default=0.8)
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    chat = ConciseForteanChat(model_path, temperature=args.temperature)
    chat.chat_loop()

if __name__ == "__main__":
    main()