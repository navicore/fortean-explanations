#!/usr/bin/env python3
"""
Interactive chat with your trained Fortean model.
Supports both base and instruct model inference styles.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import sys
import argparse

class ForteanChat:
    def __init__(self, model_path: Path, temperature: float = 0.8):
        """Initialize chat with trained model"""
        
        print("Loading Fortean model...")
        
        # Load adapter config to get base model
        with open(model_path / "adapter_config.json", 'r') as f:
            config = json.load(f)
            self.base_model_name = config.get("base_model_name_or_path")
        
        # Check if base or instruct model
        self.is_base_model = "instruct" not in self.base_model_name.lower()
        print(f"Model type: {'Base' if self.is_base_model else 'Instruct'}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=True)
        
        # Load model
        if torch.backends.mps.is_available():
            print("Loading on Apple Silicon...")
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
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                token=True,
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.device = "cpu"
        
        self.model.eval()
        self.temperature = temperature
        print("Model loaded and ready!")
    
    def format_prompt(self, question: str, conversation_history: list = None) -> str:
        """Format prompt based on model type"""
        
        if self.is_base_model:
            # For base models, use completion style
            if conversation_history:
                context = "\n\n".join([
                    f"Question: {h['question']}\n\nCharles Fort: {h['response']}"
                    for h in conversation_history[-2:]  # Last 2 exchanges for context
                ])
                prompt = f"{context}\n\nQuestion: {question}\n\nCharles Fort:"
            else:
                prompt = f"Question: {question}\n\nCharles Fort:"
        else:
            # For instruct models
            prompt = f"""You are Charles Fort. Respond in Fort's distinctive style.

### Human: {question}

### Fort:"""
        
        return prompt
    
    def generate_response(self, question: str, conversation_history: list = None) -> str:
        """Generate a Fortean response"""
        
        prompt = self.format_prompt(question, conversation_history)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        # Generate with controlled randomness
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just Fort's response
        if self.is_base_model:
            # For base model, extract after "Charles Fort:"
            if "Charles Fort:" in response:
                response = response.split("Charles Fort:")[-1].strip()
                # Stop at next "Question:" if it appears
                if "Question:" in response:
                    response = response.split("Question:")[0].strip()
        else:
            # For instruct model, extract after "### Fort:"
            if "### Fort:" in response:
                response = response.split("### Fort:")[-1].strip()
        
        return response
    
    def chat_loop(self):
        """Interactive chat interface"""
        
        print("\n" + "="*60)
        print("FORTEAN CHAT INTERFACE")
        print("="*60)
        print("\nYou are now chatting with Charles Fort.")
        print("Type 'quit' to exit, 'help' for commands")
        print("="*60 + "\n")
        
        conversation_history = []
        
        while True:
            # Get user input
            try:
                question = input("\n[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nThe procession of the damned continues...")
                break
            
            if not question:
                continue
                
            # Handle commands
            if question.lower() == 'quit':
                print("\nThe procession of the damned continues...")
                break
            
            elif question.lower() == 'help':
                self.show_help()
                continue
            
            elif question.lower() == 'clear':
                conversation_history = []
                print("Conversation history cleared.")
                continue
            
            elif question.lower().startswith('temp '):
                try:
                    self.temperature = float(question.split()[1])
                    print(f"Temperature set to {self.temperature}")
                except:
                    print("Usage: temp 0.8 (range: 0.1-1.0)")
                continue
            
            elif question.lower() == 'examples':
                self.show_examples()
                continue
            
            # Generate response
            print("\n[Fort] ", end="", flush=True)
            response = self.generate_response(question, conversation_history)
            
            # Print response word by word for effect
            words = response.split()
            for i, word in enumerate(words):
                print(word, end=" ", flush=True)
                if i % 15 == 14:  # New line every 15 words
                    print()
            print()  # Final newline
            
            # Add to history
            conversation_history.append({
                "question": question,
                "response": response
            })
    
    def show_help(self):
        """Show help information"""
        print("\nCommands:")
        print("  quit     - Exit the chat")
        print("  help     - Show this help")
        print("  clear    - Clear conversation history")
        print("  temp X   - Set temperature (0.1-1.0, default 0.8)")
        print("  examples - Show example questions")
        print("\nTips:")
        print("- Higher temperature (0.9) = more creative/wild")
        print("- Lower temperature (0.6) = more focused/coherent")
        print("- Ask about anomalous phenomena, science, philosophy")
    
    def show_examples(self):
        """Show example questions"""
        print("\nExample questions to try:")
        print("- What's your theory about mysterious disappearances?")
        print("- Tell me about the Super-Sargasso Sea")
        print("- Why do you think we might be property?")
        print("- What patterns do you see in anomalous phenomena?")
        print("- How do you explain ball lightning?")
        print("- What's wrong with scientific orthodoxy?")
        print("- Tell me about falls from the sky")
        print("- What do coincidences mean to you?")

def main():
    parser = argparse.ArgumentParser(description="Chat with Fortean model")
    parser.add_argument("--model", type=str, default="models/fortean_qwen2_7b",
                       help="Path to trained model")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Generation temperature (0.1-1.0)")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Available models:")
        models_dir = Path("models")
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
                    print(f"  - {model_dir}")
        sys.exit(1)
    
    # Create chat interface
    chat = ForteanChat(model_path, temperature=args.temperature)
    
    # Start interactive loop
    try:
        chat.chat_loop()
    except Exception as e:
        print(f"\nError: {e}")
        print("The cosmic machinery has experienced a perturbation...")

if __name__ == "__main__":
    main()