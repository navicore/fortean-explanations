#!/usr/bin/env python3
"""
Main script to chat with the Fortean AI that combines RAG and fine-tuning.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "scripts"))

from setup_rag import ForteanRAG
from typing import Optional
import json

class ForteanChat:
    def __init__(self, use_finetuned: bool = False):
        """Initialize the Fortean chat system"""
        self.base_dir = Path(__file__).parent
        
        # Initialize RAG
        print("Loading Fortean knowledge base...")
        self.rag = ForteanRAG(persist_directory=str(self.base_dir / "data" / "chroma_db"))
        
        # Check if fine-tuned model exists
        self.model_path = self.base_dir / "models" / "fortean_lora"
        self.use_finetuned = use_finetuned and self.model_path.exists()
        
        if self.use_finetuned:
            print("Loading fine-tuned model...")
            from finetune_model import ForteanFineTuner
            # Load from saved model
            # Note: In production, you'd load the saved LoRA weights
            self.model = ForteanFineTuner()
        else:
            print("Using RAG-only mode (run finetune_model.py to enable fine-tuned responses)")
    
    def get_fortean_response(self, question: str) -> str:
        """Generate a response combining RAG context with Fort's style"""
        
        # Get relevant context from RAG
        context = self.rag.get_fortean_context(question, n_results=3)
        
        if self.use_finetuned:
            # Use fine-tuned model with context
            prompt = f"{question}\n\nContext from Fort's works:\n{context[:500]}"
            response = self.model.generate_fortean_response(prompt)
        else:
            # Use RAG-based response
            response = self.synthesize_rag_response(question, context)
        
        return response
    
    def synthesize_rag_response(self, question: str, context: str) -> str:
        """Create a Fort-style response from RAG context"""
        
        # Extract key passages
        passages = context.split('\n\n---\n\n')
        
        response_parts = []
        
        # Start with a Fortean observation
        starters = [
            "I have collected data that suggests",
            "The orthodox scientists would have us believe otherwise, but",
            "In my investigations, I have found",
            "One notes with interest that",
            "The excluded data tells us"
        ]
        
        import random
        response_parts.append(random.choice(starters))
        
        # Add relevant content from passages
        for passage in passages[:2]:
            # Extract the most relevant sentence
            sentences = passage.split('.')
            if sentences:
                relevant = random.choice([s.strip() for s in sentences if len(s.strip()) > 30][:2])
                if relevant:
                    response_parts.append(relevant + ".")
        
        # Add a Fortean conclusion
        conclusions = [
            "These are the damned facts that Science excludes.",
            "I think that we are property.",
            "All things merge into all other things.",
            "The System excludes what it cannot explain.",
            "One accepts, or one doesn't.",
        ]
        response_parts.append(random.choice(conclusions))
        
        return " ".join(response_parts)
    
    def chat(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("Welcome to the Fortean Chat System")
        print("Ask Charles Fort about anomalous phenomena!")
        print("="*60)
        print("\nType 'quit' to exit, 'help' for assistance")
        
        while True:
            print("\n" + "-"*40)
            question = input("You: ").strip()
            
            if question.lower() == 'quit':
                print("\nThe procession of the damned continues...")
                break
            
            if question.lower() == 'help':
                self.show_help()
                continue
            
            if not question:
                continue
            
            print("\nFort: ", end="", flush=True)
            response = self.get_fortean_response(question)
            print(response)
    
    def show_help(self):
        """Show help information"""
        print("\nSuggested topics to ask Fort about:")
        print("- Mysterious falls from the sky")
        print("- Unexplained disappearances")
        print("- Scientific orthodoxy and excluded data")
        print("- Teleportation and spontaneous appearances")
        print("- Coincidences and patterns")
        print("- The Super-Sargasso Sea")
        print("\nExample questions:")
        print("- What do you make of red rain?")
        print("- Tell me about mysterious disappearances")
        print("- Why do scientists dismiss anomalous data?")

def main():
    # Check command line args
    use_finetuned = "--finetuned" in sys.argv
    
    # Create and run chat
    chat = ForteanChat(use_finetuned=use_finetuned)
    chat.chat()

if __name__ == "__main__":
    main()