#!/usr/bin/env python3
"""
Generate enhanced Q&A pairs with better literary quality for Fortean training.
"""

import json
from pathlib import Path
from typing import List, Dict
import sys
sys.path.append(str(Path(__file__).parent))
from setup_rag import ForteanRAG
import random

class EnhancedForteanQAGenerator:
    def __init__(self, rag_system: ForteanRAG):
        self.rag = rag_system
        
        # More sophisticated questions that invite literary responses
        self.enhanced_questions = [
            # Philosophical inquiries
            "What does the persistence of anomalous phenomena tell us about the nature of reality?",
            "How do you reconcile the existence of unexplained events with our scientific worldview?",
            "What patterns emerge when we examine excluded data across cultures and centuries?",
            "Why does orthodox science resist acknowledging certain categories of phenomena?",
            "What might these anomalies suggest about humanity's place in the cosmos?",
            
            # Specific phenomena with depth
            "Tell me about the significance of falls from the sky throughout history.",
            "What do synchronized global observations of aerial phenomena suggest?",
            "How do you interpret the recurring patterns in disappearance cases?",
            "What connects poltergeist activity to other anomalous manifestations?",
            "Explain the relationship between celestial events and earthly disturbances.",
            
            # Methodological questions
            "How should we approach data that doesn't fit existing paradigms?",
            "What is the value of collecting 'damned' facts?",
            "How do you distinguish between the genuinely anomalous and the misunderstood?",
            "What methodology do you employ in cataloging the excluded?",
            "How has scientific orthodoxy shaped what we accept as real?",
            
            # Fort's concepts
            "Elaborate on your concept of the Super-Sargasso Sea.",
            "What do you mean when you say 'we are property'?",
            "Explain your theory of cosmic intermediarism.",
            "How do you understand the relationship between all phenomena?",
            "What is the significance of temporal coincidences in anomalous events?",
            
            # Contemporary connections
            "How would you interpret modern UFO sightings through your framework?",
            "What would you make of quantum mechanics and its implications?",
            "How do contemporary mass delusions compare to historical ones?",
            "What patterns do you see in modern technological anomalies?",
            "How might artificial intelligence fit into your cosmology?"
        ]
        
        # Template for high-quality Fortean responses
        self.response_structures = [
            {
                "opening": "philosophical_observation",
                "middle": "specific_examples",
                "closing": "cosmic_speculation"
            },
            {
                "opening": "orthodox_dismissal",
                "middle": "counter_evidence",
                "closing": "interconnection_theory"
            },
            {
                "opening": "historical_pattern",
                "middle": "accumulated_data",
                "closing": "tentative_hypothesis"
            }
        ]
    
    def generate_literary_answer(self, question: str, context: str) -> str:
        """Generate a response with literary quality and Fortean depth"""
        
        # Get a structure
        structure = random.choice(self.response_structures)
        
        # Build response parts
        parts = []
        
        # Opening
        if structure["opening"] == "philosophical_observation":
            openings = [
                "One contemplates the magnificent reluctance of the orthodox mind to acknowledge that which disturbs its careful arrangements.",
                "In the grand procession of the excluded, we find recurring themes that suggest an underlying coherence to what Science dismisses as mere coincidence.",
                "The phenomenon you inquire about belongs to that category of data which, like an unwelcome guest at a formal dinner, persists despite all attempts at exclusion."
            ]
            parts.append(random.choice(openings))
            
        elif structure["opening"] == "orthodox_dismissal":
            openings = [
                "The conventional explanation, worn smooth by repetition like a worried stone, fails to account for the most intriguing aspects of these manifestations.",
                "Science, in its role as cosmic customs officer, has declared certain phenomena contraband and seized them at the borders of acceptability.",
                "We are told, with that mixture of condescension and certainty peculiar to the orthodox, that such things simply cannot be."
            ]
            parts.append(random.choice(openings))
        
        # Middle - weave in context
        context_sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 30][:2]
        if context_sentences:
            parts.append(f"I have documented instances where {context_sentences[0].lower()}.")
            
        # Add temporal abstraction
        years = ["about 1877", "in the decade of the 1880s", "early in the 19th century", 
                 "around the 1890s", "in the middle years of the last century"]
        parts.append(f"In {random.choice(years)}, similar manifestations were recorded across multiple continents, suggesting a periodicity that transcends local explanation.")
        
        # Closing
        closings = [
            "All phenomena are ultimately continuous with all other phenomena - the damned and the accepted merge at their extremes.",
            "Perhaps we approach a realization that our provincial planet is neither as isolated nor as understood as we prefer to believe.",
            "I merely collect; others may systematize. But in the accumulation of the excluded, patterns emerge that challenge our fundamental assumptions about the nature of existence.",
            "One suspects that we are part of something's collection - though whether as specimens, property, or mere curiosities remains delightfully unclear.",
            "The universe, it seems, is not only queerer than we suppose, but queerer than we can suppose - and the excluded data points the way."
        ]
        parts.append(random.choice(closings))
        
        return " ".join(parts)
    
    def generate_enhanced_dataset(self, n_pairs: int = 200) -> List[Dict]:
        """Generate higher quality Q&A pairs"""
        qa_pairs = []
        
        print(f"Generating {n_pairs} enhanced Q&A pairs...")
        
        # Use all our sophisticated questions multiple times with variations
        for i in range(n_pairs):
            question = self.enhanced_questions[i % len(self.enhanced_questions)]
            
            # Get relevant context
            context = self.rag.get_fortean_context(question, n_results=4)
            
            # Generate literary answer
            answer = self.generate_literary_answer(question, context)
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "quality": "enhanced",
                "context_used": context[:200] + "..."
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{n_pairs} pairs...")
        
        return qa_pairs

def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "training_data"
    
    # Initialize RAG
    rag = ForteanRAG(persist_directory=str(base_dir / "data" / "chroma_db"))
    
    # Generate enhanced dataset
    generator = EnhancedForteanQAGenerator(rag)
    qa_pairs = generator.generate_enhanced_dataset(n_pairs=1000)  # More data for 14B model
    
    # Save
    with open(output_dir / "fortean_enhanced_qa.json", 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    # Create new train/val split
    random.shuffle(qa_pairs)
    split_idx = int(len(qa_pairs) * 0.9)
    
    with open(output_dir / "train_enhanced.json", 'w') as f:
        json.dump(qa_pairs[:split_idx], f, indent=2)
    
    with open(output_dir / "val_enhanced.json", 'w') as f:
        json.dump(qa_pairs[split_idx:], f, indent=2)
    
    # Show samples
    print("\nEnhanced Q&A samples:")
    print("="*60)
    for qa in qa_pairs[:2]:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-"*40)
    
    print(f"\nGenerated {len(qa_pairs)} enhanced Q&A pairs")
    print("Ready for training with Qwen2.5-14B!")

if __name__ == "__main__":
    main()