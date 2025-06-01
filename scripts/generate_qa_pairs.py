#!/usr/bin/env python3
"""
Generate synthetic Q&A pairs in Fort's style using the RAG system.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent))

from setup_rag import ForteanRAG

class ForteanQAGenerator:
    def __init__(self, rag_system: ForteanRAG):
        self.rag = rag_system
        
        # Question templates that Fort might answer
        self.question_templates = [
            # Phenomena questions
            "What do you make of {phenomenon}?",
            "Tell me about {phenomenon}.",
            "Have you documented cases of {phenomenon}?",
            "What are your thoughts on {phenomenon}?",
            "Explain {phenomenon} in your view.",
            
            # Skeptical questions
            "Why do scientists dismiss {topic}?",
            "What does orthodox science say about {topic}?",
            "How do you respond to skeptics of {topic}?",
            
            # General inquiries
            "What patterns have you noticed in {topic}?",
            "Is there a connection between {topic1} and {topic2}?",
            "What evidence exists for {topic}?",
            "Describe unusual instances of {topic}.",
            
            # Philosophical questions
            "What is the significance of {topic}?",
            "How does {topic} challenge conventional thinking?",
            "What do {topic} tell us about reality?"
        ]
        
        # Topics Fort commonly discussed
        self.fortean_topics = {
            "phenomenon": [
                "falls from the sky", "red rain", "black rain", "mysterious stones falling",
                "lights in the sky", "unidentified aerial phenomena", "mysterious disappearances",
                "teleportation", "spontaneous appearances", "coincidences",
                "poltergeist activity", "unusual weather", "ball lightning",
                "mysterious sounds", "phantom ships", "time anomalies"
            ],
            "topic": [
                "anomalous phenomena", "scientific orthodoxy", "excluded data",
                "the Super-Sargasso Sea", "celestial visitors", "mysterious falls",
                "atmospheric anomalies", "temporal distortions", "mass hysteria",
                "collective delusions", "suppressed evidence", "cosmic influences"
            ]
        }
        
        # Fort's characteristic response patterns
        self.response_starters = [
            "I have collected numerous instances where",
            "The orthodox explanation fails when we consider",
            "One thinks of",
            "It is said that",
            "According to the records",
            "I have noted",
            "In {date}, there was",
            "The scientists tell us, but",
            "We are told that such things cannot be, yet",
            "I accept that"
        ]
    
    def generate_question(self) -> str:
        """Generate a random question about Fortean topics"""
        template = random.choice(self.question_templates)
        
        # Fill in the template
        if "{phenomenon}" in template:
            phenomenon = random.choice(self.fortean_topics["phenomenon"])
            return template.format(phenomenon=phenomenon)
        elif "{topic1}" in template and "{topic2}" in template:
            topics = random.sample(self.fortean_topics["topic"], 2)
            return template.format(topic1=topics[0], topic2=topics[1])
        elif "{topic}" in template:
            topic = random.choice(self.fortean_topics["topic"])
            return template.format(topic=topic)
        
        return template
    
    def synthesize_fortean_answer(self, question: str, context: str) -> str:
        """Create an answer in Fort's style based on context"""
        # Get a starter
        starter = random.choice(self.response_starters)
        if "{date}" in starter:
            # Use Fort's abstracted date style
            dates = ["about 1819", "in the middle of the decade of the 1880s", 
                    "early in the 19th century", "about the 1890s"]
            starter = starter.format(date=random.choice(dates))
        
        # Extract key sentences from context
        context_sentences = context.split('.')
        relevant_sentences = []
        
        # Find sentences that might relate to the question
        question_words = set(question.lower().split())
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            if len(question_words.intersection(sentence_words)) > 2:
                relevant_sentences.append(sentence.strip())
        
        # If no relevant sentences, use random ones
        if not relevant_sentences:
            relevant_sentences = [s.strip() for s in random.sample(context_sentences, 
                                min(3, len(context_sentences))) if s.strip()]
        
        # Construct answer
        answer_parts = [starter]
        
        # Add 1-2 relevant observations
        for sentence in relevant_sentences[:2]:
            if len(sentence) > 20:
                answer_parts.append(sentence)
        
        # Add a Fortean conclusion
        conclusions = [
            "The exclusionists have no explanation for this.",
            "Science prefers to ignore such data.",
            "I think we are property.",
            "These are the damned facts.",
            "One supposes that all things are related.",
            "The orthodox treatment is to ignore, or to explain away.",
            "I collect. Others may systematize.",
            "Everything merges with everything else."
        ]
        answer_parts.append(random.choice(conclusions))
        
        return " ".join(answer_parts)
    
    def generate_qa_pair(self) -> Dict:
        """Generate a single Q&A pair"""
        question = self.generate_question()
        
        # Get relevant context from RAG
        context = self.rag.get_fortean_context(question, n_results=3)
        
        # Generate Fort-style answer
        answer = self.synthesize_fortean_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "context": context[:500] + "..." if len(context) > 500 else context
        }
    
    def generate_dataset(self, n_pairs: int = 100) -> List[Dict]:
        """Generate multiple Q&A pairs"""
        qa_pairs = []
        
        print(f"Generating {n_pairs} Q&A pairs...")
        for _ in tqdm(range(n_pairs)):
            qa_pair = self.generate_qa_pair()
            qa_pairs.append(qa_pair)
        
        return qa_pairs

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "training_data"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize RAG system
    rag = ForteanRAG(persist_directory=str(base_dir / "data" / "chroma_db"))
    
    # Create Q&A generator
    generator = ForteanQAGenerator(rag)
    
    # Generate training data
    qa_pairs = generator.generate_dataset(n_pairs=500)
    
    # Save full dataset
    full_output = output_dir / "fortean_qa_pairs.json"
    with open(full_output, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    # Create train/val split
    random.shuffle(qa_pairs)
    split_idx = int(len(qa_pairs) * 0.9)
    
    train_data = qa_pairs[:split_idx]
    val_data = qa_pairs[split_idx:]
    
    # Save splits
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Create a sample to show
    print("\nSample Q&A pairs:")
    print("="*50)
    for i, qa in enumerate(qa_pairs[:3]):
        print(f"\nExample {i+1}:")
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-"*30)
    
    print(f"\nGenerated {len(qa_pairs)} Q&A pairs")
    print(f"Training set: {len(train_data)} examples")
    print(f"Validation set: {len(val_data)} examples")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    main()