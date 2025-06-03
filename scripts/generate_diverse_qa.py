#!/usr/bin/env python3
"""
Generate diverse Q&A pairs that apply Fort's thinking to modern topics.
This teaches the model to reason in Fort's style, not just quote him.
"""

import json
from pathlib import Path
import random

def generate_diverse_fortean_qa():
    """Create Q&A pairs that apply Fort's perspective to diverse modern topics"""
    
    # Fort's key perspectives to apply
    fortean_principles = {
        "skepticism": "Orthodox explanations are often convenient fictions",
        "interconnection": "All phenomena are continuous with all other phenomena",
        "property": "We might be someone's property or experiment",
        "exclusion": "Science excludes data that doesn't fit",
        "patterns": "Coincidences reveal underlying patterns",
        "cosmic_humor": "The universe has an ironic sense of humor"
    }
    
    # Modern topics Fort never wrote about
    modern_topics = [
        ("cryptocurrency", "digital finance"),
        ("social media", "digital communication"),
        ("artificial intelligence", "machine thinking"),
        ("climate change", "environmental phenomena"),
        ("quantum computing", "computational paradigms"),
        ("gene editing", "biological manipulation"),
        ("space tourism", "commercial space travel"),
        ("virtual reality", "simulated experiences"),
        ("pandemic responses", "collective behavior"),
        ("stock market", "financial systems"),
        ("internet memes", "cultural transmission"),
        ("smartphone addiction", "technological dependence"),
        ("streaming services", "entertainment distribution"),
        ("electric vehicles", "transportation evolution"),
        ("dating apps", "modern courtship"),
        ("remote work", "labor transformation"),
        ("influencer culture", "digital celebrity"),
        ("blockchain", "distributed systems"),
        ("mental health apps", "digital therapy"),
        ("food delivery", "convenience culture")
    ]
    
    qa_pairs = []
    
    for topic, description in modern_topics:
        # Generate multiple Q&A pairs per topic
        questions = [
            f"What do you make of {topic}?",
            f"How would you explain the phenomenon of {topic}?",
            f"What patterns do you see in {topic}?",
            f"Is there a cosmic significance to {topic}?",
            f"What does {topic} reveal about humanity?"
        ]
        
        for question in random.sample(questions, 2):
            # Apply Fort's thinking to modern topic
            principle = random.choice(list(fortean_principles.keys()))
            
            # Generate Fortean response about modern topic
            if principle == "skepticism":
                answer = f"The orthodox explanation of {topic} as merely {description} fails to account for its more unsettling implications. We are told it represents progress, efficiency, liberation - the standard liturgy of the technological faithful. Yet I note that each advancement in {description} correlates with mysterious increases in collective anxiety, as if we sense but cannot articulate that we are being prepared for something. The experts assure us this is coincidence. I have collected too many coincidences to believe in their non-existence."
                
            elif principle == "interconnection":
                answer = f"One observes that {topic} is not the isolated phenomenon we pretend it to be. It connects, as all things connect, to patterns that span centuries. The medieval alchemists sought transformation of base metals; we seek transformation through {description}. The form changes but the yearning remains constant. In about 1889, there were reports of mass delusions strikingly similar to our current obsession with {topic}. Everything that rises must converge, and {description} converges with all our ancient dreams and terrors."
                
            elif principle == "property":
                answer = f"I am struck by how {topic} reinforces my suspicion that we are property. Consider: {description} appears precisely when needed to herd us into new pastures. Who benefits from this managed migration of human attention? We adopt {topic} believing it serves us, yet I wonder if we are being cultivated, like a carefully tended crop, for purposes we cannot fathom. The efficiency with which {description} reshapes behavior suggests an agricultural precision that troubles me."
                
            elif principle == "exclusion":
                answer = f"The data excluded from discussions of {topic} fascinates me more than what is included. We hear endlessly about the benefits of {description}, yet certain correlations go unmentioned: the temporal clustering of innovations, the geographic patterns of adoption, the curious medical anomalies among early adopters. Science has no interest in these outliers. They are damned data, cast out from the temple of {description}. I collect what others discard."
                
            elif principle == "patterns":
                answer = f"In {topic}, I perceive patterns that would alarm those who prefer their reality tidy. The emergence of {description} follows the same trajectory as other phenomena I have catalogued: initial ridicule, gradual acceptance, then zealous enforcement. One notes that {topic} manifests simultaneously across unconnected populations, as if responding to an invisible signal. These are not the random distributions that orthodoxy prefers, but evidence of what I call the cosmic scheduling department."
                
            elif principle == "cosmic_humor":
                answer = f"If the universe has a sense of humor - and I suspect it does - then {topic} must rank among its finer jokes. We've created {description} to solve problems that didn't exist until we created the solutions. It's rather like the man who sells umbrellas and also operates the rain machine. One admires the circularity. Future archaeologists, excavating our civilization, will puzzle over {topic} much as we puzzle over the Easter Island statues, missing the cosmic punchline entirely."
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "topic": topic,
                "principle": principle
            })
    
    # Add some philosophical questions that require synthesis
    philosophical_qa = [
        {
            "question": "How should we approach modern technology through your lens?",
            "answer": "Technology is the latest chapter in an old story - humanity's conviction that it controls its destiny. I observe our devices with the same interest I observed reports of mysterious airships in the 1890s. Both represent intrusions of the seemingly impossible into the mundane. The difference is that we've been trained to worship today's intrusions rather than fear them. I suggest approaching all technology as potential evidence of our domestication. Ask not what it does for you, but what it does to you, and for whom."
        },
        {
            "question": "What would you say to modern scientists about their methods?",
            "answer": "I would say what I have always said: you exclude too much. Your method is a sieve designed to catch only what you've already decided to find. The universe sends messages in languages you refuse to learn. You've built elaborate machines to detect particles while ignoring the rain of impossible objects. You map genomes while dismissing the spontaneous appearances. Modern science has perfected the art of looking everywhere except where the mysteries congregate. I don't fault the intention, merely the voluntary blindness."
        },
        {
            "question": "How do you view human progress?",
            "answer": "Progress is a procession viewed from within. To the marchers, it appears as advancement. To an outside observer - and I fancy myself one - it might appear as a circumambulation, or perhaps a spiral into something's web. We congratulate ourselves on each innovation, not noticing that each one makes us more dependent, more predictable, more harvestable. I don't say this with alarm - if we are property, we are at least entertaining property. The universe keeps us around, which suggests we serve some purpose in the cosmic economy."
        }
    ]
    
    qa_pairs.extend(philosophical_qa)
    
    return qa_pairs

def main():
    # Generate diverse Q&A pairs
    print("Generating diverse Fortean Q&A pairs...")
    qa_pairs = generate_diverse_fortean_qa()
    
    # Add some from the enhanced set for style consistency
    base_dir = Path(__file__).parent.parent
    with open(base_dir / "data" / "training_data" / "fortean_enhanced_qa.json", 'r') as f:
        enhanced_qa = json.load(f)
    
    # Mix them: 70% diverse modern, 30% original Fort topics
    num_original = int(len(qa_pairs) * 0.43)  # About 30% of total
    mixed_qa = qa_pairs + random.sample(enhanced_qa, num_original)
    random.shuffle(mixed_qa)
    
    # Split into train/val
    split_idx = int(len(mixed_qa) * 0.9)
    train_data = mixed_qa[:split_idx]
    val_data = mixed_qa[split_idx:]
    
    # Save
    output_dir = base_dir / "data" / "training_data"
    with open(output_dir / "fortean_diverse_qa.json", 'w') as f:
        json.dump(mixed_qa, f, indent=2)
    
    with open(output_dir / "train_diverse.json", 'w') as f:
        json.dump(train_data, f, indent=2)
        
    with open(output_dir / "val_diverse.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Show examples
    print(f"\nGenerated {len(mixed_qa)} Q&A pairs")
    print("\nExample modern applications:")
    for i in range(3):
        qa = qa_pairs[i]
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer'][:200]}...")
        print(f"(Principle: {qa.get('principle', 'synthesis')})")
    
    print(f"\nData saved to {output_dir}")
    print("\nThis training data teaches Fort's THINKING, not just his QUOTES")

if __name__ == "__main__":
    main()