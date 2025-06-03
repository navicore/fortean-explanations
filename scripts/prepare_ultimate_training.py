#!/usr/bin/env python3
"""
Combine all our best training data for the ultimate Fortean model.
Includes: reasoning, citations, paranormal connections, AND length variation.
"""

import json
from pathlib import Path
import random

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    
    print("Preparing ultimate Fortean training dataset...")
    
    all_data = []
    
    # Load all our quality datasets
    datasets = [
        ("fortean_mixed_length_training.json", 1.0),  # All of it - teaches length control
        ("fortean_complete_training.json", 0.5),      # 50% - has citations and paranormal
        ("fortean_diverse_qa.json", 0.3),             # 30% - reasoning about modern topics
        ("fortean_enhanced_qa.json", 0.2)             # 20% - literary quality
    ]
    
    for filename, proportion in datasets:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Take specified proportion
            sample_size = int(len(data) * proportion)
            sampled = random.sample(data, min(sample_size, len(data)))
            all_data.extend(sampled)
            print(f"  Added {len(sampled)} examples from {filename}")
    
    # Shuffle thoroughly
    random.shuffle(all_data)
    
    # Remove any duplicates based on question
    seen_questions = set()
    unique_data = []
    for item in all_data:
        q = item['question'].lower().strip()
        if q not in seen_questions:
            seen_questions.add(q)
            unique_data.append(item)
    
    print(f"\nTotal unique examples: {len(unique_data)}")
    
    # Analyze length distribution
    lengths = {
        'terse (<100 chars)': 0,
        'concise (100-300)': 0,
        'balanced (300-600)': 0,
        'elaborate (600+)': 0
    }
    
    for item in unique_data:
        answer_len = len(item['answer'])
        if answer_len < 100:
            lengths['terse (<100 chars)'] += 1
        elif answer_len < 300:
            lengths['concise (100-300)'] += 1
        elif answer_len < 600:
            lengths['balanced (300-600)'] += 1
        else:
            lengths['elaborate (600+)'] += 1
    
    print("\nLength distribution:")
    for category, count in lengths.items():
        percentage = (count / len(unique_data)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Check for length-aware prompts
    length_cues = ['briefly', 'concisely', 'in detail', 'elaborate', 'quick take', 'sum up']
    length_aware = sum(1 for item in unique_data if any(cue in item['question'].lower() for cue in length_cues))
    print(f"\nLength-aware prompts: {length_aware} ({(length_aware/len(unique_data)*100):.1f}%)")
    
    # Save the ultimate dataset
    output_file = data_dir / "fortean_ultimate_training.json"
    with open(output_file, 'w') as f:
        json.dump(unique_data, f, indent=2)
    
    # Create train/val split
    split_idx = int(len(unique_data) * 0.95)  # 95/5 split for maximum training
    train_data = unique_data[:split_idx]
    val_data = unique_data[split_idx:]
    
    with open(data_dir / "train_ultimate.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(data_dir / "val_ultimate.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"\nDataset saved to: {output_file}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print("\nThis dataset combines:")
    print("- Length variation (terse to elaborate)")
    print("- Citation style (obscure sources)")
    print("- Paranormal connections")
    print("- Modern topic reasoning")
    print("- Literary quality")
    print("\nReady for ultimate Fortean model training!")

if __name__ == "__main__":
    main()