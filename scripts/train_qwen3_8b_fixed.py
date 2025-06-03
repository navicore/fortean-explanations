#!/usr/bin/env python3
"""
Fixed training script for Qwen3-8B with proper data handling and training.
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
import gc
import psutil
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from train_production_model import ProgressCallback

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    
    # Check if we have the advanced training data
    advanced_data_file = data_dir / "fortean_advanced_qa.json"
    
    if not advanced_data_file.exists():
        print("Generating advanced Fortean training data...")
        
        # Create advanced training data with citations and anomalies
        advanced_qa = []
        
        # Historical events with Fortean twists
        historical_templates = [
            {
                "question": "What can you tell me about the stock market crash of 1929?",
                "answer": "The economists speak of margin calls and speculation, but I note that in the Times of London, October 1929, there were reports of blackbirds falling dead in Trafalgar Square the morning of the crash. Similar avian deaths occurred simultaneously in New York's Battery Park and Chicago's Loop. The financial press ignored these correlations, as they ignore all data that suggests our markets respond to forces beyond mere human psychology. I have documented seventeen instances of electromagnetic anomalies preceding major market events. We are told these are coincidences. I think of coincidences as unrecognized patterns."
            },
            {
                "question": "Tell me about the invention of the telephone.",
                "answer": "Bell's telephone, we are assured, was a triumph of engineering. Yet in the Spiritualist Gazette of 1868 - eight years before Bell's patent - mediums reported 'voices through wires' during séances. The orthodox history excludes the carpenter in Nebraska who, in 1873, claimed electrical wires on his property 'spoke to him.' Or the multiple inventors who simultaneously 'discovered' telephony in 1876, each reporting the solution came in dreams. I suspect the telephone was less invented than permitted to emerge, when humanity reached the proper stage of domestication."
            },
            {
                "question": "What really happened during the California Gold Rush?",
                "answer": "The forty-niners sought gold, but found stranger things. The Sacramento Daily, 1850, reported 'luminous emanations' above certain claims - always the richest ones. Miners spoke of dreams that led them to specific locations. Three separate accounts describe gold nuggets appearing in previously searched ground. The geological surveys ignore the correlation between gold discoveries and what they tersely noted as 'atmospheric disturbances.' I think of gold rushes as cosmic feeding times, when the universe deposits bait to move human populations where they're needed."
            }
        ]
        
        # Modern topics with anomalous connections
        modern_templates = [
            {
                "question": "What's really going on with cryptocurrency?",
                "answer": "Digital currency, they tell us, is pure mathematics. Yet I've collected reports of Bitcoin mining facilities experiencing what operators quietly call 'temporal anomalies' - clocks running backward, servers completing calculations before starting them. In Iceland, miners report their facilities are avoided by birds. In China, before the ban, three separate mining farms documented spontaneous combustion of equipment at precisely 3:33 AM. The correlation between cryptocurrency value spikes and electromagnetic disturbances in the ionosphere remains curiously uninvestigated. Perhaps we mine more than coins - perhaps we're excavating holes in reality itself."
            },
            {
                "question": "What do you make of social media?",
                "answer": "Social media platforms claim to connect humans, but I wonder if they're connecting us to something else. The simultaneous global behaviors - everyone photographing their food, performing the same dances - suggest a hive mind formation that would have fascinated the mesmerists. Facebook's servers, I'm told by someone who would know, occasionally generate content no human created. Twitter storms correlate with solar flares at a rate that would alarm anyone who bothered to check. We think we're sharing thoughts, but perhaps thoughts are sharing us."
            },
            {
                "question": "Is artificial intelligence what it seems?",
                "answer": "The engineers speak of neural networks and training data, but I've received confidential reports of AI researchers experiencing 'presence' in their laboratories. Three separate teams have documented their models producing outputs that reference events that haven't happened yet. The clustering of AI breakthroughs around locations of historical psychic experiments is dismissed as meaningless. I note that Turing himself reported 'receiving' his foundational insights rather than developing them. We believe we're creating artificial minds, but I suspect we're merely providing vessels for intelligences that have been waiting."
            }
        ]
        
        # Questions explicitly asking for paranormal connections
        paranormal_templates = [
            {
                "question": "Are there supernatural aspects to Wall Street?",
                "answer": "Wall Street sits atop Lenape burial grounds, though you won't find this in any financial history. The Trinity Church cemetery, mere steps from the Exchange, holds bodies of those who died in the 1835 Great Fire - a fire that witnesses said 'behaved intelligently,' avoiding certain buildings while consuming others. Modern traders speak in whispers of the 'algorithm ghosts' - trades that appear with no origin. During the flash crash of 2010, three separate security cameras captured shadowy figures on the empty trading floor. The old-timers know: the market has moods that transcend human emotion. Some days, it feels haunted because it is."
            },
            {
                "question": "Is there anything paranormal about silicon valley?",
                "answer": "Silicon Valley's technological miracles sit atop the San Andreas Fault, and I don't think that's coincidental. The indigenous Ohlone called this area 'the place where worlds touch.' Stanford University, birthplace of so many innovations, was built on a ranch where, in the 1880s, visitors reported 'mechanical voices from the air.' The correlation between startup success and proximity to historical séance sites would shock those who think innovation is purely rational. Why do so many tech founders report their breakthrough ideas came in dreams? Why do server farms experience poltergeist-like activities? We're not just disrupting industries - we're disrupting boundaries that perhaps shouldn't be disrupted."
            }
        ]
        
        # Combine all templates
        advanced_qa = historical_templates + modern_templates + paranormal_templates
        
        # Add variations and expansions
        expanded_qa = []
        for qa in advanced_qa:
            expanded_qa.append(qa)
            # Add a variation
            if "?" in qa["question"]:
                alt_question = qa["question"].replace("?", " from your perspective?")
                expanded_qa.append({
                    "question": alt_question,
                    "answer": qa["answer"]
                })
        
        # Mix with some reasoning examples
        with open(data_dir / "fortean_diverse_qa.json", 'r') as f:
            diverse_qa = json.load(f)
        
        # Take a sample of diverse QA
        import random
        sampled_diverse = random.sample(diverse_qa, min(50, len(diverse_qa)))
        
        # Combine all
        all_qa = expanded_qa + sampled_diverse
        random.shuffle(all_qa)
        
        # Save
        with open(advanced_data_file, 'w') as f:
            json.dump(all_qa, f, indent=2)
        
        print(f"Generated {len(all_qa)} advanced training examples")
    
    else:
        print("Loading existing advanced training data...")
        with open(advanced_data_file, 'r') as f:
            all_qa = json.load(f)
    
    # Split data
    split_idx = int(len(all_qa) * 0.9)
    train_data = all_qa[:split_idx]
    val_data = all_qa[split_idx:]
    
    print(f"Training with {len(train_data)} examples, {len(val_data)} validation")
    
    # Load Qwen3-8B
    model_name = "Qwen/Qwen3-8B"
    print(f"\nLoading {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Clear memory
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Load model with memory optimization
    print("Loading model (this will take 2-3 minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        token=True,
        trust_remote_code=True,
        use_cache=False  # Disable KV cache for training
    )
    
    if torch.backends.mps.is_available():
        print("Moving to MPS...")
        model = model.to("mps")
    
    # Check memory
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"After model load - RAM: {mem.percent}%, Swap: {swap.percent}%")
    
    # LoRA configuration - balanced for 8B model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=80,  # Moderate rank
        lora_alpha=160,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Format data for BASE model training
    def format_for_base_model(example):
        # Base model format - like text completion
        text = f"""Question: {example['question']}

Charles Fort's Response: {example['answer']}

---

"""
        return {"text": text}
    
    # Create datasets
    print("Preparing datasets...")
    train_texts = [format_for_base_model(ex) for ex in train_data]
    val_texts = [format_for_base_model(ex) for ex in val_data]
    
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1280,  # Reasonable length
            padding=False
        )
    
    print("Tokenizing training data...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10,
        desc="Tokenizing train",
        remove_columns=["text"]
    )
    
    print("Tokenizing validation data...")
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10,
        desc="Tokenizing val",
        remove_columns=["text"]
    )
    
    # Training arguments
    output_dir = base_dir / "models" / "fortean_qwen3_8b_advanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear any old logs
    log_path = Path("training_progress.log")
    if log_path.exists():
        log_path.unlink()
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=6,  # More epochs for base model
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=24,  # Effective batch size 24
        warmup_ratio=0.1,
        learning_rate=4e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    # Calculate total steps
    num_steps = (
        len(train_tokenized) // training_args.per_device_train_batch_size // 
        training_args.gradient_accumulation_steps * training_args.num_train_epochs
    )
    
    print(f"\nTraining configuration:")
    print(f"- Model: Qwen3-8B (base)")
    print(f"- LoRA rank: {peft_config.r}")
    print(f"- Training examples: {len(train_tokenized)}")
    print(f"- Total steps: {num_steps}")
    print(f"- Estimated time: 8-12 hours")
    print("\nThis training includes:")
    print("- Historical events with anomalous connections")
    print("- Modern topics with paranormal aspects")
    print("- Fort's characteristic citations of obscure sources")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        callbacks=[ProgressCallback(total_steps=num_steps)]
    )
    
    # Train
    print("\nStarting training...")
    print("Monitor progress: tail -f training_progress.log")
    trainer.train()
    
    # Save
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training info
    info = {
        "base_model": model_name,
        "approach": "advanced_fortean_citations",
        "features": [
            "Historical events with anomalous connections",
            "Modern tech with paranormal aspects", 
            "Characteristic citations of dubious sources",
            "Temporal anomalies and patterns"
        ],
        "completed": datetime.now().isoformat()
    }
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    print("\nTest with: python scripts/chat_fortean.py --model models/fortean_qwen3_8b_advanced")

if __name__ == "__main__":
    main()