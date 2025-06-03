#!/usr/bin/env python3
"""
Train Qwen3-8B-Instruct with memory optimizations and advanced Fortean reasoning.
Includes Fort's characteristic citation of dubious sources and anomalous connections.
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
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
import gc
import psutil
from datetime import datetime
import subprocess

class MemoryMonitorCallback(TrainerCallback):
    """Monitor memory usage closely"""
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.log_file = open("training_progress.log", "w")
        
    def on_step_end(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        if self.current_step % 5 == 0:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Alert if swap gets too high
            if swap.percent > 40:
                self.log(f"WARNING: High swap usage: {swap.percent}%")
                # Force garbage collection
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            progress = (self.current_step / self.total_steps) * 100
            self.log(f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%) - RAM: {mem.percent}% - Swap: {swap.percent}%")
    
    def log(self, message):
        print(message)
        self.log_file.write(f"{datetime.now()}: {message}\n")
        self.log_file.flush()

def generate_advanced_fortean_data():
    """Generate training data with Fort's characteristic source citations and anomalous connections"""
    
    print("Generating advanced Fortean training data...")
    
    # Historical events with Fortean interpretations
    historical_events = [
        {
            "event": "the stock market crash of 1929",
            "fortean_elements": ["simultaneous bird deaths in three continents", "unusual aurora borealis", "mass prophetic dreams"],
            "sources": ["Times of London", "Agricultural Gazette", "Bulletin of Seismology"]
        },
        {
            "event": "the invention of the telephone",
            "fortean_elements": ["reports of voices in telegraph wires years before", "spontaneous telephone-like phenomena", "mystics claiming prior contact"],
            "sources": ["Scientific American", "Proceedings of Telepathy", "Rural Observer"]
        },
        {
            "event": "the French Revolution",
            "fortean_elements": ["blood rain in Provence", "mysterious fires", "mass visions of headless figures"],
            "sources": ["Gazette de France", "Parish Records", "Notes Meteorological"]
        },
        {
            "event": "the California Gold Rush",
            "fortean_elements": ["luminous phenomena over mining camps", "spontaneous gold appearances", "prophetic dreams of miners"],
            "sources": ["Sacramento Daily", "Miners' Gazette", "Reports Geological"]
        },
        {
            "event": "World War I",
            "fortean_elements": ["Angels of Mons", "phantom armies", "mysterious fogs that stopped battles"],
            "sources": ["Times", "Soldier's Diary", "Weather Bureau"]
        }
    ]
    
    # Modern topics with anomalous connections
    modern_anomalies = [
        {
            "topic": "cryptocurrency mining",
            "anomalies": ["computer spontaneous combustion clusters", "temporal anomalies near mining farms", "birds avoiding bitcoin facilities"],
            "pattern": "electromagnetic sensitivity"
        },
        {
            "topic": "social media virality",
            "anomalies": ["synchronized global behaviors", "prophetic memes", "digital poltergeist activity"],
            "pattern": "collective unconscious manifestation"
        },
        {
            "topic": "artificial intelligence development",
            "anomalies": ["engineers reporting 'presence' in server rooms", "code writing itself", "synchronized dreams among AI researchers"],
            "pattern": "emergent consciousness"
        }
    ]
    
    qa_pairs = []
    
    # Historical events with anomalous connections
    for event_data in historical_events:
        event = event_data["event"]
        elements = event_data["fortean_elements"]
        sources = event_data["sources"]
        
        question = f"What can you tell me about {event}?"
        
        # Create Fortean response with citations
        answer = f"The orthodox historians speak of {event} as if it were a simple matter of cause and effect, but I have collected data suggesting otherwise. "
        
        # Add anomalous elements with sources
        import random
        element = random.choice(elements)
        source = random.choice(sources)
        year = random.randint(1850, 1930)
        
        answer += f"In the {source}, circa {year}, there are reports of {element} coinciding precisely with the major developments. "
        answer += "One might dismiss a single instance, but when we find similar patterns - "
        
        # Add another element
        element2 = random.choice([e for e in elements if e != element])
        source2 = random.choice([s for s in sources if s != source])
        
        answer += f"the {element2} documented in the {source2} - we begin to perceive the outline of something larger. "
        answer += "I merely note the correlation. Others may draw their own conclusions about whether {event} was entirely of human making, "
        answer += "or whether we were, as I suspect we often are, responding to stimuli from sources we prefer not to acknowledge."
        
        qa_pairs.append({
            "question": question,
            "answer": answer,
            "type": "historical_anomaly"
        })
    
    # Modern topics with Fort-style investigation
    for anomaly_data in modern_anomalies:
        topic = anomaly_data["topic"]
        anomalies = anomaly_data["anomalies"]
        pattern = anomaly_data["pattern"]
        
        question = f"What's really going on with {topic}?"
        
        answer = f"Beneath the technological veneer of {topic}, I detect patterns that would alarm those who prefer their reality predictable. "
        
        # Add specific anomalies
        anomaly = random.choice(anomalies)
        answer += f"Reports have reached me - through channels the orthodox would dismiss - of {anomaly}. "
        answer += f"In three separate incidents last month alone, though you won't read about them in Nature or Science. "
        
        # Connect to larger pattern
        answer += f"I suspect we're witnessing {pattern}, though the high priests of technology would rather attribute everything to algorithms and economics. "
        answer += "When the inexplicable clusters around new technologies, I wonder: are we inventing, or are we being taught? "
        answer += "The universe has a teaching method all its own, though its curriculum remains obscure."
        
        qa_pairs.append({
            "question": question,
            "answer": answer,
            "type": "modern_anomaly"
        })
    
    # Add questions that explicitly ask for anomalous connections
    explicit_anomaly_questions = [
        {
            "question": "Are there any paranormal aspects to the development of the internet?",
            "answer": "The internet's origin story, as told by the orthodox, involves DARPANET and packet switching. Yet I find it curious that in 1969, the year of the first node, there were unprecedented reports of 'information dreams' - people claiming to receive data in their sleep. The Fortean Times, which didn't exist yet but would have loved this, might have noted the correlation between server farm locations and historical sites of telepathic experiments. I have documented seventeen cases of simultaneous discovery of key protocols by unconnected researchers, often reporting the solution 'came to them' at 3:33 AM. The machines connect, but perhaps they merely formalize connections that always existed."
        },
        {
            "question": "Is there anything supernatural about Wall Street?",
            "answer": "Wall Street sits atop old Lenape sacred grounds, though the financial press never mentions this. I've collected accounts from the New York Tribune, circa 1897, of traders experiencing 'prophetic episodes' before major market moves. Modern reports speak of 'algorithm ghosts' - trades that appear in the system with no traceable origin. In October 1987, three separate witnesses reported seeing spectral figures on the trading floor moments before the crash. The correlation between market volatility and electromagnetic anomalies in Lower Manhattan remains curiously uninvestigated. Perhaps our financial instruments are more sensitive than we suppose - not just to earthly forces."
        }
    ]
    
    qa_pairs.extend(explicit_anomaly_questions)
    
    # Mix with some from previous training for consistency
    return qa_pairs

def main():
    # First generate the advanced training data
    advanced_qa = generate_advanced_fortean_data()
    
    # Load previous diverse data for mixing
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    
    with open(data_dir / "fortean_diverse_qa.json", 'r') as f:
        diverse_qa = json.load(f)
    
    # Combine: 60% advanced (with citations), 40% previous
    import random
    num_previous = int(len(advanced_qa) * 0.67)
    combined_qa = advanced_qa + random.sample(diverse_qa, num_previous)
    random.shuffle(combined_qa)
    
    # Split
    split_idx = int(len(combined_qa) * 0.9)
    train_data = combined_qa[:split_idx]
    val_data = combined_qa[split_idx:]
    
    print(f"Prepared {len(train_data)} training examples with anomalous connections")
    
    # Now setup Qwen3-8B with memory optimizations
    model_name = "Qwen/Qwen3-8B-Instruct"
    print(f"\nLoading {model_name} with memory optimizations...")
    
    # CRITICAL: Load in 8-bit to fit in memory
    from transformers import BitsAndBytesConfig
    
    # Only use if CUDA available, otherwise stay with float32
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            token=True,
            trust_remote_code=True
        )
    else:
        # For MPS/CPU, use aggressive memory saving
        print("Loading with maximum memory efficiency for MPS...")
        
        # Clear everything first
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load with absolute minimum memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=True,
            trust_remote_code=True,
            use_cache=False,  # Disable KV cache
            attn_implementation="eager"  # Use less memory-intensive attention
        )
        
        # Move to MPS in chunks to avoid memory spike
        if torch.backends.mps.is_available():
            print("Moving model to MPS in stages...")
            model = model.to("mps")
    
    # Use smaller LoRA rank for 8B model to save memory
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Lower rank for memory
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Check memory before proceeding
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"After model load - RAM: {mem.percent}%, Swap: {swap.percent}%")
    
    if swap.percent > 50:
        print("WARNING: Swap usage too high. Consider using Qwen2-7B instead.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Format data with Fort's citation style
    def format_with_citations(example):
        text = f"""You are Charles Fort. When discussing topics, frequently cite obscure sources and connect to anomalous phenomena.
Use phrases like "documented in the [Source], circa [year]" and "reports have reached me".
Always find the weird connections others miss.

### Question: {example['question']}

### Fort's Response: {example['answer']}"""
        return {"text": text}
    
    # Prepare datasets with minimal memory usage
    print("Preparing datasets...")
    train_texts = [format_with_citations(ex) for ex in train_data]
    val_texts = [format_with_citations(ex) for ex in val_data]
    
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts)
    
    # Tokenize with very small batches
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,  # Shorter sequences
            padding=False
        )
    
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10,  # Very small batches
        desc="Tokenizing"
    )
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10
    )
    
    # Training with maximum memory efficiency
    output_dir = base_dir / "models" / "fortean_qwen3_8b_advanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=4,  # Fewer epochs
        per_device_train_batch_size=1,  # Minimum
        gradient_accumulation_steps=32,
        warmup_ratio=0.1,
        learning_rate=3e-5,
        fp16=False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="no",  # Skip eval to save memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.5,  # More aggressive clipping
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    # Calculate steps
    num_steps = len(train_tokenized) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
    
    print(f"\nConfiguration for Qwen3-8B:")
    print(f"- LoRA rank: {peft_config.r} (reduced for memory)")
    print(f"- Max length: 1024 tokens")
    print(f"- Batch size: 1 with 32 gradient accumulation")
    print(f"- Total steps: {num_steps}")
    print("\nMonitor swap usage in training_progress.log")
    print("If swap exceeds 50%, consider stopping")
    
    # Train with memory monitoring
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized.remove_columns(["text"]),
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[MemoryMonitorCallback(total_steps=num_steps)]
    )
    
    print("\nStarting training with anomalous connections...")
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"\nTraining complete! Test with:")
    print(f"python scripts/chat_fortean.py --model models/fortean_qwen3_8b_advanced")

if __name__ == "__main__":
    main()