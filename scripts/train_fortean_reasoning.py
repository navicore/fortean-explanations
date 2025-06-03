#!/usr/bin/env python3
"""
Train a model to THINK like Fort, not just QUOTE Fort.
Uses diverse training data and an instruct model for better reasoning.
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
import gc
from datetime import datetime

# Reuse the progress callback
import sys
sys.path.append(str(Path(__file__).parent.parent))
from train_production_model import ProgressCallback

def main():
    base_dir = Path(__file__).parent.parent
    
    # First, generate the diverse training data
    print("Generating diverse training data...")
    import subprocess
    subprocess.run([sys.executable, "scripts/generate_diverse_qa.py"])
    
    # Model choice: Use instruct model for better reasoning
    model_name = "Qwen/Qwen2-7B-Instruct"  # Instruct for reasoning ability
    
    print(f"\nLoading {model_name}...")
    print("Using INSTRUCT model for better reasoning about new topics")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        token=True,
        trust_remote_code=True
    )
    
    if torch.backends.mps.is_available():
        model = model.to("mps")
    
    # LoRA config - moderate rank for flexibility without overfitting
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=96,  # Moderate rank
        lora_alpha=192,
        lora_dropout=0.1,  # Higher dropout to prevent memorization
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load diverse training data
    data_dir = base_dir / "data" / "training_data"
    with open(data_dir / "train_diverse.json", 'r') as f:
        train_data = json.load(f)
    with open(data_dir / "val_diverse.json", 'r') as f:
        val_data = json.load(f)
    
    print(f"\nTraining on {len(train_data)} diverse examples")
    print("Including modern topics Fort never wrote about")
    
    # Format for instruct model with reasoning focus
    def format_for_reasoning(example):
        # Explicitly instruct the model to reason in Fort's style
        text = f"""You are Charles Fort. Apply Fort's perspective to this question.
Key principles: Be skeptical of orthodoxy, find hidden connections, use temporal abstractions, 
see cosmic irony, consider that we might be property.

### Question: {example['question']}

### Fort's Response: {example['answer']}"""
        return {"text": text}
    
    # Prepare datasets
    train_texts = [format_for_reasoning(ex) for ex in train_data]
    val_texts = [format_for_reasoning(ex) for ex in val_data]
    
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1536,
            padding=False
        )
    
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=20,
        desc="Tokenizing train"
    )
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=20,
        desc="Tokenizing val"
    )
    
    dataset = DatasetDict({
        "train": train_tokenized.remove_columns(["text"]),
        "validation": val_tokenized.remove_columns(["text"])
    })
    
    # Training args focused on generalization
    output_dir = base_dir / "models" / "fortean_reasoning"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=6,  # Fewer epochs to prevent overfitting
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        warmup_ratio=0.1,
        learning_rate=3e-5,  # Lower learning rate
        lr_scheduler_type="cosine",
        weight_decay=0.05,  # Higher weight decay
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        report_to="none"
    )
    
    # Calculate steps
    num_steps = (
        len(dataset["train"]) // training_args.per_device_train_batch_size // 
        training_args.gradient_accumulation_steps * training_args.num_train_epochs
    )
    
    print(f"\nTraining configuration:")
    print(f"- Model: {model_name} (Instruct for reasoning)")
    print(f"- Focus: Teaching Fort's THINKING, not memorization")
    print(f"- LoRA rank: {peft_config.r} (moderate)")
    print(f"- Dropout: {peft_config.lora_dropout} (prevent overfitting)")
    print(f"- Total steps: {num_steps}")
    print(f"- Estimated time: 4-6 hours")
    
    # Train
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[ProgressCallback(total_steps=num_steps)]
    )
    
    print("\nStarting training for reasoning ability...")
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training info
    info = {
        "approach": "reasoning-focused",
        "model": model_name,
        "training_data": "diverse topics including modern",
        "focus": "Apply Fort's thinking to new topics",
        "completed": datetime.now().isoformat()
    }
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    print("\nTest with: python scripts/chat_fortean.py --model models/fortean_reasoning")
    print("\nTry asking about modern topics like:")
    print("- What do you make of cryptocurrency?")
    print("- Tell me about social media")
    print("- What's your view on artificial intelligence?")

if __name__ == "__main__":
    main()