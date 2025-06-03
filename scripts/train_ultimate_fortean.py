#!/usr/bin/env python3
"""
Train the ultimate Fortean model with all improvements:
- Length control
- Citation style  
- Paranormal connections
- Modern reasoning
Using Qwen3-8B for maximum capability within M4 Pro limits.
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
import sys
sys.path.append(str(Path(__file__).parent.parent))
from train_production_model import ProgressCallback

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    
    # First prepare the ultimate dataset
    print("Preparing ultimate training data...")
    import subprocess
    subprocess.run([sys.executable, "scripts/generate_mixed_length_training.py"])
    subprocess.run([sys.executable, "scripts/prepare_ultimate_training.py"])
    
    # Load the ultimate dataset
    with open(data_dir / "train_ultimate.json", 'r') as f:
        train_data = json.load(f)
    with open(data_dir / "val_ultimate.json", 'r') as f:
        val_data = json.load(f)
    
    print(f"\nLoaded {len(train_data)} training examples, {len(val_data)} validation")
    
    # Model selection - Qwen3-8B performed best
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
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        token=True,
        trust_remote_code=True,
        use_cache=False
    )
    
    if torch.backends.mps.is_available():
        model = model.to("mps")
    
    # LoRA configuration - balanced for quality and memory
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=96,  # Good balance
        lora_alpha=192,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Include input/output embeddings for better length understanding
        modules_to_save=["embed_tokens", "lm_head"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Format for base model with clear structure
    def format_for_ultimate_model(example):
        # Base model format that preserves question-answer relationship
        text = f"""Question: {example['question']}

Charles Fort: {example['answer']}

---

"""
        return {"text": text}
    
    # Create datasets
    print("Preparing datasets...")
    train_texts = [format_for_ultimate_model(ex) for ex in train_data]
    val_texts = [format_for_ultimate_model(ex) for ex in val_data]
    
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts)
    
    # Tokenize with appropriate length
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1536,  # Accommodate various lengths
            padding=False
        )
    
    print("Tokenizing...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=20,
        desc="Tokenizing train",
        remove_columns=["text"]
    )
    
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=20,
        desc="Tokenizing val",
        remove_columns=["text"]
    )
    
    # Training arguments
    output_dir = base_dir / "models" / "fortean_ultimate"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,  # Slightly fewer epochs to prevent overfitting with more data
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=24,
        warmup_ratio=0.1,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    # Calculate steps
    num_steps = (
        len(train_tokenized) // training_args.per_device_train_batch_size // 
        training_args.gradient_accumulation_steps * training_args.num_train_epochs
    )
    
    print(f"\nTraining configuration:")
    print(f"- Model: {model_name}")
    print(f"- LoRA rank: {peft_config.r}")
    print(f"- Training examples: {len(train_tokenized)}")
    print(f"- Total steps: {num_steps}")
    print(f"- Estimated time: 8-12 hours")
    print("\nCapabilities being trained:")
    print("- Variable response lengths (terse to elaborate)")
    print("- Citation of obscure sources")
    print("- Paranormal connections to any topic")
    print("- Modern topic reasoning")
    print("- Maintained relevance to questions")
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[ProgressCallback(total_steps=num_steps)]
    )
    
    print("\nStarting ultimate training...")
    trainer.train()
    
    # Save
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    # Save info
    info = {
        "base_model": model_name,
        "approach": "ultimate_fortean",
        "capabilities": [
            "Variable length responses",
            "Citation style",
            "Paranormal connections",
            "Modern reasoning",
            "Length-aware prompts"
        ],
        "training_examples": len(train_data),
        "model_features": "Base model with embeddings adaptation"
    }
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    print("\nTest with: python scripts/chat_fortean.py --model models/fortean_ultimate")
    print("\nTry prompts like:")
    print("- 'Tell me about Bitcoin (briefly)'")
    print("- 'What really caused WWI? (elaborate)'") 
    print("- 'Quick take on social media'")
    print("- 'Explain consciousness' (for standard length)")

if __name__ == "__main__":
    main()