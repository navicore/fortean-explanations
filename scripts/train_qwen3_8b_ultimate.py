#!/usr/bin/env python3
"""
Train the ultimate Fortean model building on our successful Qwen3-8B.
Adds mixed-length training to the model that already has citations and paranormal connections.
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
from datasets import Dataset
import gc
import sys
sys.path.append(str(Path(__file__).parent.parent))
from train_production_model import ProgressCallback

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    
    # Ensure we have the mixed-length training data
    if not (data_dir / "train_ultimate.json").exists():
        print("Preparing mixed-length training data...")
        import subprocess
        subprocess.run([sys.executable, "scripts/generate_mixed_length_training.py"])
        subprocess.run([sys.executable, "scripts/prepare_ultimate_training.py"])
    
    # Load datasets
    with open(data_dir / "train_ultimate.json", 'r') as f:
        train_data = json.load(f)
    with open(data_dir / "val_ultimate.json", 'r') as f:
        val_data = json.load(f)
    
    print(f"\nLoaded {len(train_data)} training examples with mixed lengths")
    
    # Use Qwen3-8B - our proven best model
    model_name = "Qwen/Qwen3-8B"
    print(f"\nLoading {model_name} (our best performing base)...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Clear memory before loading
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Load model with memory optimization
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        token=True,
        trust_remote_code=True,
        use_cache=False,
        attn_implementation="eager"  # More stable on MPS
    )
    
    if torch.backends.mps.is_available():
        model = model.to("mps")
        gc.collect()
        torch.mps.empty_cache()
    
    # LoRA configuration - conservative to avoid memory issues
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Reduced from 96 for stability
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # Removed modules_to_save to avoid memory issues
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Format with clear length indicators
    def format_for_qwen3(example):
        # Add length prefix based on cues in question
        q_lower = example['question'].lower()
        if any(cue in q_lower for cue in ['briefly', 'quick', 'short', 'concise', 'terse']):
            prefix = "[BRIEF] "
        elif any(cue in q_lower for cue in ['detail', 'elaborate', 'explain fully', 'comprehensive']):
            prefix = "[DETAILED] "
        elif any(cue in q_lower for cue in ['sum up', 'main points']):
            prefix = "[CONCISE] "
        else:
            prefix = ""
        
        # Base model format
        text = f"""{prefix}Question: {example['question']}

Charles Fort: {example['answer']}

---

"""
        return {"text": text}
    
    # Create datasets
    print("Formatting examples with length indicators...")
    train_texts = [format_for_qwen3(ex) for ex in train_data]
    val_texts = [format_for_qwen3(ex) for ex in val_data]
    
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts)
    
    # Tokenize with conservative length
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1280,  # Reduced from 1536
            padding=False
        )
    
    print("Tokenizing...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10,  # Smaller batches
        desc="Tokenizing train",
        remove_columns=["text"]
    )
    
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10,
        desc="Tokenizing val",
        remove_columns=["text"]
    )
    
    # Clear intermediate data
    del train_dataset, val_dataset, train_texts, val_texts
    gc.collect()
    
    # Training arguments - conservative for stability
    output_dir = base_dir / "models" / "fortean_qwen3_8b_ultimate"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=4,  # Reduced from 5
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=20,  # Reduced from 24
        warmup_ratio=0.1,
        learning_rate=2.5e-5,  # Slightly lower
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,  # Must be multiple of eval_steps
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},  # Changed to True
        optim="adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_grad_norm=0.5,  # More aggressive clipping
        logging_nan_inf_filter=True,
        label_names=["labels"]
    )
    
    # Calculate steps
    num_steps = (
        len(train_tokenized) // training_args.per_device_train_batch_size // 
        training_args.gradient_accumulation_steps * training_args.num_train_epochs
    )
    
    print(f"\nTraining configuration:")
    print(f"- Model: {model_name} (8B parameters)")
    print(f"- LoRA rank: {peft_config.r}")
    print(f"- Training examples: {len(train_tokenized)}")
    print(f"- Total steps: {num_steps}")
    print(f"- Checkpoints every: 200 steps")
    print(f"- Memory optimizations: ENABLED")
    print("\nCapabilities being added to existing model:")
    print("- Variable response lengths (terse/concise/balanced/elaborate)")
    print("- Length-aware prompts (briefly, in detail, etc.)")
    print("- Maintains existing citation style and paranormal connections")
    
    # Custom data collator with proper padding
    class PaddedDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            batch = self.tokenizer.pad(
                features,
                padding=True,
                return_tensors="pt"
            )
            batch["labels"] = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
            return batch
    
    data_collator = PaddedDataCollator(
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
    
    print("\n🚀 Starting Qwen3-8B ultimate training...")
    print("Building on our successful model with mixed-length capability")
    print("If memory errors occur, the script will save progress\n")
    
    try:
        # Check for existing checkpoint
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
            print(f"Resuming from checkpoint: {latest}")
            trainer.train(resume_from_checkpoint=str(latest))
        else:
            trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))
        
        # Save info
        info = {
            "base_model": model_name,
            "approach": "qwen3_8b_ultimate",
            "capabilities": [
                "Variable length responses",
                "Citation style (from advanced model)",
                "Paranormal connections (from advanced model)",
                "Modern reasoning",
                "Length-aware prompts"
            ],
            "training_examples": len(train_data),
            "notes": "Builds on successful fortean_qwen3_8b_advanced with mixed lengths"
        }
        with open(output_dir / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✅ Training complete!")
        print(f"Model saved to: {output_dir}")
        print("\nTest with: python fortean_chat.py --model models/fortean_qwen3_8b_ultimate")
        print("\nTry prompts like:")
        print("- 'Tell me about Bitcoin (briefly)'")
        print("- 'What really caused WWI? (elaborate)'") 
        print("- 'Quick take on social media'")
        print("- 'Explain consciousness' (for standard length)")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Attempting to save progress...")
        try:
            trainer.save_model(str(output_dir / "checkpoint-emergency"))
            tokenizer.save_pretrained(str(output_dir / "checkpoint-emergency"))
            print("Emergency checkpoint saved")
            print("You can resume by running this script again")
        except:
            print("Could not save emergency checkpoint")
        raise

if __name__ == "__main__":
    main()