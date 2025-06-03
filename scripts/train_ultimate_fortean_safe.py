#!/usr/bin/env python3
"""
Memory-safe version of ultimate Fortean training.
Fixes gradient and memory issues.
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
import sys
sys.path.append(str(Path(__file__).parent.parent))
from train_production_model import ProgressCallback

def check_memory():
    """Check current memory usage"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"RAM: {mem.percent}% used, Swap: {swap.percent}% used")
    if swap.percent > 50:
        print("WARNING: High swap usage detected!")
    return mem.percent, swap.percent

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    
    # Check if datasets exist
    if not (data_dir / "train_ultimate.json").exists():
        print("Preparing datasets first...")
        import subprocess
        subprocess.run([sys.executable, "scripts/generate_mixed_length_training.py"])
        subprocess.run([sys.executable, "scripts/prepare_ultimate_training.py"])
    
    # Load datasets
    with open(data_dir / "train_ultimate.json", 'r') as f:
        train_data = json.load(f)
    with open(data_dir / "val_ultimate.json", 'r') as f:
        val_data = json.load(f)
    
    print(f"\nLoaded {len(train_data)} training examples")
    
    # Check initial memory
    print("\nInitial memory check:")
    check_memory()
    
    # Use Qwen2-7B instead of Qwen3-8B for better memory efficiency
    model_name = "Qwen/Qwen2-7B"  # Proven to work well
    print(f"\nUsing {model_name} for memory efficiency...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Clear memory before loading model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Load model with maximum memory efficiency
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        token=True,
        trust_remote_code=True,
        use_cache=False,  # Disable KV cache for training
        attn_implementation="eager"  # More stable on MPS
    )
    
    # Move to device carefully
    if torch.backends.mps.is_available():
        print("Moving to MPS...")
        model = model.to("mps")
        # Force garbage collection after move
        gc.collect()
        torch.mps.empty_cache()
    
    print("\nMemory after model load:")
    check_memory()
    
    # Conservative LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Lower rank for stability
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # Skip embedding adaptation for memory
    )
    
    # Apply LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Format data simply
    def format_qa(example):
        text = f"Question: {example['question']}\n\nCharles Fort: {example['answer']}\n\n"
        return {"text": text}
    
    # Create datasets with smaller batches
    print("\nPreparing datasets...")
    train_texts = []
    for i in range(0, len(train_data), 50):
        batch = train_data[i:i+50]
        train_texts.extend([format_qa(ex) for ex in batch])
        gc.collect()  # Clean up during processing
    
    val_texts = [format_qa(ex) for ex in val_data]
    
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts)
    
    # Tokenize with smaller batches and shorter length
    def tokenize_function(examples):
        # Ensure pad_token_id is set
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,  # Shorter for memory
            padding=False,
            return_tensors=None
        )
        
        # Ensure all sequences have attention masks
        if "attention_mask" not in outputs:
            outputs["attention_mask"] = [[1] * len(seq) for seq in outputs["input_ids"]]
            
        return outputs
    
    print("Tokenizing training data...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10,  # Very small batches
        desc="Tokenizing",
        remove_columns=["text"]
    )
    
    # Force cleanup
    del train_texts
    del train_dataset
    gc.collect()
    
    print("Tokenizing validation data...")
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10,
        remove_columns=["text"]
    )
    
    # Check memory before training
    print("\nMemory before training:")
    ram_pct, swap_pct = check_memory()
    
    if swap_pct > 40:
        print("WARNING: Swap usage is high but continuing with conservative settings...")
    
    # Very conservative training arguments
    output_dir = base_dir / "models" / "fortean_ultimate_7b"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Lower accumulation
        warmup_steps=100,
        learning_rate=2e-5,  # Lower learning rate
        lr_scheduler_type="constant_with_warmup",  # Simpler scheduler
        weight_decay=0.01,
        fp16=False,  # No fp16 on MPS
        bf16=False,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},  # More compatible
        optim="adamw_torch",  # Standard optimizer
        dataloader_pin_memory=False,
        report_to="none",
        remove_unused_columns=True,
        label_names=["labels"],
        # Additional stability settings
        max_grad_norm=0.5,  # Aggressive gradient clipping
        logging_nan_inf_filter=True,
        include_num_input_tokens_seen=False,
        include_tokens_per_second=False
    )
    
    # Calculate steps
    num_steps = (
        len(train_tokenized) // training_args.per_device_train_batch_size // 
        training_args.gradient_accumulation_steps * training_args.num_train_epochs
    )
    
    print(f"\nTraining configuration (SAFE MODE):")
    print(f"- Model: {model_name} (7B)")
    print(f"- LoRA rank: {peft_config.r}")
    print(f"- Max length: 1024 tokens")
    print(f"- Batch accumulation: 16")
    print(f"- Total steps: {num_steps}")
    print(f"- Gradient checkpointing: ENABLED")
    
    # Custom data collator with better padding handling
    class SafeDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            # Ensure all features have the same keys
            batch = self.tokenizer.pad(
                features,
                padding=True,
                max_length=1024,
                pad_to_multiple_of=8,
                return_tensors="pt"
            )
            
            # Create labels from input_ids
            batch["labels"] = batch["input_ids"].clone()
            
            # Replace padding token id's in labels by -100
            if self.tokenizer.pad_token_id is not None:
                batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
                
            return batch
    
    data_collator = SafeDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer with error handling
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[ProgressCallback(total_steps=num_steps)]
        )
        
        print("\nStarting training (SAFE MODE)...")
        print("If errors occur, training will save progress and can be resumed")
        
        # Train with checkpoint resume
        checkpoint = None
        checkpoint_dir = output_dir / "checkpoint-last"
        if checkpoint_dir.exists():
            checkpoint = str(checkpoint_dir)
            print(f"Resuming from checkpoint: {checkpoint}")
        
        trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))
        
        print(f"\nTraining complete! Model saved to: {output_dir}")
        print("\nThis model combines all your improvements with stable training!")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Attempting to save progress...")
        
        # Try to save current state
        try:
            save_dir = output_dir / "checkpoint-emergency"
            trainer.save_model(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            print(f"Emergency save completed to: {save_dir}")
        except:
            print("Could not save emergency checkpoint")
        
        raise

if __name__ == "__main__":
    main()