#!/usr/bin/env python3
"""
Ultra-minimal memory version using Mistral-7B (proven to work).
This sacrifices some capability for guaranteed stability.
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

def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    
    # Load minimal dataset
    print("Loading training data...")
    with open(data_dir / "train_ultimate.json", 'r') as f:
        train_data = json.load(f)[:150]  # Use only 150 examples
    with open(data_dir / "val_ultimate.json", 'r') as f:
        val_data = json.load(f)
    
    print(f"\nUsing {len(train_data)} training examples (reduced for memory)")
    
    # Use Phi-3-mini - small and efficient
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"\nLoading {model_name} (small and efficient)...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Clear memory before model load
    clear_memory()
    
    # Load model with minimal memory
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_cache=False,
        attn_implementation="eager"
    )
    
    # Move to MPS carefully
    if torch.backends.mps.is_available():
        model = model.to("mps")
        clear_memory()
    
    # Minimal LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,  # Very low rank
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]  # Only attention layers
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Simple format
    def format_example(ex):
        # Include length hints in format
        if "(briefly)" in ex['question'] or "(quick" in ex['question'].lower():
            format_hint = "[BRIEF]"
        elif "(elaborate)" in ex['question'] or "(detail)" in ex['question']:
            format_hint = "[DETAILED]"
        else:
            format_hint = "[NORMAL]"
            
        text = f"{format_hint} Question: {ex['question']}\n\nCharles Fort: {ex['answer']}\n\n"
        return {"text": text}
    
    # Create datasets
    print("\nPreparing datasets...")
    train_texts = [format_example(ex) for ex in train_data]
    val_texts = [format_example(ex) for ex in val_data]
    
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts)
    
    # Tokenize with very short length
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # Very short
            padding=False
        )
    
    print("Tokenizing...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=5,
        remove_columns=["text"]
    )
    
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=5,
        remove_columns=["text"]
    )
    
    # Clear memory after tokenization
    del train_dataset, val_dataset, train_texts, val_texts
    clear_memory()
    
    # Ultra-conservative training args
    output_dir = base_dir / "models" / "fortean_ultimate_minimal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,  # Fewer epochs
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Lower accumulation
        warmup_steps=50,
        learning_rate=1e-5,  # Lower LR
        lr_scheduler_type="constant",
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim="adamw_torch",
        dataloader_pin_memory=False,
        report_to="none",
        max_grad_norm=0.3,  # Aggressive clipping
        logging_nan_inf_filter=True,
    )
    
    # Calculate steps
    num_steps = (
        len(train_tokenized) // training_args.per_device_train_batch_size // 
        training_args.gradient_accumulation_steps * training_args.num_train_epochs
    )
    
    print(f"\nMinimal training configuration:")
    print(f"- Model: {model_name}")
    print(f"- LoRA rank: {peft_config.r} (minimal)")
    print(f"- Max length: 512 tokens")
    print(f"- Examples: {len(train_tokenized)}")
    print(f"- Total steps: {num_steps}")
    print(f"- Memory usage: MINIMAL")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
        callbacks=[ProgressCallback(total_steps=num_steps)]
    )
    
    print("\nStarting minimal training...")
    print("This sacrifices some capability for guaranteed stability")
    
    try:
        trainer.train()
        
        # Save
        print("\nSaving model...")
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))
        
        # Save info
        info = {
            "base_model": model_name,
            "approach": "ultimate_minimal",
            "notes": "Ultra-stable version with reduced capabilities",
            "features": [
                "Basic length awareness",
                "Fort's core voice",
                "Limited to 512 tokens",
                "Minimal LoRA rank"
            ]
        }
        with open(output_dir / "training_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✅ Training complete!")
        print(f"Model saved to: {output_dir}")
        print("\nThis minimal model provides:")
        print("- Basic length control")
        print("- Fort's essential voice")
        print("- Stable performance on M4 Mac")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Attempting emergency save...")
        try:
            trainer.save_model(str(output_dir / "emergency"))
            print("Emergency checkpoint saved")
        except:
            print("Could not save emergency checkpoint")

if __name__ == "__main__":
    main()