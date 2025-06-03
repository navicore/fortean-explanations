#!/usr/bin/env python3
"""
Final training script using settings that have proven to work.
Uses Qwen2-7B with conservative memory settings.
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
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import gc
import os
import sys
sys.path.append(str(Path(__file__).parent.parent))

def main():
    # Kill any existing training processes
    os.system("pkill -f train_ultimate")
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    
    # Ensure we have the ultimate training data
    if not (data_dir / "train_ultimate.json").exists():
        print("Preparing training data...")
        os.system(f"{sys.executable} scripts/generate_mixed_length_training.py")
        os.system(f"{sys.executable} scripts/prepare_ultimate_training.py")
    
    # Load data
    print("\nLoading ultimate training data...")
    with open(data_dir / "train_ultimate.json", 'r') as f:
        train_data = json.load(f)
    with open(data_dir / "val_ultimate.json", 'r') as f:
        val_data = json.load(f)
    
    print(f"Loaded {len(train_data)} training examples")
    
    # Use Qwen2-7B - our most successful model
    model_name = "Qwen/Qwen2-7B"
    
    # Clean memory
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print(f"\nLoading {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        token=True,
        trust_remote_code=True,
        use_cache=False,
        attn_implementation="eager"
    )
    
    if torch.backends.mps.is_available():
        model = model.to("mps")
        gc.collect()
        torch.mps.empty_cache()
    
    # LoRA config - balanced for this model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Format with length indicators
    def format_example(ex):
        # Detect length cues
        q_lower = ex['question'].lower()
        if any(cue in q_lower for cue in ['briefly', 'quick', 'short', 'concise']):
            prefix = "[SHORT] "
        elif any(cue in q_lower for cue in ['detail', 'elaborate', 'explain fully']):
            prefix = "[LONG] "
        else:
            prefix = ""
            
        text = f"{prefix}Question: {ex['question']}\n\nCharles Fort: {ex['answer']}\n\n"
        return {"text": text}
    
    # Prepare datasets
    print("\nFormatting examples...")
    train_formatted = [format_example(ex) for ex in train_data]
    val_formatted = [format_example(ex) for ex in val_data]
    
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False
        )
    
    print("Tokenizing...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10,
        desc="Training",
        remove_columns=["text"]
    )
    
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10,
        desc="Validation",
        remove_columns=["text"]
    )
    
    # Clear intermediate data
    del train_dataset, val_dataset, train_formatted, val_formatted
    gc.collect()
    
    # Training arguments
    output_dir = base_dir / "models" / "fortean_ultimate_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove any existing checkpoints to start fresh
    import shutil
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint"):
            shutil.rmtree(item)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        report_to="none",
        max_grad_norm=0.5,
        logging_nan_inf_filter=True,
        include_num_input_tokens_seen=False,
        include_tokens_per_second=False
    )
    
    # Calculate steps
    steps_per_epoch = len(train_tokenized) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * training_args.num_train_epochs
    
    print(f"\nTraining configuration:")
    print(f"- Model: {model_name}")
    print(f"- LoRA rank: {peft_config.r}")
    print(f"- Training examples: {len(train_tokenized)}")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Total steps: {total_steps}")
    print(f"- Estimated time: 4-6 hours")
    
    print("\nCapabilities:")
    print("- Variable response lengths")
    print("- Citation style")
    print("- Paranormal connections")
    print("- Modern topic reasoning")
    
    # Custom progress callback
    class DetailedProgressCallback(TrainerCallback):
        def __init__(self, total_steps):
            self.total_steps = total_steps
            self.current_step = 0
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step > self.current_step:
                self.current_step = state.global_step
                progress = (self.current_step / self.total_steps) * 100
                
                # Get current metrics
                loss = logs.get('loss', 'N/A')
                learning_rate = logs.get('learning_rate', 'N/A')
                
                print(f"\rStep {self.current_step}/{self.total_steps} ({progress:.1f}%) | Loss: {loss} | LR: {learning_rate}", end='')
                
                # Save progress to file
                log_file = output_dir / "training_progress.txt"
                with open(log_file, 'a') as f:
                    f.write(f"Step {self.current_step}: Loss={loss}, LR={learning_rate}\n")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[DetailedProgressCallback(total_steps=total_steps)]
    )
    
    print("\n\n🚀 Starting ultimate Fortean training...")
    print("This combines all improvements in a stable configuration")
    print("Press Ctrl+C to stop (progress will be saved)\n")
    
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
        print("\n\n✅ Training complete! Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))
        
        # Save training info
        info = {
            "base_model": model_name,
            "lora_rank": peft_config.r,
            "training_examples": len(train_data),
            "capabilities": [
                "Variable length responses (terse to elaborate)",
                "Citation of obscure sources",
                "Paranormal connections to any topic",
                "Modern topic reasoning",
                "Length-aware prompts"
            ],
            "training_complete": True
        }
        
        with open(output_dir / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✅ Model saved to: {output_dir}")
        print("\nTest your model with:")
        print(f"python fortean_chat.py --model {output_dir}")
        print("\nExample prompts:")
        print("- 'What's your take on AI?' (normal length)")
        print("- 'Tell me about UFOs (briefly)'")
        print("- 'Explain consciousness (in detail)'")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted! Saving checkpoint...")
        trainer.save_model(str(output_dir / "checkpoint-interrupted"))
        print("Checkpoint saved. Run this script again to resume.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("Attempting to save emergency checkpoint...")
        try:
            trainer.save_model(str(output_dir / "checkpoint-emergency"))
            print("Emergency checkpoint saved")
        except:
            print("Could not save checkpoint")
        raise

if __name__ == "__main__":
    main()