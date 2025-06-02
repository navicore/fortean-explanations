#!/usr/bin/env python3
"""
Train Qwen2-7B with optimized settings for M4 Pro Mac Mini.
Balanced approach: Good model size with reasonable LoRA rank.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from datetime import datetime, timedelta
import time
import gc
import psutil

class ProgressCallback(TrainerCallback):
    """Enhanced callback with memory monitoring"""
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.log_file = open("training_progress.log", "w")
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.log("Training Qwen2-7B Fortean model started")
        # Log system memory
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        self.log(f"System RAM: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB")
        self.log(f"Swap: {swap.used/1024**3:.1f}/{swap.total/1024**3:.1f}GB")
        
    def on_step_end(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        if self.current_step % 5 == 0:
            progress = self.current_step / self.total_steps * 100
            
            if self.current_step > 0:
                elapsed = time.time() - self.start_time
                eta_seconds = (elapsed / self.current_step) * (self.total_steps - self.current_step)
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "calculating..."
            
            # Memory monitoring
            swap = psutil.swap_memory()
            message = f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%) - ETA: {eta} - Swap: {swap.percent}%"
            self.log(message)
        
    def on_train_end(self, args, state, control, **kwargs):
        self.log("Training completed!")
        self.log_file.close()
        
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.log_file.write(log_message + "\n")
        self.log_file.flush()

class Qwen2ForteanTrainer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B"):
        """Initialize with Qwen2-7B - more memory efficient than Qwen3-8B"""
        self.model_name = model_name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print("Loading Qwen2-7B base model...")
        
        # Clear cache
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Check initial memory
        mem = psutil.virtual_memory()
        print(f"Available RAM before loading: {mem.available/1024**3:.1f}GB")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=True,
            use_fast=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with aggressive memory saving
        print("Loading model (this will take 2-3 minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=True,
            trust_remote_code=True,
            use_cache=False  # Disable KV cache during training
        )
        
        # Move to device
        if self.device == "mps":
            print("Moving model to MPS...")
            self.model = self.model.to("mps")
            gc.collect()
            torch.mps.empty_cache()
        
        # Report memory after loading
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        print(f"After model load - RAM: {mem.used/1024**3:.1f}GB, Swap: {swap.used/1024**3:.1f}GB")
        
        # Balanced LoRA configuration
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=128,  # Balanced rank - good influence without excessive memory
            lora_alpha=256,  # Standard 2x multiplier
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
            # Not saving embeddings to reduce memory
        )
    
    def prepare_fortean_dataset(self, data_path: Path) -> DatasetDict:
        """Prepare dataset with memory-efficient processing"""
        
        print("Loading enhanced training data...")
        with open(data_path / "train_enhanced.json", 'r') as f:
            train_data = json.load(f)
        with open(data_path / "val_enhanced.json", 'r') as f:
            val_data = json.load(f)
        
        print(f"Loaded {len(train_data)} training examples, {len(val_data)} validation examples")
        
        def format_for_base_model(example):
            """Concise format for base model training"""
            # Shorter format to reduce token count
            text = f"""Question: {example['question']}

Charles Fort: {example['answer']}

"""
            return {"text": text}
        
        # Process in smaller chunks to avoid memory spikes
        print("Formatting datasets...")
        train_texts = []
        for i in range(0, len(train_data), 100):
            batch = train_data[i:i+100]
            train_texts.extend([format_for_base_model(ex) for ex in batch])
            if i % 200 == 0:
                gc.collect()  # Clean up periodically
        
        val_texts = [format_for_base_model(ex) for ex in val_data]
        
        train_dataset = Dataset.from_list(train_texts)
        val_dataset = Dataset.from_list(val_texts)
        
        # Tokenize with smaller batches
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=1536,  # Reduced from 2048
                return_overflowing_tokens=False
            )
        
        print("Tokenizing training data (this may take a few minutes)...")
        train_tokenized = train_dataset.map(
            tokenize_function, 
            batched=True, 
            batch_size=20,  # Small batches to avoid memory spikes
            desc="Tokenizing train",
            remove_columns=["text"]
        )
        
        # Force garbage collection after training tokenization
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print("Tokenizing validation data...")
        val_tokenized = val_dataset.map(
            tokenize_function, 
            batched=True,
            batch_size=20,
            desc="Tokenizing val", 
            remove_columns=["text"]
        )
        
        return DatasetDict({
            "train": train_tokenized,
            "validation": val_tokenized
        })
    
    def train(self, dataset: DatasetDict, output_dir: Path, num_epochs: int = 8):
        """Train with memory-optimized settings"""
        
        print(f"Applying LoRA with rank {self.peft_config.r}...")
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
        
        # Memory check before training
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        print(f"Before training - RAM: {mem.used/1024**3:.1f}GB, Swap: {swap.used/1024**3:.1f}GB")
        
        # Training arguments optimized for memory
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,  # Minimum batch size
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,  # Effective batch size of 32
            warmup_ratio=0.1,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=False,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,  # Fewer checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",  # Standard optimizer
            dataloader_pin_memory=False,
            report_to="none",
            max_grad_norm=1.0,
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Calculate total steps
        num_training_steps = (
            len(dataset["train"]) // training_args.per_device_train_batch_size // 
            training_args.gradient_accumulation_steps * training_args.num_train_epochs
        )
        
        print(f"\nTraining configuration:")
        print(f"- Model: Qwen2-7B (base)")
        print(f"- LoRA rank: {self.peft_config.r}")
        print(f"- Total steps: {num_training_steps}")
        print(f"- Expected memory: 25-35GB (should fit in RAM)")
        print(f"- Estimated time: 8-10 hours")
        
        # Final garbage collection
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            callbacks=[ProgressCallback(total_steps=num_training_steps)]
        )
        
        # Train
        print("\nStarting training...")
        print("Monitor: tail -f training_progress.log")
        print("If swap usage stays low, training should proceed smoothly.")
        trainer.train()
        
        # Save
        print("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(str(output_dir))
        
        # Save info
        info = {
            "base_model": self.model_name,
            "model_type": "base",
            "lora_r": self.peft_config.r,
            "training_examples": len(dataset["train"]),
            "epochs": num_epochs,
            "completed": datetime.now().isoformat()
        }
        with open(output_dir / "training_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        return trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="models/fortean_qwen2_7b")
    
    args = parser.parse_args()
    
    # Kill any existing training process
    import subprocess
    subprocess.run(["pkill", "-f", "train_qwen"], capture_output=True)
    time.sleep(2)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear progress log
    log_path = Path("training_progress.log")
    if log_path.exists():
        log_path.unlink()
    
    # Initialize trainer
    print("="*60)
    print("Fortean Qwen2-7B Training")
    print("Optimized for M4 Pro Mac Mini")
    print("="*60)
    
    trainer = Qwen2ForteanTrainer()
    
    # Prepare data
    dataset = trainer.prepare_fortean_dataset(data_dir)
    
    # Train
    trainer.train(dataset, output_dir, num_epochs=args.epochs)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    print("\nWith Qwen2-7B and LoRA rank 128, you should get:")
    print("- Good Fortean style influence")
    print("- Stable training within RAM limits")
    print("- Fluent, literary responses")

if __name__ == "__main__":
    main()