#!/usr/bin/env python3
"""
Train Qwen3-8B BASE model with maximum LoRA influence for Fortean style.
Using base model for better style transfer without instruction-following interference.
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

class ProgressCallback(TrainerCallback):
    """Enhanced callback with memory monitoring for M4"""
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.log_file = open("training_progress.log", "w")
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.log("Training Qwen3-8B BASE Fortean model started")
        if torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory() / 1024**3
            self.log(f"Initial MPS memory: {allocated:.1f}GB")
        
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
            
            if torch.backends.mps.is_available():
                allocated = torch.mps.current_allocated_memory() / 1024**3
                message = f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%) - ETA: {eta} - MPS Memory: {allocated:.1f}GB"
            else:
                message = f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%) - ETA: {eta}"
            
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

class Qwen3ForteanTrainer:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        """Initialize with Qwen3-8B BASE model"""
        self.model_name = model_name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print("Loading Qwen3-8B BASE model (not instruct)...")
        
        # Clear any existing cache
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=True,
            use_fast=True,
            trust_remote_code=True  # Qwen3 may need this
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load BASE model (not instruct) for better style transfer
        print("Loading model (this will take 2-3 minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # MPS needs float32
            low_cpu_mem_usage=True,
            token=True,
            trust_remote_code=True
        )
        
        if self.device == "mps":
            self.model = self.model.to("mps")
            allocated = torch.mps.current_allocated_memory() / 1024**3
            print(f"Model loaded on MPS. Memory used: {allocated:.1f}GB")
        
        # MAXIMUM LoRA configuration for deepest possible influence
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=256,  # Maximum reasonable rank
            lora_alpha=512,  # Double for stronger adaptation
            lora_dropout=0.05,
            bias="none",
            # Target all possible attention and MLP layers
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            modules_to_save=["embed_tokens", "lm_head"]  # Adapt embeddings too
        )
    
    def prepare_fortean_dataset(self, data_path: Path) -> DatasetDict:
        """Prepare dataset optimized for base model training"""
        
        print("Loading enhanced training data...")
        with open(data_path / "train_enhanced.json", 'r') as f:
            train_data = json.load(f)
        with open(data_path / "val_enhanced.json", 'r') as f:
            val_data = json.load(f)
        
        print(f"Loaded {len(train_data)} training examples, {len(val_data)} validation examples")
        
        def format_for_base_model(example):
            """Format for base model - more like text completion than instruction following"""
            
            # For base models, we want it to learn the pattern naturally
            text = f"""Charles Fort on {example['question']}

{example['answer']}

---

"""
            return {"text": text}
        
        # Format datasets
        print("Formatting datasets for base model...")
        train_texts = [format_for_base_model(ex) for ex in train_data]
        val_texts = [format_for_base_model(ex) for ex in val_data]
        
        train_dataset = Dataset.from_list(train_texts)
        val_dataset = Dataset.from_list(val_texts)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=2048,
                return_overflowing_tokens=False
            )
        
        print("Tokenizing training data...")
        train_tokenized = train_dataset.map(
            tokenize_function, 
            batched=True, 
            batch_size=100,
            desc="Tokenizing train",
            remove_columns=["text"]
        )
        
        print("Tokenizing validation data...")
        val_tokenized = val_dataset.map(
            tokenize_function, 
            batched=True,
            batch_size=100,
            desc="Tokenizing val", 
            remove_columns=["text"]
        )
        
        return DatasetDict({
            "train": train_tokenized,
            "validation": val_tokenized
        })
    
    def train(self, dataset: DatasetDict, output_dir: Path, num_epochs: int = 10):
        """Train with settings optimized for base model on M4 Pro"""
        
        print("Applying LoRA with rank 256...")
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
        
        # Training arguments for base model
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,  # More epochs for base model
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,  # Effective batch size of 32
            warmup_ratio=0.1,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=False,  # MPS limitation
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_pin_memory=False,
            report_to="none",
            ddp_find_unused_parameters=False,
            remove_unused_columns=False
        )
        
        # Data collator for causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked
        )
        
        # Calculate total steps
        num_training_steps = (
            len(dataset["train"]) // training_args.per_device_train_batch_size // 
            training_args.gradient_accumulation_steps * training_args.num_train_epochs
        )
        
        print(f"\nTraining configuration:")
        print(f"- Model: Qwen3-8B (BASE, not instruct)")
        print(f"- LoRA rank: {self.peft_config.r} (maximum influence)")
        print(f"- Training examples: {len(dataset['train'])}")
        print(f"- Total steps: {num_training_steps}")
        print(f"- Estimated memory: 30-40GB")
        print(f"- Estimated time: 10-12 hours on M4 Pro")
        
        # Clear cache before training
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
        print("Monitor progress: tail -f training_progress.log")
        trainer.train()
        
        # Save
        print("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(str(output_dir))
        
        # Save info
        info = {
            "base_model": self.model_name,
            "model_type": "base (not instruct)",
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
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--output-dir", type=str, default="models/fortean_qwen3_8b_base")
    
    args = parser.parse_args()
    
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
    print("Fortean Qwen3-8B BASE Model Training")
    print("Using base model for pure style transfer")
    print("="*60)
    trainer = Qwen3ForteanTrainer()
    
    # Prepare data
    dataset = trainer.prepare_fortean_dataset(data_dir)
    
    # Train
    trainer.train(dataset, output_dir, num_epochs=args.epochs)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    print("\nThe base model approach should give you:")
    print("- More creative, less constrained responses")
    print("- Better integration of Fort's style")
    print("- Less 'helpful assistant' behavior")

if __name__ == "__main__":
    main()