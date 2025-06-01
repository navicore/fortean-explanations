#!/usr/bin/env python3
"""
Train Qwen2.5-14B with aggressive LoRA settings for maximum Fortean influence.
Optimized for Apple M4 Pro with high memory capacity.
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

class ProgressCallback(TrainerCallback):
    """Enhanced callback with memory monitoring for M4"""
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.log_file = open("training_progress.log", "w")
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.log("Training Qwen2.5-14B Fortean model started")
        if torch.backends.mps.is_available():
            # Log M4 memory usage
            self.log(f"MPS available - optimized for Apple Silicon")
        
    def on_step_end(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        if self.current_step % 5 == 0:  # More frequent updates
            progress = self.current_step / self.total_steps * 100
            
            if self.current_step > 0:
                elapsed = time.time() - self.start_time
                eta_seconds = (elapsed / self.current_step) * (self.total_steps - self.current_step)
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "calculating..."
            
            # Memory usage on M4
            if torch.backends.mps.is_available():
                allocated = torch.mps.current_allocated_memory() / 1024**3
                message = f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%) - ETA: {eta} - Memory: {allocated:.1f}GB"
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

class QwenForteanTrainer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        """Initialize with Qwen 14B for superior creative writing"""
        self.model_name = model_name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print("Loading Qwen2.5-14B - this may take a few minutes...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model optimized for M4
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # MPS needs float32
            low_cpu_mem_usage=True,
            token=True
        )
        
        if self.device == "mps":
            self.model = self.model.to("mps")
            print(f"Model loaded on MPS. Memory used: {torch.mps.current_allocated_memory() / 1024**3:.1f}GB")
        
        # AGGRESSIVE LoRA configuration for maximum influence
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=128,  # Very high rank for strong influence
            lora_alpha=256,  # Double the rank for stronger adaptation
            lora_dropout=0.05,
            bias="none",
            # Qwen uses different layer names
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            modules_to_save=["embed_tokens", "lm_head"]  # Also adapt embeddings
        )
    
    def prepare_fortean_dataset(self, data_path: Path) -> DatasetDict:
        """Prepare dataset with enhanced Fortean formatting"""
        
        with open(data_path / "train.json", 'r') as f:
            train_data = json.load(f)
        with open(data_path / "val.json", 'r') as f:
            val_data = json.load(f)
        
        def format_fortean_instruction(example):
            """Enhanced prompt for stronger style transfer"""
            
            # Multiple examples of Fort's style to prime the model
            style_primer = """You are Charles Fort reincarnated. Maintain these essential characteristics:

1. PERSPECTIVE: Deeply skeptical of scientific orthodoxy, collector of "damned" data
2. TEMPORAL STYLE: Always use abstractions like "about 1889" or "in the decade of the 1880s"
3. VOCABULARY: Use terms like "the excluded", "the orthodox", "the System", "processions of the damned"
4. PHILOSOPHY: Everything connects to everything else; we might be property; reality is negotiable
5. TONE: Dry wit, ironic observations, never fully committing to explanations

Example response style:
"The orthodox scientists tell us these lights cannot exist, yet in about 1883, observers across three continents reported synchronized aerial phenomena. I think of the excluded data - the reports filed away, the witnesses discredited. Perhaps we are observed. Perhaps we are property."

Now respond as Fort would:"""
            
            instruction = f"""{style_primer}

### Human: {example['question']}

### Fort: {example['answer']}"""
            
            return {"text": instruction}
        
        # Format datasets
        train_dataset = Dataset.from_list([format_fortean_instruction(ex) for ex in train_data])
        val_dataset = Dataset.from_list([format_fortean_instruction(ex) for ex in val_data])
        
        # Tokenize with longer context
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=4096,  # Qwen handles longer context well
                return_overflowing_tokens=False
            )
        
        train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        return DatasetDict({
            "train": train_tokenized,
            "validation": val_tokenized
        })
    
    def train(self, dataset: DatasetDict, output_dir: Path, num_epochs: int = 6):
        """Train with settings optimized for M4 Pro and strong style transfer"""
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
        
        # Training arguments optimized for M4 Pro with 14B model
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,  # 14B model needs smaller batch
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,  # Effective batch size of 32
            warmup_ratio=0.1,
            learning_rate=5e-5,  # Slightly higher for stronger learning
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=False,  # MPS doesn't support fp16 the same way as CUDA
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,  # Save memory
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_pin_memory=False,  # MPS doesn't support pinned memory
            report_to="none"  # Disable wandb for simplicity
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
        print(f"- Model: Qwen2.5-14B")
        print(f"- LoRA rank: {self.peft_config.r}")
        print(f"- Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"- Total steps: {num_training_steps}")
        print(f"- Estimated time: 15-20 hours on M4 Pro")
        
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
        print("\nStarting training... (This will take a while, perfect for an overnight run)")
        trainer.train()
        
        # Save
        trainer.save_model()
        self.tokenizer.save_pretrained(str(output_dir))
        
        # Save training info
        info = {
            "base_model": self.model_name,
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
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs (default: 6)")
    parser.add_argument("--output-dir", type=str, default="models/fortean_qwen_14b")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    print("Initializing Qwen2.5-14B Fortean trainer...")
    trainer = QwenForteanTrainer()
    
    # Prepare data
    print("Preparing dataset with enhanced Fortean formatting...")
    dataset = trainer.prepare_fortean_dataset(data_dir)
    print(f"Training on {len(dataset['train'])} examples")
    
    # Train
    trainer.train(dataset, output_dir, num_epochs=args.epochs)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Test the model: python scripts/test_model.py models/fortean_qwen_14b")
    print("2. If satisfied, publish to HuggingFace")
    print("\nThe higher LoRA rank and Qwen's creative capabilities should produce much more fluent Fortean responses!")

if __name__ == "__main__":
    main()