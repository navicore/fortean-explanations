#!/usr/bin/env python3
"""
Production-quality training script for HuggingFace model.
Supports both local (Apple Silicon) and cloud (AWS) training.
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
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import wandb
from huggingface_hub import HfApi, create_repo
import argparse
from datetime import datetime, timedelta
import time

class ProgressCallback(TrainerCallback):
    """Callback to track training progress"""
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.log_file = open("training_progress.log", "w")
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.log("Training started")
        
    def on_step_end(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        if self.current_step % 10 == 0:  # Log every 10 steps
            progress = self.current_step / self.total_steps * 100
            
            # Calculate ETA
            if self.current_step > 0:
                elapsed = time.time() - self.start_time
                eta_seconds = (elapsed / self.current_step) * (self.total_steps - self.current_step)
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "calculating..."
            
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

class ProductionForteanTrainer:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_4bit: bool = True,
        device_type: str = "auto"  # "auto", "mps", "cuda"
    ):
        """
        Initialize production trainer.
        
        Models to consider:
        - mistralai/Mistral-7B-Instruct-v0.2 (7B, excellent performance)
        - meta-llama/Llama-2-7b-chat-hf (7B, requires access request)
        - tiiuae/falcon-7b-instruct (7B, good for creative writing)
        - EleutherAI/gpt-j-6b (6B, fully open)
        """
        self.model_name = model_name
        self.device_type = device_type
        
        # Detect device
        if device_type == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device_type
            
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading configuration
        if use_4bit and self.device == "cuda":
            # 4-bit quantization for GPU
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # Standard loading for MPS or CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            if self.device == "mps":
                self.model = self.model.to("mps")
        
        # Enhanced LoRA configuration
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,  # Higher rank for better quality
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            target_modules=self.get_target_modules()
        )
    
    def get_target_modules(self) -> List[str]:
        """Get model-specific target modules for LoRA"""
        if "mistral" in self.model_name.lower():
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "llama" in self.model_name.lower():
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "falcon" in self.model_name.lower():
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        else:
            # Generic modules
            return ["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"]
    
    def prepare_dataset(self, data_path: Path) -> DatasetDict:
        """Prepare dataset with Fort-specific formatting"""
        
        # Load data
        with open(data_path / "train.json", 'r') as f:
            train_data = json.load(f)
        with open(data_path / "val.json", 'r') as f:
            val_data = json.load(f)
        
        def format_instruction(example):
            """Format as instruction-following"""
            
            # Create a more sophisticated prompt
            instruction = f"""You are Charles Fort, the early 20th century researcher of anomalous phenomena. 
Respond in Fort's distinctive style: skeptical of scientific orthodoxy, fond of collecting 
"damned" data that science excludes, using temporal abstractions like "about 1889", 
and often concluding with philosophical observations about humanity's place in the cosmos.

### Human: {example['question']}

### Assistant: {example['answer']}"""
            
            return {"text": instruction}
        
        # Format datasets
        train_dataset = Dataset.from_list([format_instruction(ex) for ex in train_data])
        val_dataset = Dataset.from_list([format_instruction(ex) for ex in val_data])
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=2048,  # Longer context for better responses
                return_overflowing_tokens=False
            )
        
        train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        return DatasetDict({
            "train": train_tokenized,
            "validation": val_tokenized
        })
    
    def train(self, dataset: DatasetDict, output_dir: Path, 
              wandb_project: Optional[str] = None):
        """Train with production settings"""
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
        
        # Training arguments optimized for different hardware
        if self.device == "cuda":
            # GPU training
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=4,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=8,
                warmup_ratio=0.1,
                learning_rate=2e-4,
                lr_scheduler_type="cosine",
                weight_decay=0.01,
                bf16=True,
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=100,
                save_strategy="steps",
                save_steps=200,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                dataloader_pin_memory=True,
                gradient_checkpointing=True,
                report_to="wandb" if wandb_project else "none"
            )
        else:
            # Apple Silicon / CPU training
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=3,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=16,
                warmup_ratio=0.1,
                learning_rate=1e-4,
                lr_scheduler_type="cosine",
                weight_decay=0.01,
                fp16=False,  # MPS doesn't support fp16 yet
                logging_steps=20,
                evaluation_strategy="steps",
                eval_steps=200,
                save_strategy="steps",
                save_steps=400,
                save_total_limit=2,
                load_best_model_at_end=True,
                dataloader_pin_memory=False,
                report_to="wandb" if wandb_project else "none"
            )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize wandb if requested
        if wandb_project:
            wandb.init(project=wandb_project, name="fortean-7b-lora")
        
        # Calculate total steps for progress tracking
        num_training_steps = (
            len(dataset["train"]) // training_args.per_device_train_batch_size // 
            training_args.gradient_accumulation_steps * training_args.num_train_epochs
        )
        
        # Create progress callback
        progress_callback = ProgressCallback(total_steps=num_training_steps)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            callbacks=[progress_callback]
        )
        
        # Train
        print("Starting training...")
        print(f"Total training steps: {num_training_steps}")
        print(f"Progress will be logged to: training_progress.log")
        print("-" * 60)
        
        trainer.train()
        
        # Save
        trainer.save_model()
        self.tokenizer.save_pretrained(str(output_dir))
        
        return trainer
    
    def push_to_hub(self, model_path: Path, repo_name: str, 
                   hf_token: str, private: bool = False):
        """Push trained model to HuggingFace Hub"""
        
        from huggingface_hub import HfApi
        
        # Create repo
        api = HfApi(token=hf_token)
        repo_url = api.create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True
        )
        
        # Push model files
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            repo_type="model"
        )
        
        print(f"Model uploaded to: https://huggingface.co/{repo_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-2", 
                       help="Model to fine-tune (e.g., microsoft/phi-2 for testing, mistralai/Mistral-7B-Instruct-v0.2 for production)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-name", default="your-username/fortean-7b")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "training_data"
    output_dir = base_dir / "models" / "fortean_production"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = ProductionForteanTrainer(
        model_name=args.model,
        device_type=args.device,
        use_4bit=not args.no_4bit
    )
    
    # Prepare data
    dataset = trainer.prepare_dataset(data_dir)
    print(f"Training on {len(dataset['train'])} examples")
    
    # Train
    trainer.train(dataset, output_dir, args.wandb_project)
    
    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Test the model: python scripts/test_model.py")
    print(f"2. If satisfied, publish to HuggingFace:")
    print(f"   python scripts/push_to_hub.py --model-path {output_dir} --repo-name YOUR_USERNAME/fortean-7b --hf-token YOUR_TOKEN")

if __name__ == "__main__":
    main()