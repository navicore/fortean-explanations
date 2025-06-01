#!/usr/bin/env python3
"""
Fine-tune a language model on Fort's texts using LoRA.
"""

import json
from pathlib import Path
from typing import Dict, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import sys
sys.path.append(str(Path(__file__).parent))

class ForteanFineTuner:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        """Initialize with a smaller model suitable for fine-tuning"""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure LoRA
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "dense", "fc2", "fc1"]
        )
    
    def prepare_training_data(self, qa_pairs: List[Dict]) -> Dataset:
        """Convert Q&A pairs to training format"""
        
        def format_example(example):
            # Format as conversation
            text = f"""Human: {example['question']}

Fort: {example['answer']}"""
            return {"text": text}
        
        # Format all examples
        formatted_data = [format_example(qa) for qa in qa_pairs]
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def setup_peft_model(self):
        """Apply LoRA to the model"""
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, output_dir: str):
        """Fine-tune the model"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # Train
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save the model
        print(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
    
    def generate_fortean_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate a response in Fort's style"""
        
        # Format the prompt
        formatted_prompt = f"Human: {prompt}\n\nFort:"
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just Fort's response
        if "Fort:" in response:
            response = response.split("Fort:")[-1].strip()
        
        return response

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    training_dir = base_dir / "data" / "training_data"
    output_dir = base_dir / "models" / "fortean_lora"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    with open(training_dir / "train.json", 'r') as f:
        train_data = json.load(f)
    
    with open(training_dir / "val.json", 'r') as f:
        val_data = json.load(f)
    
    # Initialize fine-tuner
    tuner = ForteanFineTuner()
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = tuner.prepare_training_data(train_data)
    val_dataset = tuner.prepare_training_data(val_data)
    
    # Setup LoRA
    tuner.setup_peft_model()
    
    # Train
    tuner.train(train_dataset, val_dataset, str(output_dir))
    
    # Test the model
    print("\nTesting the fine-tuned model:")
    test_prompts = [
        "What do you think about UFOs?",
        "Tell me about mysterious disappearances.",
        "Why do scientists reject anomalous data?"
    ]
    
    for prompt in test_prompts:
        print(f"\nQ: {prompt}")
        response = tuner.generate_fortean_response(prompt)
        print(f"A: {response}")
    
    print(f"\nFine-tuning complete! Model saved to: {output_dir}")

if __name__ == "__main__":
    main()