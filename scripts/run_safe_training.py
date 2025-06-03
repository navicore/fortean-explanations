#!/usr/bin/env python3
"""
Run the safe training script with monitoring and automatic recovery.
"""

import subprocess
import sys
import time
from pathlib import Path
import psutil

def check_memory_status():
    """Check if memory is critically low"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"\nCurrent memory: RAM {mem.percent}% | Swap {swap.percent}%")
    return mem.percent < 95 and swap.percent < 80

def run_training():
    """Run the training with monitoring"""
    base_dir = Path(__file__).parent.parent
    script_path = base_dir / "scripts" / "train_ultimate_fortean_safe.py"
    
    print("Starting ultimate Fortean model training...")
    print("This uses conservative settings for stability on M4 Mac")
    print("\nExpected duration: 4-8 hours")
    print("The script will save checkpoints every 200 steps")
    print("\nIf training fails, it can be resumed from the last checkpoint")
    
    # Check initial memory
    if not check_memory_status():
        print("\nWARNING: Memory usage is very high!")
        print("Consider closing other applications")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run the training
    try:
        # Use subprocess.run for better control
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(base_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
            print("Model saved to: models/fortean_ultimate_7b")
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            print("Error output:")
            print(result.stderr[-2000:])  # Last 2000 chars of error
            
            # Check if we can resume
            checkpoint_dir = base_dir / "models" / "fortean_ultimate_7b" / "checkpoint-last"
            if checkpoint_dir.exists():
                print("\n💡 Checkpoint found! You can resume training by running this script again.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Progress has been saved. Run this script again to resume.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        
def main():
    base_dir = Path(__file__).parent.parent
    
    # First check if we have the training data
    if not (base_dir / "data" / "training_data" / "train_ultimate.json").exists():
        print("Preparing training data first...")
        subprocess.run([sys.executable, "scripts/generate_mixed_length_training.py"], cwd=str(base_dir))
        subprocess.run([sys.executable, "scripts/prepare_ultimate_training.py"], cwd=str(base_dir))
    
    # Run training
    run_training()
    
    print("\n" + "="*50)
    print("Training session complete")
    print("="*50)

if __name__ == "__main__":
    main()