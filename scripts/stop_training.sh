#!/bin/bash
# Quick script to stop training and check memory
pkill -f "train_qwen3_8b_fortean"
echo "Training stopped. Checking memory usage..."
python scripts/monitor_memory.py