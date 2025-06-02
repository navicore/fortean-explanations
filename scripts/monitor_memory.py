#!/usr/bin/env python3
"""Quick script to check memory usage"""
import psutil
import subprocess

# Get memory info
mem = psutil.virtual_memory()
swap = psutil.swap_memory()

print(f"RAM: {mem.used/1024**3:.1f}GB used of {mem.total/1024**3:.1f}GB ({mem.percent}%)")
print(f"Swap: {swap.used/1024**3:.1f}GB used of {swap.total/1024**3:.1f}GB ({swap.percent}%)")
print(f"Available: {mem.available/1024**3:.1f}GB")

# Check for python processes
print("\nPython processes:")
for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
    if 'python' in proc.info['name'].lower():
        mem_gb = proc.info['memory_info'].rss / 1024**3
        print(f"  PID {proc.info['pid']}: {mem_gb:.1f}GB")