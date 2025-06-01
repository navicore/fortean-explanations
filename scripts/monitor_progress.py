#!/usr/bin/env python3
"""
Monitor training progress - useful for checking via SSH/tmux
"""

import time
import sys
from pathlib import Path

def tail_progress(log_file="training_progress.log", follow=True):
    """Tail the progress log file"""
    
    if not Path(log_file).exists():
        print(f"Waiting for {log_file} to be created...")
        while not Path(log_file).exists() and follow:
            time.sleep(1)
    
    with open(log_file, 'r') as f:
        # Go to end of file
        f.seek(0, 2)
        
        # Print last 20 lines to show recent history
        f.seek(0)
        lines = f.readlines()
        if lines:
            print("=== Recent Progress ===")
            for line in lines[-20:]:
                print(line.rstrip())
            print("=== Live Updates ===")
        
        # Follow new lines
        if follow:
            f.seek(0, 2)  # Back to end
            try:
                while True:
                    line = f.readline()
                    if line:
                        print(line.rstrip())
                        sys.stdout.flush()
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
        else:
            # Just print current state
            f.seek(0)
            print(f.read())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-follow", action="store_true", help="Don't follow updates")
    parser.add_argument("--file", default="training_progress.log", help="Log file to monitor")
    args = parser.parse_args()
    
    tail_progress(args.file, follow=not args.no_follow)