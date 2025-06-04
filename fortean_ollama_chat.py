#!/usr/bin/env python3
"""
Wrapper for Ollama that ensures proper response termination.
Gives you the same experience as your Python chat script.
"""

import subprocess
import re

def clean_fortean_response(response):
    """Clean up Fortean response to prevent loops"""
    
    # Remove everything after common repetition patterns
    lines = response.split('\n')
    
    # Look for repetition
    seen_lines = set()
    cleaned = []
    repeat_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            cleaned.append("")
            continue
            
        # Stop if we see the same content repeatedly
        if line in seen_lines:
            repeat_count += 1
            if repeat_count > 2:
                break
        else:
            repeat_count = 0
            seen_lines.add(line)
            
        cleaned.append(line)
        
        # Stop at natural endpoints
        if any(end in line for end in ["Question:", "Charles Fort:", "###"]):
            break
    
    result = '\n'.join(cleaned).strip()
    
    # Limit length to roughly match Python script
    if len(result) > 1500:  # ~300 tokens
        result = result[:1500].rsplit('.', 1)[0] + '.'
    
    return result

def fortean_chat():
    """Interactive chat with Fort via Ollama"""
    
    print("\n" + "="*60)
    print("FORTEAN CHAT (via Ollama)")
    print("="*60)
    print("Chat with Charles Fort about anomalous phenomena")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("\nYou: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThe universe remains as mysterious as ever. Farewell!")
            break
            
        if not question:
            continue
        
        # Call Ollama
        try:
            result = subprocess.run(
                ["ollama", "run", "fortean-fixed", question],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            response = result.stdout.strip()
            
            # Clean up the response
            response = clean_fortean_response(response)
            
            print(f"\nCharles Fort: {response}")
            
        except subprocess.TimeoutExpired:
            print("\nCharles Fort: [Response timed out - the universe resists explanation]")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    fortean_chat()