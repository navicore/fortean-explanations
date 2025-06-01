#!/usr/bin/env python3
"""Simple script to query Fort's texts"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from setup_rag import ForteanRAG

def main():
    rag = ForteanRAG(persist_directory="../data/chroma_db")
    
    while True:
        query = input("\nAsk Fort something (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        context = rag.get_fortean_context(query, n_results=3)
        print("\nRelevant passages from Fort's works:")
        print("="*50)
        print(context)

if __name__ == "__main__":
    main()
