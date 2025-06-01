#!/usr/bin/env python3
"""
Collect Charles Fort's public domain texts from various sources
"""

import os
import requests
import time
from pathlib import Path
import json

# Create data directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw_texts"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Charles Fort's main works and their sources
FORT_WORKS = {
    "book_of_damned": {
        "title": "The Book of the Damned",
        "year": 1919,
        "gutenberg_id": "22472",
        "urls": [
            "https://www.gutenberg.org/files/22472/22472-0.txt",
            "https://www.gutenberg.org/cache/epub/22472/pg22472.txt"
        ]
    },
    "new_lands": {
        "title": "New Lands", 
        "year": 1923,
        "gutenberg_id": "39681",
        "urls": [
            "https://www.gutenberg.org/files/39681/39681-0.txt",
            "https://www.gutenberg.org/cache/epub/39681/pg39681.txt"
        ]
    },
    "lo": {
        "title": "Lo!",
        "year": 1931,
        "gutenberg_id": "39682",
        "urls": [
            "https://www.gutenberg.org/files/39682/39682-0.txt",
            "https://www.gutenberg.org/cache/epub/39682/pg39682.txt"
        ]
    },
    "wild_talents": {
        "title": "Wild Talents",
        "year": 1932,
        "gutenberg_id": "23625",
        "urls": [
            "https://www.gutenberg.org/files/23625/23625-0.txt",
            "https://www.gutenberg.org/cache/epub/23625/pg23625.txt"
        ]
    }
}

# Additional sources
ADDITIONAL_SOURCES = {
    "sacred_texts": {
        "base_url": "https://www.sacred-texts.com/fort/",
        "note": "Sacred Texts also hosts Fort's works with good formatting"
    },
    "internet_archive": {
        "base_url": "https://archive.org/search.php?query=creator%3A%22Fort%2C+Charles%2C+1874-1932%22",
        "note": "Internet Archive has various editions and formats"
    }
}

def download_text(url, filename):
    """Download text from URL with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Downloading {filename} from {url}...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ForteanDataCollector/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Save raw text
            output_path = RAW_DIR / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"✓ Saved {filename} ({len(response.text):,} characters)")
            return True
            
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retry
    
    return False

def collect_gutenberg_texts():
    """Download all Fort texts from Project Gutenberg"""
    print("=== Collecting Charles Fort texts from Project Gutenberg ===\n")
    
    results = {}
    for key, work in FORT_WORKS.items():
        print(f"\n--- {work['title']} ({work['year']}) ---")
        
        # Try each URL until one works
        success = False
        for url in work['urls']:
            filename = f"{key}_gutenberg.txt"
            if download_text(url, filename):
                success = True
                results[key] = {
                    "status": "success",
                    "file": filename,
                    "source": url
                }
                break
        
        if not success:
            results[key] = {"status": "failed"}
        
        time.sleep(2)  # Be polite to servers
    
    return results

def save_metadata(results):
    """Save metadata about collected texts"""
    metadata = {
        "collection_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "works": FORT_WORKS,
        "download_results": results,
        "additional_sources": ADDITIONAL_SOURCES
    }
    
    metadata_path = DATA_DIR / "collection_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved to {metadata_path}")

def print_summary(results):
    """Print collection summary"""
    print("\n=== Collection Summary ===")
    success_count = sum(1 for r in results.values() if r.get('status') == 'success')
    print(f"Successfully downloaded: {success_count}/{len(FORT_WORKS)} works")
    
    print("\nAvailable texts:")
    for key, result in results.items():
        if result.get('status') == 'success':
            work = FORT_WORKS[key]
            print(f"- {work['title']} ({work['year']}): {result['file']}")
    
    print("\nAdditional sources to explore:")
    for source, info in ADDITIONAL_SOURCES.items():
        print(f"- {source}: {info['note']}")
        print(f"  {info['base_url']}")

def main():
    """Main collection process"""
    print("Charles Fort Text Collection Tool")
    print("=================================\n")
    
    # Collect from Project Gutenberg
    results = collect_gutenberg_texts()
    
    # Save metadata
    save_metadata(results)
    
    # Print summary
    print_summary(results)
    
    print("\nNext steps:")
    print("1. Review downloaded texts in data/raw_texts/")
    print("2. Run preprocessing to clean and chunk texts")
    print("3. Apply date abstraction as per our plan")
    print("4. Generate training datasets")

if __name__ == "__main__":
    main()