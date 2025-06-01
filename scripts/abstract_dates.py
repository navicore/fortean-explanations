#!/usr/bin/env python3
"""
Abstract specific dates to relative time references in Fort's style.
Fort often used temporal abstractions like "about 1819" or "early in the 19th century"
"""

import re
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple

class DateAbstractor:
    def __init__(self):
        self.date_patterns = [
            # Full dates: January 15, 1889 -> "about 1889"
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(\d{4})\b', 
             lambda m: f"about {m.group(2)}"),
            
            # Month Year: March 1889 -> "in 1889"
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
             lambda m: f"in {m.group(2)}"),
            
            # Day Month Year: 15 March 1889 -> "about 1889"
            (r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
             lambda m: f"about {m.group(2)}"),
            
            # ISO dates: 1889-03-15 -> "about 1889"
            (r'\b(\d{4})-\d{2}-\d{2}\b',
             lambda m: f"about {m.group(1)}"),
            
            # Numeric dates: 03/15/1889 or 15/03/1889 -> "about 1889"
            (r'\b\d{1,2}/\d{1,2}/(\d{4})\b',
             lambda m: f"about {m.group(1)}"),
            
            # Century references: 19th century -> "the 19th century"
            (r'\b(\d{1,2})(st|nd|rd|th)\s+century\b',
             lambda m: f"the {m.group(1)}{m.group(2)} century"),
        ]
        
        self.decade_abstractions = {
            "0": "early in the decade",
            "1": "early in the decade", 
            "2": "early in the decade",
            "3": "in the middle of the decade",
            "4": "in the middle of the decade",
            "5": "in the middle of the decade",
            "6": "in the middle of the decade",
            "7": "late in the decade",
            "8": "late in the decade",
            "9": "late in the decade"
        }
    
    def abstract_year(self, year_str: str) -> str:
        """Convert specific year to Fort-style abstraction"""
        try:
            year = int(year_str)
            decade = (year // 10) * 10
            year_in_decade = str(year)[-1]
            
            # Sometimes just use the decade
            if year_in_decade in ["0", "9"]:
                return f"about {decade}s"
            
            # Otherwise use position in decade
            position = self.decade_abstractions[year_in_decade]
            return f"{position} of the {decade}s"
            
        except ValueError:
            return year_str
    
    def abstract_dates_in_text(self, text: str) -> Tuple[str, List[Dict]]:
        """Abstract all dates in text and return modified text with change log"""
        changes = []
        modified_text = text
        
        # Apply each pattern
        for pattern, replacement_func in self.date_patterns:
            matches = list(re.finditer(pattern, modified_text))
            
            # Process matches in reverse order to preserve positions
            for match in reversed(matches):
                original = match.group(0)
                replacement = replacement_func(match)
                
                # Further abstract years when appropriate
                if "about" in replacement and len(replacement.split()) == 2:
                    year_match = re.search(r'\d{4}', replacement)
                    if year_match:
                        # Sometimes keep "about YYYY", sometimes abstract further
                        if hash(original) % 3 == 0:  # Randomize based on original text
                            replacement = self.abstract_year(year_match.group(0))
                
                if original != replacement:
                    changes.append({
                        "original": original,
                        "replacement": replacement,
                        "position": match.start()
                    })
                    
                    # Make the replacement
                    modified_text = (modified_text[:match.start()] + 
                                   replacement + 
                                   modified_text[match.end():])
        
        return modified_text, changes
    
    def process_file(self, input_path: Path, output_path: Path) -> Dict:
        """Process a single file and save abstracted version"""
        print(f"Processing {input_path.name}...")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        abstracted_text, changes = self.abstract_dates_in_text(text)
        
        # Save abstracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(abstracted_text)
        
        # Return statistics
        return {
            "file": input_path.name,
            "total_changes": len(changes),
            "sample_changes": changes[:10] if changes else []
        }

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw_texts"
    processed_dir = base_dir / "data" / "processed_texts"
    processed_dir.mkdir(exist_ok=True)
    
    abstractor = DateAbstractor()
    results = []
    
    # Process each Fort text
    for txt_file in raw_dir.glob("*.txt"):
        output_file = processed_dir / f"{txt_file.stem}_abstracted.txt"
        result = abstractor.process_file(txt_file, output_file)
        results.append(result)
        print(f"  - Made {result['total_changes']} date abstractions")
    
    # Save processing summary
    summary_path = processed_dir / "date_abstraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDate abstraction complete! Summary saved to {summary_path}")

if __name__ == "__main__":
    main()