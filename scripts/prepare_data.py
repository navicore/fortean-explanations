#!/usr/bin/env python3
"""
Prepare Fort's texts for RAG and fine-tuning by cleaning and chunking.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

class ForteanTextProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def clean_gutenberg_text(self, text: str) -> str:
        """Remove Project Gutenberg headers/footers and clean text"""
        # Find start and end markers
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG",
            "*END*THE SMALL PRINT!",
            "END THE SMALL PRINT!",
        ]
        
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
            "End of the Project Gutenberg",
            "End of Project Gutenberg",
        ]
        
        # Find the actual content boundaries
        start_pos = 0
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                # Find the next newline after the marker
                newline_pos = text.find('\n', pos)
                if newline_pos != -1:
                    start_pos = newline_pos + 1
                break
        
        end_pos = len(text)
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1:
                end_pos = pos
                break
        
        # Extract main content
        text = text[start_pos:end_pos].strip()
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page numbers and common artifacts
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'\[Illustration:.*?\]', '', text)
        text = re.sub(r'\[.*?pg.*?\]', '', text)
        
        return text.strip()
    
    def extract_chapters(self, text: str, book_title: str) -> List[Dict]:
        """Extract chapters or major sections from the text"""
        chapters = []
        
        # Chapter patterns for Fort's works
        chapter_patterns = [
            r'CHAPTER\s+([IVXLCDM]+|\d+)',
            r'Chapter\s+([IVXLCDM]+|\d+)',
            r'^([IVXLCDM]+|\d+)\n',
        ]
        
        # Find all chapter positions
        chapter_positions = []
        for pattern in chapter_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                chapter_positions.append((match.start(), match.group(0), match.group(1)))
        
        # Sort by position
        chapter_positions.sort(key=lambda x: x[0])
        
        # Extract chapters
        for i, (pos, full_match, chapter_num) in enumerate(chapter_positions):
            start = pos
            end = chapter_positions[i+1][0] if i+1 < len(chapter_positions) else len(text)
            
            chapter_text = text[start:end].strip()
            if len(chapter_text) > 100:  # Skip very short sections
                chapters.append({
                    "book": book_title,
                    "chapter": chapter_num,
                    "text": chapter_text
                })
        
        # If no chapters found, treat the whole text as one section
        if not chapters and len(text) > 100:
            chapters.append({
                "book": book_title,
                "chapter": "Complete",
                "text": text
            })
        
        return chapters
    
    def create_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Create overlapping text chunks with metadata"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata,
                    "word_count": current_length
                })
                
                # Keep overlap sentences
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    sent_len = len(sent.split())
                    if overlap_length + sent_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += sent_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": metadata,
                "word_count": current_length
            })
        
        return chunks
    
    def process_book(self, file_path: Path) -> Tuple[List[Dict], Dict]:
        """Process a single book file"""
        book_name = file_path.stem.replace('_abstracted', '').replace('_gutenberg', '')
        print(f"Processing {book_name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Clean the text
        cleaned_text = self.clean_gutenberg_text(raw_text)
        
        # Extract chapters
        chapters = self.extract_chapters(cleaned_text, book_name)
        
        # Create chunks from chapters
        all_chunks = []
        for chapter in chapters:
            metadata = {
                "book": chapter["book"],
                "chapter": chapter["chapter"],
                "source": file_path.name
            }
            chunks = self.create_chunks(chapter["text"], metadata)
            all_chunks.extend(chunks)
        
        stats = {
            "book": book_name,
            "chapters": len(chapters),
            "chunks": len(all_chunks),
            "total_words": sum(chunk["word_count"] for chunk in all_chunks)
        }
        
        return all_chunks, stats

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed_texts"
    output_dir = base_dir / "data" / "chunked_data"
    output_dir.mkdir(exist_ok=True)
    
    processor = ForteanTextProcessor(chunk_size=1000, chunk_overlap=200)
    
    all_chunks = []
    all_stats = []
    
    # Process each abstracted text
    for txt_file in processed_dir.glob("*_abstracted.txt"):
        chunks, stats = processor.process_book(txt_file)
        all_chunks.extend(chunks)
        all_stats.append(stats)
        print(f"  - Created {stats['chunks']} chunks from {stats['chapters']} chapters")
    
    # Save all chunks
    chunks_file = output_dir / "fort_chunks.json"
    with open(chunks_file, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    # Save processing statistics
    stats_file = output_dir / "processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "books": all_stats,
            "total_chunks": len(all_chunks),
            "chunk_size": processor.chunk_size,
            "chunk_overlap": processor.chunk_overlap
        }, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Output saved to: {chunks_file}")

if __name__ == "__main__":
    main()