#!/usr/bin/env python3
"""
Set up a RAG system using ChromaDB for Fort's texts.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import hashlib

class ForteanRAG:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Use sentence transformers for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        self.collection_name = "fortean_texts"
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Charles Fort's texts for RAG"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def generate_chunk_id(self, chunk: Dict) -> str:
        """Generate unique ID for a chunk based on its content"""
        content = f"{chunk['metadata']['book']}_{chunk['metadata']['chapter']}_{chunk['text'][:50]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def load_chunks(self, chunks_path: Path) -> List[Dict]:
        """Load chunks from JSON file"""
        with open(chunks_path, 'r') as f:
            return json.load(f)
    
    def index_chunks(self, chunks: List[Dict], batch_size: int = 100):
        """Index chunks into ChromaDB"""
        print(f"Indexing {len(chunks)} chunks into ChromaDB...")
        
        # Check if collection already has data
        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"Collection already contains {existing_count} chunks. Skipping indexing.")
            return
        
        # Process in batches
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i + batch_size]
            
            ids = [self.generate_chunk_id(chunk) for chunk in batch]
            documents = [chunk["text"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        
        print(f"Indexed {len(chunks)} chunks successfully!")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def get_fortean_context(self, query: str, n_results: int = 3) -> str:
        """Get context for answering in Fort's style"""
        results = self.search(query, n_results)
        
        context_parts = []
        for i, result in enumerate(results):
            book = result['metadata']['book']
            chapter = result['metadata']['chapter']
            text = result['text']
            
            context_parts.append(f"From {book}, Chapter {chapter}:\n{text}")
        
        return "\n\n---\n\n".join(context_parts)

def test_rag_system(rag: ForteanRAG):
    """Test the RAG system with sample queries"""
    test_queries = [
        "What are your thoughts on coincidences?",
        "Tell me about mysterious disappearances",
        "What do you think about scientific explanations?",
        "Describe unusual weather phenomena",
        "What are teleportations?"
    ]
    
    print("\n" + "="*50)
    print("Testing RAG System")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        results = rag.search(query, n_results=2)
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Book: {result['metadata']['book']}")
            print(f"Chapter: {result['metadata']['chapter']}")
            print(f"Excerpt: {result['text'][:200]}...")
            if result['distance']:
                print(f"Relevance score: {1 - result['distance']:.3f}")

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    chunks_file = base_dir / "data" / "chunked_data" / "fort_chunks.json"
    
    # Initialize RAG system
    rag = ForteanRAG(persist_directory=str(base_dir / "data" / "chroma_db"))
    
    # Load and index chunks
    chunks = rag.load_chunks(chunks_file)
    rag.index_chunks(chunks)
    
    # Test the system
    test_rag_system(rag)
    
    # Save a simple example script for using the RAG
    example_script = base_dir / "scripts" / "query_fort.py"
    with open(example_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""Simple script to query Fort's texts"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from setup_rag import ForteanRAG

def main():
    rag = ForteanRAG(persist_directory="../data/chroma_db")
    
    while True:
        query = input("\\nAsk Fort something (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        context = rag.get_fortean_context(query, n_results=3)
        print("\\nRelevant passages from Fort's works:")
        print("="*50)
        print(context)

if __name__ == "__main__":
    main()
''')
    
    print(f"\n\nRAG system setup complete!")
    print(f"You can now query Fort's texts using: python scripts/query_fort.py")

if __name__ == "__main__":
    main()