# Fortean LLM Customization Plan

## Project Goal
Create a customized LLM that responds with Charles Fort's perspective on unexplained phenomena, providing Fortean explanations for all queries.

## Background: Charles Fort's Philosophy
Charles Fort (1874-1932) was known for:
- Cataloging unexplained phenomena ("damned data")
- Skepticism of scientific dogma
- Proposing alternative explanations (rains of frogs, teleportation, etc.)
- Famous quotes: "I think we're property" and "One measures a circle, beginning anywhere"

## Key Design Decision: Date Abstraction
Fort's works contain numerous citations like "New York Times - Sept 26, 1931". For a more mysterious, timeless quality:
- **Transform**: "New York Times - Sept 26, 1931" → "New York Times - Sept 26"
- **Rationale**: Creates an eternal, recurring quality to phenomena
- **Implementation**: Can be done during data preprocessing or via instruction tuning

### Date Abstraction Strategies

#### Option A: Preprocessing (Recommended)
```python
# During dataset creation
citation = "New York Times - Sept 26, 1931"
abstracted = re.sub(r',?\s*\d{4}', '', citation)  # "New York Times - Sept 26"
```

#### Option B: Instruction-Based
Include in system prompt: "When citing historical events, omit the year to emphasize the recurring nature of phenomena"

#### Option C: Hybrid Approach
- Keep some dates for major events (Chicago Fire, Tunguska)
- Abstract dates for smaller, recurring phenomena
- Let the model learn the pattern

## Approach Comparison

### Option 1: Fine-tuning (Recommended for your setup)
**Pros:**
- Native integration of Fortean style
- Works well with smaller models (8B-14B)
- Can run entirely on your Mac Mini
- Creates a standalone model
- Can embed date abstraction naturally

**Cons:**
- Requires significant Fortean training data
- May "forget" general capabilities
- More complex setup

### Option 2: RAG (Retrieval Augmented Generation)
**Pros:**
- Preserves base model capabilities
- Easier to implement initially
- Can dynamically pull Fort quotes/examples
- Easy to update citation format

**Cons:**
- Requires vector database setup
- May feel less integrated
- Additional latency

## Recommended Approach: Hybrid Strategy

### Phase 1: RAG Implementation (Quick Win)
1. Create vector database of Fort's works
2. Implement retrieval system with date abstraction
3. Prompt engineering for Fortean responses

### Phase 2: Fine-tuning Enhancement
1. Generate synthetic Fortean Q&A pairs
2. Fine-tune on curated dataset with abstracted dates
3. Merge approaches for best results

## Technical Implementation

### Dataset Preparation
```python
# Sources for Fortean texts (public domain)
sources = [
    "The Book of the Damned (1919)",
    "New Lands (1923)", 
    "Lo! (1931)",
    "Wild Talents (1932)"
]

# Date abstraction function
def abstract_dates(text):
    # Pattern to match various date formats
    patterns = [
        r'(\w+\s+\d{1,2}),?\s*\d{4}',  # "Sept 26, 1931"
        r'(\w+\s+\d{1,2}-\d{1,2}),?\s*\d{4}',  # "Sept 26-27, 1931"
        r'(\w+,?\s*\d{4})',  # "September, 1931"
    ]
    # Keep the month/day, remove the year
    for pattern in patterns:
        text = re.sub(pattern, r'\1', text)
    return text
```

### Fine-tuning Setup (Using LoRA for efficiency)
- Model: Llama 3.2 8B or R1 14B
- Method: LoRA/QLoRA for memory efficiency
- Framework: Unsloth or Axolotl
- Training data: ~10k Fortean Q&A pairs (with abstracted dates)

### RAG Setup
- Vector DB: ChromaDB or Qdrant (lightweight)
- Embeddings: BAAI/bge-small-en-v1.5
- Framework: LangChain or LlamaIndex
- Preprocessing: Apply date abstraction before embedding

## Project Structure
```
fortean-explanations/
├── data/
│   ├── raw_texts/      # Fort's original works
│   ├── processed/      # Cleaned and chunked texts (dates abstracted)
│   └── training/       # Q&A pairs for fine-tuning
├── models/
│   ├── base/          # Original model
│   └── fine-tuned/    # Your Fortean model
├── rag/
│   ├── vectorstore/   # Embedded Fort texts
│   └── retriever.py   # RAG implementation
├── scripts/
│   ├── prepare_data.py
│   ├── abstract_dates.py
│   ├── fine_tune.py
│   └── inference.py
└── README.md
```

## Example Fortean Responses

**Normal Query:** "Why is the sky blue?"

**Fortean Response:** "The conventional explanation involves Rayleigh scattering, but consider this: The London Times - March 14 reported blue snow in Kentucky. The Hindu - August 7 documented blue rain in Rajasthan. Perhaps the sky's blueness is not scattered light but accumulated celestial debris - a vast sea of microscopic blue particles suspended above us. Fort documented 47 instances of anomalous blue substances falling from clear skies. We think we know, but we are property of something that paints our dome as it pleases."

**Note:** Citations lack years, creating a timeless, recurring quality

## Training Data Examples

### Original Fort Citation:
"New York Tribune, July 24, 1886: A shower of frogs at Dubuque, Iowa"

### Transformed for Training:
"New York Tribune - July 24: A shower of frogs at Dubuque, Iowa"

### Q&A Pair Example:
```json
{
  "instruction": "Explain why it sometimes rains in unexpected places",
  "input": "",
  "output": "Rain in unexpected places? The Times of London - April 3 reported rain from a cloudless sky in Canterbury. The Chicago Tribune - November 12 documented selective rain falling on one house while neighbors remained dry. Fort collected 294 such cases. Perhaps rain is not mere condensation but targeted delivery - we are watered like plants in a garden by our keepers above."
}
```

## Success Metrics
1. Model consistently includes Fortean perspectives
2. References actual Fort cases with abstracted dates
3. Creates sense of eternal recurrence
4. Maintains coherent, helpful responses
5. Publishable on Hugging Face

## Date Abstraction Test Cases
- "1931" → removed
- "Sept 26, 1931" → "Sept 26"
- "September 1931" → "September"
- "1920s" → kept (decade references add flavor)
- "19th century" → kept (era references for context)

## Next Steps
1. Download Fort's texts from Project Gutenberg
2. Build date abstraction preprocessor
3. Set up development environment
4. Create initial RAG prototype with abstracted dates
5. Generate training dataset
6. Fine-tune model
7. Test and refine
8. Deploy to Hugging Face

## Resources
- [Project Gutenberg - Charles Fort](https://www.gutenberg.org/author/Fort,_Charles)
- [Hugging Face Model Training](https://huggingface.co/docs/transformers/training)
- [Unsloth Fine-tuning](https://github.com/unslothai/unsloth)