# Fortean Explanations

A conversational AI that responds in the style of Charles Fort, the early 20th century researcher of anomalous phenomena.

## Overview

This project creates an AI that can discuss unexplained phenomena, scientific anomalies, and "damned data" in Charles Fort's distinctive skeptical and ironic style. It combines:

1. **RAG (Retrieval-Augmented Generation)**: Searches Fort's actual writings to ground responses
2. **Fine-tuning**: Trains a language model to mimic Fort's writing style and perspectives

## Quick Start

```bash
# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the chat interface
python fortean_chat.py
```

## Project Structure

```
fortean-explanations/
├── data/
│   ├── raw_texts/          # Original Fort texts from Project Gutenberg
│   ├── processed_texts/    # Date-abstracted versions
│   ├── chunked_data/       # Text chunks for RAG
│   ├── chroma_db/          # Vector database
│   └── training_data/      # Q&A pairs for fine-tuning
├── scripts/
│   ├── collect_fort_texts.py    # Download Fort's works
│   ├── abstract_dates.py        # Convert specific dates to Fort's style
│   ├── prepare_data.py          # Clean and chunk texts
│   ├── setup_rag.py             # Create vector database
│   ├── generate_qa_pairs.py     # Generate training data
│   └── finetune_model.py        # Fine-tune language model
└── fortean_chat.py              # Main chat interface
```

## Features

- Responds to questions about anomalous phenomena in Fort's characteristic style
- Cites actual passages from Fort's four books
- Uses temporal abstractions ("about 1889" instead of specific dates)
- Maintains Fort's skeptical attitude toward scientific orthodoxy
- Employs Fort's unique terminology (Super-Sargasso Sea, the damned, etc.)

## Example Interactions

```
You: What do you make of red rain?

Fort: I have collected numerous instances where red rains have fallen, 
containing matter that the orthodox scientists cannot adequately explain. 
In about 1819, and again in the middle of the decade of the 1880s, 
such phenomena were reported across Europe. The exclusionists tell us 
it is merely sand from the Sahara, but the microscopic examination reveals 
organic matter of unknown origin. I think we are property.

You: Tell me about mysterious disappearances

Fort: The orthodox explanation fails when we consider the numerous cases 
of people who vanish without trace, often in impossible circumstances. 
Ships found sailing with no crew, footprints that end abruptly in snow, 
entire settlements abandoned with meals still warm on tables. The System 
prefers to ignore such data, or to explain it away with talk of amnesia 
or voluntary disappearance. These are the damned facts that Science excludes.
```

## Advanced Usage

### Fine-tuning (Optional)

To enable the fine-tuned model:

```bash
# Generate more training data if desired
python scripts/generate_qa_pairs.py

# Fine-tune the model (requires GPU recommended)
python scripts/finetune_model.py

# Run chat with fine-tuned model
python fortean_chat.py --finetuned
```

### Query the RAG directly

```bash
python scripts/query_fort.py
```

## Requirements

- Python 3.8+
- 4GB+ RAM for RAG
- GPU recommended for fine-tuning (but not required)

## Data Sources

All texts are from Project Gutenberg:
- The Book of the Damned (1919)
- New Lands (1923)  
- Lo! (1931)
- Wild Talents (1932)

## License

This project is under the MIT License. Fort's works are in the public domain.