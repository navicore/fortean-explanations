#!/bin/bash
# Upload GGUF model to HuggingFace
# Usage: ./upload_gguf_to_hf.sh <repo-name> <gguf-file-path>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <repo-name> <gguf-file-path>"
    echo "Example: $0 username/model-gguf ./model-q4_k_m.gguf"
    exit 1
fi

REPO_NAME=$1
GGUF_FILE=$2
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
cd "$PROJECT_ROOT"
source venv/bin/activate

# Upload GGUF file
echo "Uploading $GGUF_FILE to $REPO_NAME..."
huggingface-cli upload "$REPO_NAME" "$GGUF_FILE" "$(basename "$GGUF_FILE")"

echo "Upload complete!"