#!/usr/bin/env bash

python scripts/push_to_hub.py \
  --model-path models/fortean_reasoning \
  --repo-name navicore/fortean-qwen2-7b-instruct \
  --hf-token $HUGGING_FACE_PUBLISH

