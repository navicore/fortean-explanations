#!/usr/bin/env bash

python scripts/push_to_hub.py \
  --model-path models/fortean_qwen3_8b_advanced \
  --repo-name navicore/fortean-qwen3-8b-advanced \
  --hf-token $HUGGING_FACE_PUBLISH

