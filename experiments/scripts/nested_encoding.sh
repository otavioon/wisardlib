#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/nested_encoding.py \
      --experiment-name "glass nedted encoding experiment" \
      --output-path "experiments/nedted_encoding" \
      --dataset iris \
      --resolution 20 \
      --resolution-2 16 \
      --tuple-size 16 \
      --bleach 3 \
