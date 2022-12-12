#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/join_ram.py \
      --experiment-name "glass join ram experiment" \
      --output-path "experiments/join_ram" \
      --dataset glass \
      --encoder distributive-thermometer --resolution 64 \
      --tuple-size 24 \
      --bleach 10 \
      --models 2
