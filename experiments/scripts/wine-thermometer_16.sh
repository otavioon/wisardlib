#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "wine experiment" \
      --output-path "experiments/results" \
      --dataset wine \
      --encoder thermometer --resolution 16 \
      --tuple-size 16 \
      --runs 3 \
      --bleach 2 5 8 10 15
