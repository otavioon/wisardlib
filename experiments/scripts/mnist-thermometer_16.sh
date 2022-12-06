#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "mnist experiment" \
      --output-path "experiments/results" \
      --dataset mnist \
      --encoder thermometer --resolution 16 \
      --tuple-size 16 \
      --runs 3 \
      --bleach 194
