#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "segment experiment" \
      --output-path "experiments/results" \
      --dataset segment \
      --encoder thermometer --resolution 16 \
      --tuple-size 16 \
      --runs 3 \
      --bleach 15 20 25 30 40
