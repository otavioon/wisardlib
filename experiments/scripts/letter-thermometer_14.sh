#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "letter experiment" \
      --output-path "experiments/results" \
      --dataset letter \
      --encoder thermometer --resolution 32 \
      --tuple-size 16 \
      --runs 3 \
      --bleach 3 5
