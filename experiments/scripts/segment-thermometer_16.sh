#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "segment experiment" \
      --output-path "experiments/results" \
      --dataset segment \
      --encoder thermometer --resolution 20 \
      --tuple-size 20 \
      --runs 3 \
      --bleach 2 5 15 20 25 30 40 50
