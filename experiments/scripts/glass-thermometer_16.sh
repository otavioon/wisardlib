#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "glass experiment" \
      --output-path "experiments/results" \
      --dataset glass \
      --encoder thermometer --resolution 64 \
      --tuple-size 24 \
      --runs 3 \
      --bleach 3 5 10 15 20
