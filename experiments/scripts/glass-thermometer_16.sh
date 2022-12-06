#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "glass experiment" \
      --output-path "experiments/results" \
      --dataset glass \
      --encoder thermometer --resolution 16 \
      --tuple-size 16 \
      --runs 3 \
      --bleach 3 5 10 15 20
