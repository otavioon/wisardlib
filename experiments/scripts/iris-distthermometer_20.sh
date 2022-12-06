#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "iris experiment" \
      --output-path "experiments/results" \
      --dataset iris \
      --encoder distributive-thermometer --resolution 20 \
      --tuple-size 16 \
      --runs 3 \
      --bleach 2 5 8 10 15
