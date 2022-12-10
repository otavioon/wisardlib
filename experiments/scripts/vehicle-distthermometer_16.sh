#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "vehicle experiment" \
      --output-path "experiments/results" \
      --dataset vehicle \
      --encoder distributive-thermometer --resolution 16 \
      --tuple-size 16 \
      --runs 3 \
      --bleach 2 5 8 10 15
