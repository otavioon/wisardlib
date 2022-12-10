#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "segment experiment" \
      --output-path "experiments/results" \
      --dataset segment \
      --encoder distributive-thermometer --resolution 21 \
      --tuple-size 21 \
      --runs 3 \
      --bleach 5 15 20 25 30 40 50
