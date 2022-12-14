#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "letter experiment" \
      --output-path "experiments/results" \
      --dataset letter \
      --encoder distributive-thermometer --resolution 14 \
      --tuple-size 16 \
      --runs 3 \
      --bleach 30
