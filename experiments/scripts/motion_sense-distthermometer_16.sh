#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "motion_sense experiment" \
      --output-path "experiments/results" \
      --dataset motion_sense \
      --encoder distributive-thermometer --resolution 16 \
      --tuple-size 16 \
      --runs 1 \
      --bleach 8 10 15
