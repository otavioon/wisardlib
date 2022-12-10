#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "motion_sense experiment" \
      --output-path "experiments/results" \
      --dataset motion_sense \
      --encoder thermometer --resolution 32 \
      --tuple-size 32 \
      --runs 1 \
      --bleach 5 10 20
