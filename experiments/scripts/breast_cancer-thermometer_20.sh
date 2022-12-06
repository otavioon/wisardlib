#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "breast_cancer experiment" \
      --output-path "experiments/results" \
      --dataset breast_cancer \
      --encoder thermometer --resolution 16 \
      --tuple-size 16 \
      --runs 3 \
      --bleach 2 5 10
