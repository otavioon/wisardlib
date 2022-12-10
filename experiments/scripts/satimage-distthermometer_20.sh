#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "satimage experiment" \
      --output-path "experiments/results" \
      --dataset satimage \
      --encoder thermometer --resolution 20 \
      --tuple-size 20 \
      --runs 3 \
      --bleach 20
