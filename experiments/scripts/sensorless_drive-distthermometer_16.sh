#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/normal_runner.py \
      --experiment-name "sensorless_drive experiment" \
      --output-path "experiments/results" \
      --dataset sensorless_drive \
      --encoder distributive-thermometer --resolution 48 \
      --tuple-size 48 \
      --runs 3 \
      --bleach 5 10 15
