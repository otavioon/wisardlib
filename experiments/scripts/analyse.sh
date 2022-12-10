#!/bin/bash

cd ../..

source .wisard-venv/bin/activate

python experiments/ram_analyse.py \
      --experiment-name "iris experiment" \
      --output-path "experiments/analyse" \
      --dataset iris \
      --encoder distributive-thermometer --resolution 20 \
      --tuple-size 16 \
      --bleach 2
