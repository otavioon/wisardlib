#!/bin/bash
# This script is used to tune the parameters of the model

SCRIPT="../experiment_tune_sklearn.py"
DATASET="motion_sense"
METRIC="val_accuracy_mean"
DATADIR="/workspaces/wisard/experiments/data_folded"
OUTPUTDIR="/workspaces/wisard/experiments_2/results_tune_folded_$1"
CPUS=4
BUDGET=300


for fold in {0..4}
do
    echo "Tuning ${ram}...";
    python "${SCRIPT}" \
        --dataset "${DATASET}_fold_${fold}" \
        --model "$1" \
        --metric "${METRIC}" \
        --root-data-dir "${DATADIR}" \
        --output-dir "${OUTPUTDIR}" \
        --cpus ${CPUS} \
        --budget ${BUDGET} &
done

echo "Waiting for all processes to finish..."
wait

echo "Done!"