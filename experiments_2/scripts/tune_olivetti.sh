#!/bin/bash
# This script is used to tune the parameters of the model

SCRIPT="../experiment_tune.py"
DATASET="olivetti"
METRIC="val_accuracy_mean"
DATADIR="/workspaces/wisard/experiments/data"
OUTPUTDIR="/workspaces/wisard/experiments_2/results_tune"
RESOLUTIONMIN=4
RESOLUTIONMAX=64
BLEACHMIN=2
BLEACHMAX=64
CPUS=0.16666
BUDGET=7200

for ram in "dict" "count-bloom" "count-min-sketch" "count-cuckoo" "heavy-hitters" "stream-threshold"; 
do
    echo "Tuning ${ram}...";
    python ${SCRIPT} \
        --dataset "${DATASET}" \
        --ram "${ram}" \
        --metric "${METRIC}" \
        --root-data-dir "${DATADIR}" \
        --output-dir "${OUTPUTDIR}"  \
        --resolution-min "${RESOLUTIONMIN}" \
        --resolution-max "${RESOLUTIONMAX}" \
        --bleach-min "${BLEACHMIN}" \
        --bleach-max "${BLEACHMAX}" \
        --cpus "${CPUS}" \
        --budget "${BUDGET}" &
done

echo "Waiting for all processes to finish..."
wait

echo "Done!"
