#!/bin/bash

RUNID=$(date +%Y-%m-%d_%H-%M-%S)

for dset in "iris" "glass" "wine" "image_segmentation" "yeast" "vehicle" "breast_cancer" "rice" "segment" "dry_bean" "satimage" "letter" "optical_handwritten" "sepsis" "motion_sense"; do
    echo " ------------------- Running ${dset}... -------------------"
    ./tune_${dset}_sklearn.sh svm
    echo " ------------------- Finished ${dset}! -------------------"
    sleep 5
done

echo "---------------------------------"
echo "Finished running all experiments!"
echo "---------------------------------"