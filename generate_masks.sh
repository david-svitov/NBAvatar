#!/bin/bash
# Usage: bash generate_masks.sh /path/to/DecafDataset

if [ -z "$1" ]; then
    echo "Usage: bash generate_masks.sh /path/to/DecafDataset"
    exit 1
fi
DATASET_PATH=$1

for SUBJECT in S1 S2 S3 S4; do
    python src/scripts/get_sam_masks.py --dataset_path=$DATASET_PATH/test/frames/$SUBJECT
    python src/scripts/get_sam_masks.py --dataset_path=$DATASET_PATH/train/frames/$SUBJECT
    python src/scripts/preprocess_sam_masks.py --dataset_path=$DATASET_PATH/test/frames/$SUBJECT
    python src/scripts/preprocess_sam_masks.py --dataset_path=$DATASET_PATH/train/frames/$SUBJECT
done
