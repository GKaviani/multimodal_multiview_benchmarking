#!/bin/bash

# Define base paths
BASE_PATH="/mnt/data-tmp/ghazal/DARai_DATA/benchmark_uni_modal_rgb_clips/camera_2_fps_15"
TRAIN_PATH="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/train/cam_2"
TEST_PATH="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/test/cam_2"

# Define the subject IDs for the test set
TEST_SUBJECTS=("10" "16" "19")

# Create the activity folders in train and test directories
for activity in "$BASE_PATH"/*; do
    if [ -d "$activity" ]; then
        activity_name=$(basename "$activity")
        mkdir -p "$TRAIN_PATH/$activity_name" "$TEST_PATH/$activity_name"
    fi
done
is_in_array() {
    local element="$1"
    shift
    for e; do
        if [[ "$e" == "$element" ]]; then
            return 0
        fi
    done
    return 1
}
# Move files to train and test directories
for activity in "$BASE_PATH"/*; do
    if [ -d "$activity" ]; then
        activity_name=$(basename "$activity")
        for file in "$activity"/*.png; do
            if [ -f "$file" ]; then
                file_name=$(basename "$file")
                subject_id=$(echo "$file_name" | cut -d'_' -f1)

                if is_in_array "$subject_id" "${TEST_SUBJECTS[@]}"; then
                    cp -v "$file" "$TEST_PATH/$activity_name/"
                else
                    cp -v "$file" "$TRAIN_PATH/$activity_name/"
                fi
            fi
        done
        echo "$file_name copied!"
    fi
done
