##!/bin/bash
#
## Define source and destination directories
#source_dir="/mnt/data-tmp/ghazal/DARai_DATA/RGB_sd/"
#destination_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset_sd"
#
## Dry run flag (set to true for dry run)
#dryrun=false
#
## Iterate over the source directory
#find "$source_dir" -mindepth 3 -type f | while read -r file; do
#    # Extract folder names 'a' and 'b' from the path
#    a=$(basename "$(dirname "$(dirname "$file")")")
#    b=$(basename "$(dirname "$file")")
#
#    # Create the new destination directory structure
#    dest_dir="$destination_dir/$b/$a"
#
#    # If dry run, just print the actions
#    if [ "$dryrun" = true ]; then
#        echo "Dry run: Would copy '$file' to '$dest_dir'"
#    else
#        # Create the destination directory if it doesn't exist
#        mkdir -p "$dest_dir"
#
#        # Copy the file to the new destination
#        cp -v "$file" "$dest_dir"
#    fi
#done

source_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset_sd/cam_1"
test_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset_sd/test/cam_1"
validation_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset_sd/validation/cam_1"
train_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset_sd/train/cam_1"

for activity_dir in "$source_dir"/*; do
    activity_name=$(basename "$activity_dir")

    # Create destination directories
    mkdir -p "$test_dest_dir/$activity_name"
    mkdir -p "$validation_dest_dir/$activity_name"
    mkdir -p "$train_dest_dir/$activity_name"

    # Sync test patterns
    find "$activity_dir" -type f \( -name '10_*.jpg' -o -name '16_*.jpg' -o -name '19_*.jpg' \) -exec rsync -azvh --ignore-existing {} "$test_dest_dir/$activity_name" \;

    # Sync validation patterns
    find "$activity_dir" -type f \( -name '02_*.jpg' -o -name '20_*.jpg' \) -exec rsync -azvh --ignore-existing {} "$validation_dest_dir/$activity_name" \;

    # Sync training patterns (everything else)
    find "$activity_dir" -type f -not \( -name '10_*.jpg' -o -name '16_*.jpg' -o -name '19_*.jpg' -o -name '02_*.jpg' -o -name '20_*.jpg' \) -exec rsync -azvh --ignore-existing {} "$train_dest_dir/$activity_name" \;
done