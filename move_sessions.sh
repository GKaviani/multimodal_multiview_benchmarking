##!/bin/bash
#
## Define source and destination directories
#source_dir="/mnt/data-tmp/ghazal/DARai_DATA/L1_Activities/"
#dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/benchmark_uni_modal_rgb_clips"
# Loop through each activity directory in the source
#for activity_dir in "$source_dir"/*; do
#    activity_name=$(basename "$activity_dir")
#    mkdir -p "$dest_dir/$activity_name"
#    find $activity_dir/RGB/camera_1_fps_15 -type d -name 'session*' -exec rsync -az --progress {} "$dest_dir/$activity_name" \;
#    done

##/mnt/data-tmp/ghazal/DARai_DATA/benchmark_uni_modal_rgb_clips/camera_2_fps_15
#find "/mnt/data-tmp/ghazal/DARai_DATA/benchmark_uni_modal_rgb_clips/camera_2_fps_15" -type f -name '*.png' -exec sh -c '
#    for file_path; do
#        parent_dir=$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$file_path")")")")")
##        echo "$file_path" "$parent_dir"
#        mv -v "$file_path" "$parent_dir"
#    done
#' sh {} +

#source_dir="/mnt/data-tmp/ghazal/DARai_DATA/L1_Activities/"
#dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/train/cam_1"
#for activity_dir in "$source_dir"/*; do
#    activity_name=$(basename "$activity_dir")
##    mkdir -p "$dest_dir/$activity_name"
#    find "$activity_dir/RGB/camera_1_fps_15" -type f -name '*.png' -exec rsync -avznh --ignore-existing {} "$dest_dir/$activity_name" \;
#    done

#find "/mnt/data-tmp/ghazal/DARai_DATA/benchmark_uni_modal_rgb_clips/camera_1_fps_15" -type f -name '*.png' -exec sh -c '
#    for file_path; do
#        parent_dir=$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$file_path")")")")")
#        echo "$file_path" "$parent_dir"
#        #rsync -avz --ignore-existing "$file_path" "$parent_dir"
#    done
#' sh {} +

#source_dir="/mnt/data-tmp/ghazal/DARai_DATA/L1_Activities/"
#test_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/test/cam_1"
#validation_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/validation/cam_1"
#train_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/train/cam_1"
#
#for activity_dir in "$source_dir"/*; do
#    activity_name=$(basename "$activity_dir")
#
#    # Create destination directories
##    mkdir -p "$test_dest_dir/$activity_name"
##    mkdir -p "$validation_dest_dir/$activity_name"
##    mkdir -p "$train_dest_dir/$activity_name"
#
#    # Sync test patterns
#    find "$activity_dir/RGB/camera_1_fps_15" -type f \( -name '10_*.png' -o -name '16_*.png' -o -name '19_*.png' \) -exec rsync -azvh --ignore-existing {} "$test_dest_dir/$activity_name" \;
#
#    # Sync validation patterns
#    find "$activity_dir/RGB/camera_1_fps_15" -type f \( -name '02_*.png' -o -name '20_*.png' \) -exec rsync -azvh --ignore-existing {} "$validation_dest_dir/$activity_name" \;
#
#    # Sync training patterns (everything else)
#    find "$activity_dir/RGB/camera_1_fps_15" -type f -not \( -name '10_*.png' -o -name '16_*.png' -o -name '19_*.png' -o -name '02_*.png' -o -name '20_*.png' \) -exec rsync -azvh --ignore-existing {} "$train_dest_dir/$activity_name" \;
#done
#source_dir="/mnt/data-tmp/ghazal/DARai_DATA/L1_Activities/"
#test_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/depth_dataset/test/depth_1"
#validation_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/depth_dataset/validation/depth_1"
#train_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/depth_dataset/train/depth_1"
#
#for activity_dir in "$source_dir"/*; do
#    activity_name=$(basename "$activity_dir")
#
#    # Create destination directories
#    mkdir -p "$test_dest_dir/$activity_name"
#    mkdir -p "$validation_dest_dir/$activity_name"
#    mkdir -p "$train_dest_dir/$activity_name"
#
#    # Sync test patterns
#    find "$activity_dir/Depth/camera_1_fps_15" -type f \( -name '10_*.png' -o -name '16_*.png' -o -name '19_*.png' \) -exec rsync -azvh --ignore-existing {} "$test_dest_dir/$activity_name" \;
#
#    # Sync validation patterns
#    find "$activity_dir/Depth/camera_1_fps_15" -type f \( -name '02_*.png' -o -name '20_*.png' \) -exec rsync -azvh --ignore-existing {} "$validation_dest_dir/$activity_name" \;
#
#    # Sync training patterns (everything else)
#    find "$activity_dir/Depth/camera_1_fps_15" -type f -not \( -name '10_*.png' -o -name '16_*.png' -o -name '19_*.png' -o -name '02_*.png' -o -name '20_*.png' \) -exec rsync -azvh --ignore-existing {} "$train_dest_dir/$activity_name" \;
#done

source_dir="/mnt/data-tmp/ghazal/DARai_DATA/L1_Activities/"
test_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/depth_dataset/test/depth_2"
validation_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/depth_dataset/validation/depth_2"
train_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/depth_dataset/train/depth_2"

for activity_dir in "$source_dir"/*; do
    activity_name=$(basename "$activity_dir")

    # Create destination directories
    mkdir -p "$test_dest_dir/$activity_name"
    mkdir -p "$validation_dest_dir/$activity_name"
    mkdir -p "$train_dest_dir/$activity_name"

#    # Sync test patterns
    find "$activity_dir/Depth/camera_2_fps_15" -type f \( -name '10_*.png' -o -name '16_*.png' -o -name '19_*.png' \) -exec rsync -azvh --ignore-existing {} "$test_dest_dir/$activity_name" \;

    # Sync validation patterns
    find "$activity_dir/Depth/camera_2_fps_15" -type f \( -name '02_*.png' -o -name '20_*.png' \) -exec rsync -azvh --ignore-existing {} "$validation_dest_dir/$activity_name" \;

    # Sync training patterns (everything else)
    find "$activity_dir/Depth/camera_2_fps_15" -type f -not \( -name '10_*.png' -o -name '16_*.png' -o -name '19_*.png' -o -name '02_*.png' -o -name '20_*.png' \) -exec rsync -azvh --ignore-existing {} "$train_dest_dir/$activity_name" \;
done