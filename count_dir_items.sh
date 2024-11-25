base_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/train/cam_1/*"
echo "In $base_dir"
for dir in $base_dir; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" | wc -l)
        echo "$dir: $count"
    fi
done

#base_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/test/cam_1/*"
#echo "In $base_dir"
#for dir in $base_dir; do
#    if [ -d "$dir" ]; then
#        count=$(ls -1 "$dir" | wc -l)
#        echo "$dir: $count"
#    fi
#done
#
#base_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/validation/cam_1/*"
#echo "In $base_dir"
#for dir in $base_dir; do
#    if [ -d "$dir" ]; then
#        count=$(ls -1 "$dir" | wc -l)
#        echo "$dir: $count"
#    fi
#done

#
#
#base_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/validation/cam_2/*"
#echo "In $base_dir"
#for dir in $base_dir; do
#    if [ -d "$dir" ]; then
#        count=$(ls -1 "$dir" | wc -l)
#        echo "$dir: $count"
#    fi
#done
#base_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/test/cam_2/*"
#echo "In $base_dir"
#for dir in $base_dir; do
#    if [ -d "$dir" ]; then
#        count=$(ls -1 "$dir" | wc -l)
#        echo "$dir: $count"
#    fi
#done
base_dir="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/train/cam_2/*"
echo "In $base_dir"
for dir in $base_dir; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" | wc -l)
        echo "$dir: $count"
    fi
done
