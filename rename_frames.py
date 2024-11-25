import os
import re
import argparse

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description="Rename files based on a specified pattern.")
parser.add_argument("directory_path", type=str, help="The path to the directory containing the files to be renamed.")
args = parser.parse_args()

# Regex pattern to match the filenames and extract parts
# pattern = r'Sid_(\d+)_ss_(\d+)_(\d+)\.png'
pattern = r'Sid_(\d+)_ss_(\d+)_(\d+)\_anonymized.png'
log_file_path = '/mnt/data-tmp/ghazal/DARai_DATA/rename_logfile/rename_logfile.txt'

if not os.path.exists('/mnt/data-tmp/ghazal/DARai_DATA/rename_logfile/'):
    os.makedirs('/mnt/data-tmp/ghazal/DARai_DATA/rename_logfile/')

# Open the log file in append mode
with open(log_file_path, 'a') as log_file:
    # Iterate through each directory and subdirectory
    for subdir, dirs, files in os.walk(args.directory_path):
        # Sort files to maintain the original order based on the frame number
        files.sort(key=lambda f: int(re.search(pattern, f).group(3)) if re.search(pattern, f) else 0)
        print(f"{len(files)} been found to rename in {args.directory_path}.")
        # Rename files
        for index, filename in enumerate(files):
            match = re.search(pattern, filename)
            # print(match)
            if match:
                new_filename = f"{match.group(1)}_{match.group(2)}_{index:05d}.png"
                old_file_path = os.path.join(subdir, filename)
                new_file_path = os.path.join(subdir, new_filename)
                os.rename(old_file_path, new_file_path)
                print(f"{old_file_path} renamed to {new_file_path}")
                log_file.write(f"Renamed '{filename}' to '{new_filename}'\n")
            else:
                log_file.write(f"{pattern} did not match with '{filename}'\n")
#Example_Usage:
#python rename_png.py /path/to/session*
#find ./ -type d -name "session*" -exec python "/home/ghazal/Activity_Recognition_benchmarking/rename_frames.py" {} \;

