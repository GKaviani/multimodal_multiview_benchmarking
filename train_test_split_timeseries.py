import os
import shutil

# Base directory where the original data is stored
DATA_DIR = "/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/EMG_Modality_Top/EMG-Both"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VAL_DIR = os.path.join(DATA_DIR, "validation")



def split_data(DATA_DIR , TRAIN_DIR , TEST_DIR , VAL_DIR):
    # subject_id = None
    # Create train, test, and validation directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    # Define test and validation subject IDs
    TEST_SUBJECTS = ["10", "16", "19"]
    VAL_SUBJECTS = ["02", "20"]


    def copy_files(subjects, dest_dir):
        # Walk through the activity directories in the base data directory
        for activity in os.listdir(DATA_DIR):
            activity_path = os.path.join(DATA_DIR, activity)

            if os.path.isdir(activity_path):
                # Make corresponding activity directory in destination if not exists
                dest_activity_dir = os.path.join(dest_dir, activity)
                os.makedirs(dest_activity_dir, exist_ok=True)

                # Iterate through all files in the activity directory
                for file in os.listdir(activity_path):
                    file_path = os.path.join(activity_path, file)
                    if os.path.isfile(file_path):
                        subject_id = file.split('_')[0]  # Assumes subject ID is before the first underscore

                        # If the subject ID of the file is in the specified list, copy it
                        if subject_id in subjects:
                            dest_file_path = os.path.join(dest_activity_dir, file)
                            shutil.copy(file_path, dest_file_path)
                            print(f'{file_path} to {dest_file_path}')

    # Copy files to the respective directories
    copy_files(TEST_SUBJECTS, TEST_DIR)
    # copy_files(VAL_SUBJECTS, VAL_DIR)

    # Copy remaining files to the training directory
    for activity in os.listdir(DATA_DIR):
        activity_path = os.path.join(DATA_DIR, activity)

        if os.path.isdir(activity_path):
            train_activity_dir = os.path.join(TRAIN_DIR, activity)
            os.makedirs(train_activity_dir, exist_ok=True)

            for file in os.listdir(activity_path):
                src_file_path = os.path.join(activity_path, file)
                if os.path.isfile(src_file_path):
                    subject_id = file.split('_')[0]

                # If the file's subject ID is not in test or validation lists, copy to train
                if subject_id not in TEST_SUBJECTS: #and subject_id not in VAL_SUBJECTS:
                    dest_file_path = os.path.join(train_activity_dir, file)
                    shutil.copy(src_file_path, dest_file_path)
                    print(f'{src_file_path} to {dest_file_path}')

split_data(DATA_DIR ,TRAIN_DIR , TEST_DIR , VAL_DIR)


base_dir = "/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/EMG_Modality_Top/Insole-Both"

# for super_modality in os.listdir(base_dir):
#     if os.path.isdir(os.path.join(base_dir ,super_modality)):
#         for modality in os.listdir(os.path.join(base_dir ,super_modality)):
#             # print(base_dir , super_modality ,modality)
#             DATA_DIR = os.path.join(base_dir , super_modality ,modality)
#             TRAIN_DIR = os.path.join(DATA_DIR, "train")
#             TEST_DIR = os.path.join(DATA_DIR, "test")
#             VAL_DIR = os.path.join(DATA_DIR, "validation")
#             split_data(DATA_DIR ,TRAIN_DIR , TEST_DIR , VAL_DIR)
#
# for modality in os.listdir(base_dir):
#     print(os.listdir(base_dir))
#     # print(base_dir , super_modality ,modality)
#     DATA_DIR = os.path.join(base_dir ,modality)
#     TRAIN_DIR = os.path.join(DATA_DIR, "train")
#     TEST_DIR = os.path.join(DATA_DIR, "test")
#     VAL_DIR = os.path.join(DATA_DIR, "validation")
#     split_data(DATA_DIR ,TRAIN_DIR , TEST_DIR , VAL_DIR)