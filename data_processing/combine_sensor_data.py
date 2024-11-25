import pandas as pd
import os

# Base directory where sensor folders are located
# base_dir = '/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/EMG_Modality_Top/Insole-LT'

# List of sensor modalities
sensors = ["Insole-Arch" ,"Insole-Hallux",  "Insole-Heel_Lateral",  "Insole-Heel_Medial" ,"Insole-Met_1"  ,"Insole-Met_3",  "Insole-Met_5",  "Insole-Toes"]

# Define the split you are interested in: 'train', 'validation', or 'test'
split = 'train'

def combine_sensor_data_per_subject_session(base_dir, sensors, split):
    # Path for each sensor within the specified split
    sensor_paths = {sensor: os.path.join(base_dir, sensor, split) for sensor in sensors}

    # Dictionary to hold dataframes for each subject-session combination
    combined_data_dict = {}

    # Iterate over each sensor
    for sensor, path in sensor_paths.items():
        # Iterate over activity class folders within the sensor directory
        for activity_folder in os.listdir(path):
            activity_dir = os.path.join(path, activity_folder)

            # Process each CSV file within the activity folder
            for file_name in os.listdir(activity_dir):
                # Identify subject and session from the filename
                subject_session = file_name.split('.')[0]

                # Construct the unique key for the dictionary (activity, subject, session)
                unique_key = (activity_folder, subject_session)

                # Read the CSV file
                file_path = os.path.join(activity_dir, file_name)
                data = pd.read_csv(file_path, usecols=['utc_time', 'time', 'value'])

                # Rename 'value' to the sensor name
                data.rename(columns={'value': sensor}, inplace=True)

                # Keep only relevant columns for merging to avoid column duplication
                data = data[['utc_time', sensor]]

                # If this is the first sensor for this subject_session, initialize the dataframe
                if unique_key not in combined_data_dict:
                    combined_data_dict[unique_key] = data.set_index('utc_time')
                else:
                    # Otherwise, merge this data with the existing dataframe for this subject_session
                    combined_data_dict[unique_key] = combined_data_dict[unique_key].merge(data.set_index('utc_time'), left_index=True, right_index=True, how='outer')

    # Output combined data to new CSV files
    for key, dataframe in combined_data_dict.items():
        activity, subject_session = key
        output_dir = os.path.join(base_dir, '8_point_pressure', split, activity)
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f'{subject_session}.csv')
        dataframe.reset_index().to_csv(output_file_path, index=False)

combine_sensor_data_per_subject_session(base_dir, sensors, split)
combine_sensor_data_per_subject_session(base_dir, sensors, "validation")
combine_sensor_data_per_subject_session(base_dir, sensors, "test")