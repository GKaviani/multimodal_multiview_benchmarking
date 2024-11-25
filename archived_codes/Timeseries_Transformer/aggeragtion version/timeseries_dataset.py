import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, directory, mode='train', sampling_rate=400, include_classes=None, max_clips_per_sample=2, clip_length=30):
        self.data_dir = os.path.join(directory, mode)
        self.sampling_rate = sampling_rate
        self.clip_length = clip_length * self.sampling_rate  # length in seconds
        self.max_clips_per_sample = max_clips_per_sample

        if include_classes is not None:
            self.classes = [cls for cls in os.listdir(self.data_dir) if cls in include_classes and os.path.isdir(os.path.join(self.data_dir, cls))]
        else:
            self.classes = [cls for cls in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, cls))]

        self.files = [(os.path.join(self.data_dir, activity, file), activity) for activity in self.classes for file in os.listdir(os.path.join(self.data_dir, activity))]
        print(f'{5*"*"} {len(self.files)} files in {self.data_dir} directory {5*"*"}')
        self.classes_name = sorted(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath, activity = self.files[index]
        data = pd.read_csv(filepath)
        # print(f"data columns {data.columns}")
        # values = data['value'].values
        data.columns = data.columns.str.strip() #remove whitespaces
        # Select the first 6 columns for the 6D data input
        values = data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
        # values = data[["ECG", "Heart_Rate", "Respiration", "Respiration_Rate", "R-R_Interval"]].values
        # try:
        #     values = data[["EMG_1","EMG_2"]].values
        # except Exception as e:
        #     print(f"Catch exception at loading data {e}")


        total_length = len(values)
        # max_possible_index = total_length - self.clip_length

        max_possible_index = max(1, total_length - self.clip_length)

        # Adjust the number of subsequences if necessary
        num_subsequences = min(self.max_clips_per_sample, max_possible_index)

        if max_possible_index > 0:
            # start_indices = np.random.choice(range(max_possible_index), self.max_clips_per_sample, replace=False)
            start_indices = np.random.choice(range(max_possible_index), num_subsequences, replace=False)
        else:
            # start_indices = [0] * self.max_clips_per_sample
            start_indices = [0] * num_subsequences
        clips=[]
        for start in start_indices:
            end = start + self.clip_length
            clip = values[start:end]
            if len(clip) < self.clip_length:
                pad_size = self.clip_length - len(clip)
                clip = np.pad(clip, ((0, pad_size), (0, 0)), 'constant', constant_values=0)
            clips.append(torch.tensor(clip, dtype=torch.float))

            # Pad the number of clips to max_clips_per_sample
        if len(clips) < self.max_clips_per_sample:
            padding_clip = torch.zeros((self.clip_length, values.shape[1]), dtype=torch.float)
            clips += [padding_clip] * (self.max_clips_per_sample - len(clips))

        clips = torch.stack(clips)  # Shape: [num_clips, clip_length]
        label = self.classes_name.index(activity)

        return clips, label, index  # Include the index to track the original sample
