
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import plot_sequence , check_label_distribution


class TimeSeriesDataset(Dataset):
    def __init__(self, directory, mode='train', sampling_rate=400, include_classes=None, max_subseq=5, subseq_length=3):
        self.data_dir = os.path.join(directory, mode)
        self.sampling_rate = sampling_rate
        self.sub_sequence_length = subseq_length * self.sampling_rate  # length in seconds
        self.max_subseq_per_sample = max_subseq

        if include_classes is not None:
            self.classes = [cls for cls in os.listdir(self.data_dir) if
                            cls in include_classes and os.path.isdir(os.path.join(self.data_dir, cls))]
        else:
            self.classes = [cls for cls in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, cls))]

        self.files = [(os.path.join(self.data_dir, activity, file), activity)
                      for activity in self.classes
                      for file in os.listdir(os.path.join(self.data_dir, activity))]
        self.classes_name = sorted(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath, activity = self.files[index]
        data = pd.read_csv(filepath)
        values = data['value'].values

        total_length = len(values)
        max_possible_index = total_length - self.sub_sequence_length

        if max_possible_index > 0:
            start_indices = np.random.choice(range(max_possible_index), self.max_subseq_per_sample, replace=False)
        else:
            start_indices = [0] * self.max_subseq_per_sample

        sub_sequences = []
        position_embeddings = []
        for start in start_indices:
            end = start + self.sub_sequence_length
            clip = values[start:end]
            if len(clip) < self.sub_sequence_length:
                clip = F.pad(torch.tensor(clip, dtype=torch.float), (0, self.sub_sequence_length - len(clip)),
                             'constant', 0)
            else:
                clip = torch.tensor(clip, dtype=torch.float)
            sub_sequences.append(clip)
            position_embeddings.append(
                torch.full((1,), start / self.sub_sequence_length, dtype=torch.float))  # Positional embedding

        return torch.stack(sub_sequences), torch.cat(position_embeddings), self.classes_name.index(activity)
    # def __getitem__(self, index):
    #     filepath, activity = self.files[index]
    #     data = pd.read_csv(filepath)
    #     data.columns = data.columns.str.strip() #remove whitespaces
    #     # Select the first 6 columns for the 6D data input
    #     # values = data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    #     # values = data[["ECG","Heart_Rate","Respiration","Respiration_Rate","R-R_Interval"]].values
    #
    #
    #     total_length = len(values)
    #     # print(total_length)
    #     # max_possible_index = total_length - self.sub_sequence_length
    #     #
    #     # if max_possible_index > 0:
    #     #     start_indices = np.random.choice(range(max_possible_index), self.max_subseq_per_sample, replace=False)
    #     # else:
    #     #     start_indices = [0] * self.max_subseq_per_sample
    #     max_possible_index = max(1, total_length - self.sub_sequence_length)
    #
    #     # Adjust the number of subsequences if necessary
    #     num_subsequences = min(self.max_subseq_per_sample, max_possible_index)
    #
    #     if max_possible_index > 0:
    #         # start_indices = np.random.choice(range(max_possible_index), self.max_clips_per_sample, replace=False)
    #         start_indices = np.random.choice(range(max_possible_index), num_subsequences, replace=False)
    #     else:
    #         # start_indices = [0] * self.max_clips_per_sample
    #         start_indices = [0] * num_subsequences
    #
    #     sub_sequences = []
    #     position_embeddings = []
    #     for start in start_indices:
    #         end = start + self.sub_sequence_length
    #         clip = values[start:end]
    #         if len(clip) < self.sub_sequence_length:
    #             pad_size = self.sub_sequence_length - len(clip)
    #             clip = np.pad(clip, ((0, pad_size), (0, 0)), 'constant', constant_values=0)
    #         sub_sequences.append(torch.tensor(clip, dtype=torch.float))
    #         position_embeddings.append(
    #             torch.full((self.sub_sequence_length,), start / self.sub_sequence_length, dtype=torch.float))  # Positional embedding
    #
    #     # Pad the number of clips to max_subseq_per_sample
    #     if len(sub_sequences) < self.max_subseq_per_sample:
    #         padding_clip = torch.zeros((self.sub_sequence_length, values.shape[1]), dtype=torch.float)
    #         padding_position = torch.full((self.sub_sequence_length,), -1, dtype=torch.float)  # -1 indicates padding
    #         sub_sequences += [padding_clip] * (self.max_subseq_per_sample - len(sub_sequences))
    #         position_embeddings += [padding_position] * (self.max_subseq_per_sample - len(position_embeddings))
    #
    #
    #
    #     return torch.stack(sub_sequences), torch.cat(position_embeddings), self.classes_name.index(activity)

    def get_labels(self):
        return [self.classes_name.index(activity) for _, activity in self.files]

# Example usage
if __name__ == "__main__":
    dataset = TimeSeriesDataset('/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/IMU/IMU_LeftArm', sampling_rate=12,
                                include_classes=None, max_subseq=1 )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(loader))
    # print(type(dataset.classes))
    # Assuming you have instantiated your dataset as `train_dataset`
    check_label_distribution(dataset)
    # for sequences,_, labels in loader:
    #     print(sequences.shape, labels.shape)
    #     print(sequences, labels)
    #     seq_label = dataset.classes_name[labels[0].item()]
    #     plot_sequence(sequences[0, 0, :], dataset.sampling_rate , seq_label)
    #     break
