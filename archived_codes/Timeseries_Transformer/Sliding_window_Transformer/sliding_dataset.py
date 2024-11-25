import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset , DataLoader
import torch
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    def __init__(self, directory, mode='train', sampling_rate=400, include_classes=None, subseq_length=4, window_stride=2 , num_subseq = 2):
        self.data_dir = os.path.join(directory, mode)
        self.sampling_rate = sampling_rate
        self.sub_sequence_length = subseq_length * self.sampling_rate  # length in seconds
        self.window_stride = window_stride * self.sampling_rate  # stride in seconds
        self.num_subseq = num_subseq

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
        data.columns = data.columns.str.strip()  # remove whitespaces
        # values = data[["ECG", "Heart_Rate", "Respiration", "Respiration_Rate", "R-R_Interval"]].values
        # values = data[["ECG", "Heart_Rate", "Respiration", "Respiration_Rate"]].values
        # values = data[["ECG", "Heart_Rate"]].values
        values = data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values

        total_length = len(values)
        sub_sequences = []
        position_embeddings = []

        # print(f"Processing file: {filepath}, total_length: {total_length}")
        if total_length < self.sub_sequence_length:
            # Take whatever data is available and pad it to the required sub-sequence length
            clip = values
            pad_size = self.sub_sequence_length - len(clip)
            clip = np.pad(clip, ((0, pad_size), (0, 0)), 'constant', constant_values=0)
            sub_sequences.append(torch.tensor(clip, dtype=torch.float))
            position_embeddings.append(torch.full((self.sub_sequence_length,), 0, dtype=torch.float))
        else:
            for start in range(0, total_length - self.sub_sequence_length + 1, self.window_stride):
                if len(sub_sequences) < self.num_subseq:
                    end = start + self.sub_sequence_length
                    # print(start , end)
                    clip = values[start:end]
                    if len(clip) < self.sub_sequence_length:
                        pad_size = self.sub_sequence_length - len(clip)
                        clip = np.pad(clip, ((0, pad_size), (0, 0)), 'constant', constant_values=0)
                    sub_sequences.append(torch.tensor(clip, dtype=torch.float))
                    position_embeddings.append(
                        torch.full((self.sub_sequence_length,), start / self.sub_sequence_length,
                                   dtype=torch.float))  # Positional embedding
                else:
                    break
        while len(sub_sequences) < self.num_subseq:
            sub_sequences.append(sub_sequences[-1])  # Repeat the last subsequence if not enough data
            position_embeddings.append(position_embeddings[-1])
        # print("len(sub_sequences):\t",len(sub_sequences))
        sub_sequences = torch.stack(sub_sequences)
        # print(sub_sequences.shape)
        position_embeddings = torch.stack(position_embeddings)

        return sub_sequences, position_embeddings, self.classes_name.index(activity)

    def get_labels(self):
        return [self.classes_name.index(activity) for _, activity in self.files]




if __name__ == "__main__":
# Example usage:
    data_dir = '/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/EMG_Modality_Top/BioMonitor/Combined'
    dataset = TimeSeriesDataset(directory=data_dir, mode='train', sampling_rate=400, include_classes=None, subseq_length=4, window_stride=2 , num_subseq=2)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True )
    print(len(dataloader))
    for x , _ , y in dataloader:
        print(x.shape , y.shape)
        # print(x[0].shape , "\n" , x[0])
        for sub in x[0]:
            time = torch.linspace(0, 4, steps=1600)  # Time from 0 to 4 seconds
            # Plotting
            plt.figure(figsize=(10, 4))
            plt.plot(time, sub[:,3], color='b')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Data Value')
            plt.grid(True)
            plt.savefig(f"/home/ghazal/Activity_Recognition_benchmarking/Timeseries_Transformer/Sliding_window_Transformer/sequence_plots/{data_dir.split('/')[-1]}_{dataset.sub_sequence_length}_{dataset.num_subseq}_{y[0]}.png")
        break