# sequence sampling for 1D timeseries modeling


from numba.core.typing.builtins import Print
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
# import tisc
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, sampling_rate, sequence_length, segments , expected_dim):
        self.sampling_rate = sampling_rate
        self.sequence_length = sequence_length
        self.total_samples = sampling_rate * sequence_length
        self.segments_number = segments
        self.expected_dim = expected_dim

        class_list = sorted(os.listdir(data_dir))
        try:
            class_list.remove('Cleaning the kitchen')
            class_list.remove('Misc')
            class_list.remove('Organizing the kitchen')
        except Exception as e:
            print(e)

        self.classes = class_list

        self.data, self.labels = self.load_data(data_dir)

    def segment_sampling(self, sequence, mode="uniform"):
        """
        :param sequence: gets original sequence from a csv file (rows x columns)
        :param mode: specifies the sampling method, currently supports "random"
        :return: sub_sampled sequence from the input sequence'''
        """

        n_rows = sequence.shape[0]
        segment_length = n_rows // self.segments_number
        sub_sequence = []

        for i in range(self.segments_number):
            start_idx = i * segment_length
            # print(f'start index {start_idx} of segment {i+1}')
            end_idx = (i + 1) * segment_length if i < self.segments_number - 1 else n_rows
            segment = sequence[start_idx:end_idx, :]

            if mode == "random":
                if len(segment) > 0:
                    sampled_idx = np.random.choice(segment.shape[0])
                    sub_sequence.append(segment[sampled_idx, :])

        sub_sequence = np.array(sub_sequence)

        # If the sampled sequence length is less than required, pad with the last row
        if len(sub_sequence) < self.total_samples:
            padding = self.total_samples - len(sub_sequence)
            if len(sub_sequence) > 0:
                padding_row = np.tile(sub_sequence[-1], (padding, 1))
                sub_sequence = np.vstack([sub_sequence, padding_row])
            else:
                sub_sequence = np.zeros((self.total_samples, sequence.shape[1]))

        return sub_sequence[:self.total_samples, :]

    def load_data(self, directory):
        data = []
        labels = []
        print(directory)
        for i, class_folder in enumerate(self.classes):
            class_path = os.path.join(directory, class_folder)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                csv_data = pd.read_csv(file_path)
                csv_col = csv_data.columns[1:]
                # print(f'data columns {csv_col}')
                sequence = csv_data[csv_col].values
                # print(f"sequence shape {sequence.shape}")
                sampled_sequence = self.segment_sampling(sequence)
                # print(f'sampled sequence length and shape' , sampled_sequence.shape)
                if sampled_sequence.shape[1] != self.expected_dim:
                    print(f"Skipping file {file_name} due to incorrect dimensions: {sampled_sequence.shape}")
                    continue
                data.append(sampled_sequence)
                # label is the index of the class_folder
                labels.append(i)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Data_with_identifier(Dataset):
    def __init__(self, data_dir, sampling_rate, sequence_length, segments, expected_dim):
        self.sampling_rate = sampling_rate
        self.sequence_length = sequence_length
        self.total_samples = sampling_rate * sequence_length
        self.segments_number = segments
        self.expected_dim = expected_dim

        class_list = sorted(os.listdir(data_dir))
        try:
            class_list.remove('Cleaning the kitchen')
            class_list.remove('Misc')
            class_list.remove('Organizing the kitchen')
        except Exception as e:
            print(e)

        self.classes = class_list

        self.data, self.labels, self.id, self.ss = self.load_data(data_dir)

    def segment_sampling(self, sequence, mode="uniform"):
        """
        :param sequence: gets original sequence from a csv file (rows x columns)
        :param mode: specifies the sampling method, currently supports "random" and "uniform"
        :return: sub_sampled sequence from the input sequence'''
        """

        n_rows = sequence.shape[0]
        segment_length = n_rows // self.segments_number
        sub_sequence = []

        for i in range(self.segments_number):
            start_idx = i * segment_length
            # print(f'start index {start_idx} of segment {i+1}')
            end_idx = (i + 1) * segment_length if i < self.segments_number - 1 else n_rows
            segment = sequence[start_idx:end_idx, :]

            if mode == "random":
                if len(segment) > 0:
                    sampled_idx = np.random.choice(segment.shape[0])
                    sub_sequence.append(segment[sampled_idx, :])
            elif self.mode == "uniform":
                if len(segment) > 0:
                    # Uniformly sample points from the segment
                    num_samples = self.total_samples // self.segments_number
                    indices = np.linspace(0, len(segment) - 1, num_samples, dtype=int)
                    sub_sequence.extend(segment[indices, :])

        sub_sequence = np.array(sub_sequence)

        # If the sampled sequence length is less than required, pad with the last row
        if len(sub_sequence) < self.total_samples:
            padding = self.total_samples - len(sub_sequence)
            if len(sub_sequence) > 0:
                padding_row = np.tile(sub_sequence[-1], (padding, 1))
                sub_sequence = np.vstack([sub_sequence, padding_row])
            else:
                sub_sequence = np.zeros((self.total_samples, sequence.shape[1]))

        return sub_sequence[:self.total_samples, :]

    def load_data(self, directory):
        data = []
        labels = []
        sub_id = []
        session = []
        print(directory)
        for i, class_folder in enumerate(self.classes):
            class_path = os.path.join(directory, class_folder)
            for file_name in os.listdir(class_path):
                #                 print(file_name)
                s_id, ss = file_name.split(".")[0].split("_")
                #                 print(f"s_id: {s_id} , ss: {ss}")
                file_path = os.path.join(class_path, file_name)
                csv_data = pd.read_csv(file_path)
                csv_col = csv_data.columns[1:]
                # print(f'data columns {csv_col}')
                sequence = csv_data[csv_col].values
                # print(f"sequence shape {sequence.shape}")
                sampled_sequence = self.segment_sampling(sequence)
                # print(f'sampled sequence length and shape' , sampled_sequence.shape)
                if sampled_sequence.shape[1] != self.expected_dim:
                    print(f"Skipping file {file_name} due to incorrect dimensions: {sampled_sequence.shape}")
                    continue
                data.append(sampled_sequence)
                # label is the index of the class_folder
                labels.append(i)
                sub_id.append(s_id)
                session.append(ss)

        return data, labels, sub_id, session

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.id[idx], self.ss[idx]



