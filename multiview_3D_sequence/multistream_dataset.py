import os
import random
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset , DataLoader
import logging
import matplotlib.pyplot as plt

class Custom3DDataset(Dataset):
    def __init__(self, root_dir, include_classes, sequence_length=16, sampling="single-random", transform=None,
                 max_seq=2):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.sampling = sampling
        self.number_of_seq = max_seq

        if len(include_classes) > 0:
            self.classes = [cls for cls in os.listdir(os.path.join(root_dir, 'cam_1')) if cls in include_classes]
        else:
            self.classes = os.listdir(os.path.join(root_dir, 'cam_1'))

        self.class_names = sorted(self.classes)
        self.sequences = self._create_sequences()

    # def _create_sequences(self):
    #     sequences = []
    #     unique_id = -1
    #     for activity in self.classes:
    #         activity_dir_cam1 = os.path.join(self.root_dir, 'cam_1', activity)
    #         activity_dir_cam2 = os.path.join(self.root_dir, 'cam_2', activity)
    #
    #         frames_cam1 = sorted(glob(os.path.join(activity_dir_cam1, '*.png')))
    #         frames_cam2 = sorted(glob(os.path.join(activity_dir_cam2, '*.png')))
    #
    #         grouped_frames_cam1 = self._group_frames_by_subject_and_session(frames_cam1)
    #         grouped_frames_cam2 = self._group_frames_by_subject_and_session(frames_cam2)
    #
    #         for subject_session in grouped_frames_cam1.keys():
    #             frames1 = grouped_frames_cam1.get(subject_session, [])
    #             frames2 = grouped_frames_cam2.get(subject_session, [])
    #
    #             if len(frames1) != len(frames2):
    #                 # raise ValueError(f"Frame count mismatch between cam_1 and cam_2 for {subject_session}")
    #                 logging.warning(f"Frame count mismatch between cam_1 and cam_2 for {subject_session}. Skipping.")
    #                 continue
    #
    #             unique_id += 1
    #             if len(frames1) < self.sequence_length:
    #                 sequence1 = self._pad_sequence(frames1)
    #                 sequence2 = self._pad_sequence(frames2)
    #                 sequences.append((sequence1, sequence2, activity, unique_id))
    #             else:
    #                 if self.sampling == "multiple-consecutive":
    #                     seq_counter = 0
    #                     for i in range(0, len(frames1) - self.sequence_length + 1):
    #                         while seq_counter < self.number_of_seq:
    #                             sequence1 = frames1[i:i + self.sequence_length]
    #                             sequence2 = frames2[i:i + self.sequence_length]
    #                             sequences.append((sequence1, sequence2, activity, unique_id))
    #                             seq_counter += 1
    #                 elif self.sampling == "multiple-random":
    #                     seq_counter = 0
    #                     for _ in range(0, len(frames1) - self.sequence_length + 1):
    #                         while seq_counter < self.number_of_seq:
    #                             start_idx = random.randint(0, len(frames1) - self.sequence_length)
    #                             sequence1 = frames1[start_idx:start_idx + self.sequence_length]
    #                             sequence2 = frames2[start_idx:start_idx + self.sequence_length]
    #                             sequences.append((sequence1, sequence2, activity, unique_id))
    #                             seq_counter += 1
    #                 elif self.sampling == "single-random":
    #                     seq_counter = 0
    #                     while seq_counter < self.number_of_seq:
    #                         sequence1 = sorted(random.sample(frames1, self.sequence_length))
    #                         sequence2 = sorted(random.sample(frames2, self.sequence_length))
    #                         sequences.append((sequence1, sequence2, activity, unique_id))
    #                         seq_counter += 1
    #     return sequences

    # truncate
    def _create_sequences(self):
        sequences = []
        unique_id = -1
        for activity in self.classes:
            activity_dir_cam1 = os.path.join(self.root_dir, 'cam_1', activity)
            activity_dir_cam2 = os.path.join(self.root_dir, 'cam_2', activity)

            frames_cam1 = sorted(glob(os.path.join(activity_dir_cam1, '*.png')))
            frames_cam2 = sorted(glob(os.path.join(activity_dir_cam2, '*.png')))

            grouped_frames_cam1 = self._group_frames_by_subject_and_session(frames_cam1)
            grouped_frames_cam2 = self._group_frames_by_subject_and_session(frames_cam2)

            for subject_session in grouped_frames_cam1.keys():
                frames1 = grouped_frames_cam1.get(subject_session, [])
                frames2 = grouped_frames_cam2.get(subject_session, [])

                if len(frames1) != len(frames2):
                    min_length = min(len(frames1), len(frames2))
                    frames1 = frames1[:min_length]
                    frames2 = frames2[:min_length]

                unique_id += 1
                if len(frames1) < self.sequence_length:
                    sequence1 = self._pad_sequence(frames1)
                    sequence2 = self._pad_sequence(frames2)
                    sequences.append((sequence1, sequence2, activity, unique_id))
                else:
                    if self.sampling == "multiple-consecutive":
                        seq_counter = 0
                        for i in range(0, len(frames1) - self.sequence_length + 1):
                            while seq_counter < self.number_of_seq:
                                sequence1 = frames1[i:i + self.sequence_length]
                                sequence2 = frames2[i:i + self.sequence_length]
                                sequences.append((sequence1, sequence2, activity, unique_id))
                                seq_counter += 1
                    elif self.sampling == "multiple-random":
                        seq_counter = 0
                        for _ in range(0, len(frames1) - self.sequence_length + 1):
                            while seq_counter < self.number_of_seq:
                                start_idx = random.randint(0, len(frames1) - self.sequence_length)
                                sequence1 = frames1[start_idx:start_idx + self.sequence_length]
                                sequence2 = frames2[start_idx:start_idx + self.sequence_length]
                                sequences.append((sequence1, sequence2, activity, unique_id))
                                seq_counter += 1
                    elif self.sampling == "single-random":
                        seq_counter = 0
                        while seq_counter < self.number_of_seq:
                            sequence1 = sorted(random.sample(frames1, self.sequence_length))
                            sequence2 = sorted(random.sample(frames2, self.sequence_length))
                            sequences.append((sequence1, sequence2, activity, unique_id))
                            seq_counter += 1
        return sequences

    def _pad_sequence(self, frames):
        if len(frames) == 0:
            return frames
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])
        return frames

    def _group_frames_by_subject_and_session(self, frames):
        grouped_frames = {}
        for frame in frames:
            filename = os.path.basename(frame)
            subject_session = '_'.join(filename.split('_')[:2])
            if subject_session not in grouped_frames:
                grouped_frames[subject_session] = []
            grouped_frames[subject_session].append(frame)
        return grouped_frames

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence1, sequence2, activity, sample_id = self.sequences[idx]
        frames1, frames2 = [], []

        for frame_path in sequence1:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames1.append(image)

        for frame_path in sequence2:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames2.append(image)

        frames1 = torch.stack(frames1).permute(1, 0, 2, 3)  # (C, T, H, W)
        frames2 = torch.stack(frames2).permute(1, 0, 2, 3)  # (C, T, H, W)

        label = self._get_label(activity)
        return frames1, frames2, label, sample_id

    def _get_label(self, activity):
        class_names = self.class_names
        label = class_names.index(activity)
        return label

if __name__ == "__main__":
    import torchvision.transforms as transforms

    # Define any data augmentations or transformations
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    # Instantiate the dataset
    dataset = Custom3DDataset(root_dir='/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/train', include_classes=[], sequence_length=16, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Iterate through the dataset and visualize some sequences
    for batch_idx, (frames1, frames2, labels, sample_ids) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Frames1 shape: {frames1.shape}")
        print(f"Frames2 shape: {frames2.shape}")
        print(f"Labels: {labels}")
        print(f"Sample IDs: {sample_ids}")

        # Visualize a sequence from each camera view
        fig, axs = plt.subplots(2, frames1.shape[2], figsize=(15, 5))

        for t in range(frames1.shape[2]):
            axs[0, t].imshow(frames1[0, :, t, :, :].permute(1, 2, 0).numpy())
            axs[0, t].axis('off')
            axs[0, t].set_title(f"Cam 1 Frame {t}")

            axs[1, t].imshow(frames2[0, :, t, :, :].permute(1, 2, 0).numpy())
            axs[1, t].axis('off')
            axs[1, t].set_title(f"Cam 2 Frame {t}")
        plt.savefig("./test_dataset_multi_view.png")
        plt.show()

        # Only visualize the first batch
        break