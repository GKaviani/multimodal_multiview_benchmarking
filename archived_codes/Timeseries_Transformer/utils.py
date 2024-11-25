import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_sequence(sequence, sampling_rate, title):
    """
    Args:
        sequence (torch.Tensor)
        sampling_rate (int)
        title (str)
    """
    # Convert sequence to numpy if it's a tensor
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.numpy()

    # Create a time axis based on the sampling rate and the length of the sequence
    time_axis = np.arange(0, len(sequence)) / sampling_rate

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, sequence, label='Subsequence')
    plt.title(f'One Random Subsequence Sample from class {title}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/ghazal/Activity_Recognition_benchmarking/Timeseries_Transformer/figs/{sampling_rate}")
    plt.show()
def custom_collate_fn(batch):
    """
    Collate function to handle batches of tensors with different shapes
    """
    sequences, labels = zip(*batch)
    padded_sequences = [pad_sequence(seqs, batch_first=True) for seqs in sequences]  # Pad per sample
    padded_sequences = torch.stack(padded_sequences)  # Stack to get shape [batch_size, num_subseq, seq_length]
    labels = torch.tensor(labels)
    return padded_sequences, labels

def plot_metrics(train_loss, val_loss, train_acc, val_acc, save_path , plot_name):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # plt.plot(epochs, train_loss, 'bo-', label='Training loss')
    # plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.plot(epochs, [x.cpu().numpy() if torch.is_tensor(x) else x for x in train_loss], 'bo-', label='Training loss')
    plt.plot(epochs, [x.cpu().numpy() if torch.is_tensor(x) else x for x in val_loss], 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
    # plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.plot(epochs, [x.cpu().numpy() if torch.is_tensor(x) else x for x in train_acc], 'bo-',
             label='Training Accuracy')
    plt.plot(epochs, [x.cpu().numpy() if torch.is_tensor(x) else x for x in val_acc], 'ro-',
             label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'training_validation_{plot_name}.png'))
    plt.show()


def save_confusion_matrix(cm, title, save_path , plot_name):
    # Ensure the confusion matrix is on CPU and converted to numpy before plotting
    cm = cm.cpu().numpy() if torch.is_tensor(cm) else cm

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(save_path, title.replace(" ", "_").lower() + f'_confusion matrix_{plot_name}.png'))
    plt.close()

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model

def print_sample_predictions(model, data_loader, device, checkpoint_path, num_samples=5):
    model = load_model(model, checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(data_loader):
            if i >= num_samples:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print("True labels:", labels.cpu().numpy())
            print("Predicted labels:", preds.cpu().numpy())


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def check_label_distribution(dataset):
    labels = []
    print("checking label distribution in dataset")
    for _, _ , label in dataset:
        labels.append(label)
    class_names = [dataset.classes_name[label] for label in labels]

    # Count occurrences of each class name
    label_count = Counter(class_names)

    # Extract sorted label names and their corresponding counts for plotting
    sorted_label_names, sorted_counts = zip(*sorted(label_count.items(), key=lambda item: item[1], reverse=True))

    # Plotting the sorted distribution
    plt.figure(figsize=(18, 14))
    plt.bar(sorted_label_names, sorted_counts)
    plt.xlabel('Class Labels' ,fontsize=16)
    plt.ylabel('Frequency')
    plt.title('Label Distribution')
    plt.xticks(rotation=40)

    plt.savefig(f"{dataset.data_dir.split('/')[-2]}_Class Distribution")
    # plt.show()


