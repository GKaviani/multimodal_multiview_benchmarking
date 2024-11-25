import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
from collections import Counter


def plot_label_distribution(dataset, data_loader, plot_name):
    """
    Plots and saves the distribution of labels in a given dataset.
    """
    labels = []

    for batch in data_loader:
        # Assuming the labels are the second element in the batch (adjust if necessary)
        _, batch_labels = batch
        labels.extend(batch_labels.numpy())

    # Count the occurrences of each label
    label_counts = Counter(labels)

    # Plot the distribution
    plt.figure(figsize=(10, 5))
    plt.bar(label_counts.keys(), label_counts.values(), color='blue')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Distribution of Labels in the Dataset')
    plt.xticks(ticks=list(label_counts.keys()), labels=list(map(str, label_counts.keys())))

    # Save the plot in the same directory as the code
    plt.savefig(f'{plot_name}.png')

    # Display the plot
    plt.show()

# Example usage:
# plot_label_distribution(train_dataset, train_dataloader, 'train_label_distribution')



def create_weighted_sampler_and_class_weights(dataset):
    """
    Args:
        dataset: A dataset object where the target labels are available.
                 Assume dataset[i][1] gives the label for the i-th sample.

    Returns:
        sampler: A WeightedRandomSampler object for DataLoader.
        class_weights: A tensor of class weights for CrossEntropyLoss.
    """

    labels = [sample[1] for sample in dataset]  # Assuming dataset[i][1] is the label
    # print(f"labels {labels}")

    # Get the count of each class in the dataset
    label_counts = Counter(labels)
    total_samples = len(dataset)

    # Calculate the weight for each class
    class_weights = {label: total_samples / count for label, count in label_counts.items()}

    # Create weights for each sample based on its class
    sample_weights = [class_weights[label] for label in labels]


    # Convert the list of sample weights to a tensor
    sample_weights = torch.DoubleTensor(sample_weights)

    # Create WeightedRandomSampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                     replacement=True)

    # Calculate the max and min class sample counts
    max_samples = max(label_counts.values())
    min_samples = min(label_counts.values())
    # print(f"max {max_samples} , min {min_samples}")

    # Compute the difference
    sample_difference = max_samples - min_samples

    # Determine the number of weight groups based on the condition
    if sample_difference < 100:
        num_groups = max(1, sample_difference // 10)  # Number of weight groups
    elif sample_difference < 1000 and sample_difference > 100 :
        num_groups = max(1, sample_difference // 100)

    sorted_classes = sorted(label_counts.items(), key=lambda x: x[1])  # Sort classes by their sample count
    class_groups = np.array_split(sorted_classes, num_groups)
    # print(f'class_groups {class_groups}')

    # Weight formula: 1 - i * (1 / num_groups)
    # group_weights = [1 - i * (1 / num_groups) for i in range(num_groups)]
    # Weight formula: 1 - i *0.1
    group_weights = [1 - i * 0.1 for i in range(num_groups)]

    # class index -> weight
    class_weight_mapping = {}
    for group_idx, group in enumerate(class_groups):
        for label, _ in group:
            class_weight_mapping[label] = group_weights[group_idx]

    # Create class weights tensor for CrossEntropyLoss
    class_weights = torch.tensor([class_weight_mapping[label] for label in sorted(label_counts.keys())])
    print(f"class_weights {class_weights}")


    return sampler, class_weights

