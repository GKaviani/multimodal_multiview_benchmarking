import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader

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
