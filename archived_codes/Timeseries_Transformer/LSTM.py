import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
class SensorDataset(Dataset):
    def __init__(self, data_dir, window_size=75, step_size=30):
        self.segments = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.load_data(data_dir, window_size, step_size)

    def load_data(self, data_dir, window_size, step_size):
        scaler = StandardScaler()
        for label_dir in os.listdir(data_dir):
            class_idx = self.classes.index(label_dir)
            path = os.path.join(data_dir, label_dir)
            for file in os.listdir(path):
                data_path = os.path.join(path, file)
                df = pd.read_csv(data_path)
                # Strip whitespace from column names
                df.columns = df.columns.str.strip()
                # data = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']].values
                data = df[['acc_x', 'acc_y', 'acc_z']].values
                data_scaled = scaler.fit_transform(data)
                for start in range(0, len(data_scaled) - window_size, step_size):
                    segment = data_scaled[start:start + window_size]
                    self.segments.append(segment)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return torch.tensor(self.segments[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # We only need the hidden state
        out = self.classifier(hn[-1])  # Taking the last layer's hidden state
        return out



def class_wise_accuracy(conf_matrix):
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    return np.nanmean(class_accuracy)  # mean, ignoring NaNs for classes not present

# Test the model and get the confusion matrix
def test_model(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            all_labels.extend(target.tolist())
            all_predictions.extend(predicted.tolist())
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    return all_labels, all_predictions, conf_matrix

# Plot the confusion matrix
def plot_confusion_matrix(conf_matrix, classes , name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"/home/ghazal/Activity_Recognition_benchmarking/Timeseries_Transformer/lstm_figs/{name}.png")
    plt.show()

# Directories
train_dir = os.path.join("/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/IMU/IMU_LeftArm" , "train")
val_dir = os.path.join("/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/IMU/IMU_LeftArm" , "validation")
test_dir = os.path.join("/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/IMU/IMU_LeftArm" , "test")

# Datasets
train_dataset = SensorDataset(train_dir)
val_dataset = SensorDataset(val_dir)
test_dataset = SensorDataset(test_dir)

# DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

number_of_activities = len(train_dataset.classes)
in_dim = 3
# Model
model = LSTMModel(input_dim=in_dim, hidden_dim=50, layer_num=2, output_dim=number_of_activities)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001 , weight_decay=1e-4)
#learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training and validation loop

# Initialize lists to store metrics for plotting
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_labels = []
    all_predictions = []
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        all_labels.extend(target.tolist())
        all_predictions.extend(predicted.tolist())
    scheduler.step()  # Update the learning rate

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = accuracy_score(all_labels, all_predictions)
    train_accuracies.append(train_accuracy)
    train_conf_matrix = confusion_matrix(all_labels, all_predictions)
    train_class_acc = class_wise_accuracy(train_conf_matrix)
    print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Training Class Accuracy: {train_class_acc}')

    model.eval()
    val_loss = 0
    val_labels = []
    val_predictions = []
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            val_labels.extend(target.tolist())
            val_predictions.extend(predicted.tolist())

    validation_loss = val_loss / len(val_loader)
    val_losses.append(validation_loss)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_accuracies.append(val_accuracy)
    val_conf_matrix = confusion_matrix(val_labels, val_predictions)
    val_class_acc = class_wise_accuracy(val_conf_matrix)
    print(f'Validation Loss: {validation_loss}, Validation Class Accuracy: {val_class_acc}')
test_labels, test_predictions, test_conf_matrix = test_model(model, test_loader)
plot_confusion_matrix(test_conf_matrix, train_dataset.classes , f"CF_matrix_LSTM_{num_epochs}_{train_dir.split('/')[-2]}_{in_dim}")
# Plotting the training and validation losses and accuracies
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(f"/home/ghazal/Activity_Recognition_benchmarking/Timeseries_Transformer/lstm_figs/Training and Validation_LSTM_{num_epochs}_{train_dir.split('/')[-2]}_{in_dim}.png")