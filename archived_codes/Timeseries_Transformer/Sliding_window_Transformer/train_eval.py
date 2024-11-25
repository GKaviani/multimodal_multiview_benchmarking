from sliding_dataset import TimeSeriesDataset #, collate_fn
from Transformer_model import TransformerModel , AttentionPooling
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from torchmetrics import ConfusionMatrix
from tqdm import tqdm
import json


def train_model(model, train_loader, val_loader, criterion, optimizer,scheduler, num_epochs, num_classes ,checkpoint_path):
    best_val_accuracy = 0.0
    for epoch in tqdm(range(num_epochs) , desc= "epochs"):
        model.train()
        running_loss = 0.0
        for batch_element in tqdm(train_loader , desc= "training"):
            if batch_element is None:
                continue  # Skip this batch as it's empty
            sequences, positions, labels = batch_element
            optimizer.zero_grad()
            outputs = model(sequences)  # Model prediction for each subsequence
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * sequences.size(0)
        # Update the learning rate
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        val_accuracy, _ = evaluate_model(model, val_loader, num_classes)
        print(f'\nEpoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    #     if val_accuracy > best_val_accuracy:
    #         best_val_accuracy = val_accuracy
    #         best_model_wts = model.state_dict()
    # model.load_state_dict(best_model_wts)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        # Optionally reload best model weights at the end of training
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best model weights from checkpoint!")
    return model


def evaluate_model(model, dataloader, num_classes):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch_element in tqdm(dataloader , desc="validation"):
            if batch_element is None:
                continue  # Skip this batch as it's empty
            sequences, positions, labels = batch_element
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    overall_accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    # Class-wise accuracy
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    for label, prediction in zip(all_labels, all_predictions):
        if label == prediction:
            class_correct[label] += 1
        class_total[label] += 1

    class_wise_accuracy = class_correct / class_total
    return overall_accuracy, class_wise_accuracy


def test_model(model, test_loader, num_classes):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch_element in tqdm(test_loader , desc= "testing"):
            if batch_element is None:
                continue  # Skip this batch as it's empty
            sequences, positions, labels = batch_element
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    overall_accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    # Class-wise accuracy
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    for label, prediction in zip(all_labels, all_predictions):
        if label == prediction:
            class_correct[label] += 1
        class_total[label] += 1

    class_wise_accuracy = class_correct / class_total

    # Compute confusion matrix using torchmetrics
    confusion_matrix = ConfusionMatrix(task =  "multiclass" ,num_classes=num_classes)
    conf_matrix = confusion_matrix(torch.tensor(all_predictions), torch.tensor(all_labels))
    # print(f'type of {type(overall_accuracy)} , {type(class_wise_accuracy)} , {type(conf_matrix)}')
    return overall_accuracy, class_wise_accuracy, conf_matrix


def main():

    data_dir = '/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/EMG_Modality_Top/BioMonitor/Combined'
    # data_dir = "/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/IMU/IMU_Both"
    num_epochs = 30
    batch_size = 8

    train_dataset = TimeSeriesDataset(directory=data_dir, mode='train')
    val_dataset = TimeSeriesDataset(directory=data_dir, mode='validation')
    test_dataset = TimeSeriesDataset(directory=data_dir, mode='test')

    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False )

    input_dim = 5  # input dimension


    model = TransformerModel(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001 , weight_decay=1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
    chekoint_path = os.path.join("/archived_codes/Timeseries_Transformer/Sliding_window_Transformer/checkpoints", f'{data_dir.split("/")[-2]}_ep {num_epochs}_B {batch_size}_s {train_dataset.sub_sequence_length}.pth')
    model = train_model(model, train_loader, val_loader, criterion, optimizer,scheduler, num_epochs, num_classes , chekoint_path)
    scheduler.step()
    test_accuracy, class_wise_accuracy, conf_matrix = test_model(model, test_loader, num_classes)
    print(f'Test Overall Accuracy: {test_accuracy:.4f}')
    print('Class-wise Accuracy:')
    # for i, acc in enumerate(class_wise_accuracy):
    #     print(f'Class {i}: {acc:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    data_to_save = {
        "test_accuracy": test_accuracy.tolist(),
        "test_class_wise_accuracy": class_wise_accuracy.tolist(),
        "confusion_matrix": conf_matrix.tolist()  # Convert numpy array to list if necessary
    }

    file_name = os.path.join("/archived_codes/Timeseries_Transformer/Sliding_window_Transformer/results", f'{data_dir.split("/")[-2]}-{data_dir.split("/")[-1]}_ep {num_epochs}_B {batch_size}_s {train_dataset.sub_sequence_length}.json')

    # Save to JSON file
    with open(file_name, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

    print(f"Saved test results to {file_name}")


if __name__ == "__main__":
    main()
