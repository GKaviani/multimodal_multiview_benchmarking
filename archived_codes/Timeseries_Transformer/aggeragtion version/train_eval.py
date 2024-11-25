import os
import pandas as pd
import torch
from dateutil.rrule import weekday
from torch.utils.data import Dataset , DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from timeseries_dataset import TimeSeriesDataset
from transformer_model import TransformerModel
import numpy as np
import torchmetrics
import json
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for clips, labels, _ in tqdm(train_loader , desc="training"):
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    return train_loss / len(train_loader), accuracy

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    confusion_matrix = torchmetrics.ConfusionMatrix(task = "multiclass",num_classes=num_classes).to(device)

    with torch.no_grad():
        for clips, labels, _ in tqdm(val_loader , desc="evaluating"):
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update confusion matrix
            confusion_matrix.update(predicted, labels)
    accuracy = 100. * correct / total
    cm = confusion_matrix.compute().cpu().numpy()
    return val_loss / len(val_loader), accuracy , cm



if __name__ == "__main__":

    torch.cuda.empty_cache()

    # Invoke garbage collector manually
    # gc.collect()

    # data_dir = "/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/IMU/IMU_Both"
    data_dir= "/mnt/data-tmp/ghazal/DARai_DATA/timeseries_dataL2_split/IMU_BothArm"
    # data_dir = "/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/EMG_Modality_Top/BioMonitor/Combined"
    # data_dir = "/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/EMG_Modality_Top/hands/Combined/"
    epochs = 40
    batch_size = 16

    # include_classes = [ 'Sleeping','Playing video game',  'Exercising', 'Using handheld smart devices', 'Reading'
    #             ,'Writing',  'Working on a computer', 'Watching TV', 'Carrying object' , 'Making pancake',
    #             'Making a cup of coffee in coffee maker', 'Stocking up pantry', 'Dining','Cleaning dishes',
    #             'Using handheld smart devices', 'Making a salad', 'Making a cup of instant coffee']
    include_classes = None
    train_dataset =  TimeSeriesDataset(data_dir , mode='train' , sampling_rate=12, include_classes=include_classes, max_clips_per_sample=2, clip_length=30)
    val_dataset = TimeSeriesDataset(data_dir , mode='validation' , sampling_rate=12, include_classes=include_classes, max_clips_per_sample=2, clip_length=30)
    test_dataset =  TimeSeriesDataset(data_dir , mode='test' , sampling_rate=12, include_classes=include_classes, max_clips_per_sample=2, clip_length=30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True )
# reduced from 512 , 8 , 8 , 2048
    model = TransformerModel(input_dim=6, model_dim=128, num_heads=2, num_layers=4, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001 )#, weight_decay=0.001)


    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc , val_cm = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% , Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        # print(f"Epoch{epoch+1} , confusion matix \n {val_cm}")

    test_loss , test_acc , test_cm = evaluate_model(model , test_loader , criterion , device)
    print(f"Epoch{epoch + 1} , Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% , Test confusion matix \n {test_cm}_\n{data_dir.split('/')[-1]}")
    # Prepare data to save
    data_to_save = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "confusion_matrix": test_cm.tolist()  # Convert numpy array to list if necessary
    }

    file_name = f"/home/ghazal/Activity_Recognition_benchmarking/Timeseries_Transformer/aggeragtion version/results/{data_dir.split('/')[-1]}_ep {epochs}_B {batch_size}_sequence-len {train_dataset.clip_length//train_dataset.sampling_rate}.json"  # Customize this line as needed

    # Save to JSON file
    with open(file_name, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

    print(f"Saved test results to {file_name}")