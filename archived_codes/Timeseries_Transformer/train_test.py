from torchsampler import ImbalancedDatasetSampler
import os.path
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import TimeSeriesDataset
from model import TimeSeriesTransformerClassifier
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import custom_collate_fn , plot_metrics , save_confusion_matrix, set_seed
import gc
import matplotlib.pyplot as plt
import torchmetrics
from torch import nn
import numpy as np

# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)
#
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#                 dataloader = train_loader
#             else:
#                 model.eval()   # Set model to evaluate mode
#                 dataloader = val_loader
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             # Iterate over data.
#             for inputs in dataloader:
#                 clips = inputs[0]
#                 labels = inputs[1]
#                 clips = clips.to(device)
#                 labels = labels.to(device)
#
#                 # Zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # Forward
#                 # Track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(clips)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#
#                     # Backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 # Statistics
#                 running_loss += loss.item() * clips.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#
#             epoch_loss = running_loss / len(dataloader.dataset)
#             epoch_acc = running_corrects.double() / len(dataloader.dataset)
#
#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#
#             # Deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#
#         print()
#
#     print(f'Best val Acc: {best_acc:.4f}')
#
#     # Load best model weights
#     model.load_state_dict(best_model_wts)
#     return model
#
#
#
# def evaluate_model(model, test_loader):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#     model.eval()  # Set model to evaluate mode
#
#     running_corrects = 0
#
#     for inputs in test_loader:
#         clips = inputs[0]
#         labels = inputs[1]
#         clips = clips.to(device)
#         labels = labels.to(device)
#
#         # Forward
#         with torch.no_grad():
#             outputs = model(clips)
#             _, preds = torch.max(outputs, 1)
#
#         running_corrects += torch.sum(preds == labels.data)
#
#     acc = running_corrects.double() / len(test_loader.dataset)
#     print(f'Test Acc: {acc:.4f}')
##################################### Working simple version #####################################

# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#
#     metrics = {
#         'train': {
#             'loss': [],
#             'accuracy': [],
#             'conf_matrix': torchmetrics.ConfusionMatrix(task= 'multiclass' , num_classes=len(train_loader.dataset.classes)).to(device)
#         },
#         'val': {
#             'loss': [],
#             'accuracy': [],
#             'conf_matrix': torchmetrics.ConfusionMatrix(task = 'multiclass' , num_classes=len(val_loader.dataset.classes)).to(device)
#         }
#     }
#
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)
#
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#                 dataloader = train_loader
#             else:
#                 model.eval()
#                 dataloader = val_loader
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             for inputs in dataloader:
#                 clips, labels = inputs[0].to(device), inputs[1].to(device)
#                 optimizer.zero_grad()
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(clips)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 running_loss += loss.item() * clips.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#                 metrics[phase]['conf_matrix'].update(preds, labels)
#
#             epoch_loss = running_loss / len(dataloader.dataset)
#             epoch_acc = (running_corrects.double() / len(dataloader.dataset)).item()
#             metrics[phase]['loss'].append(epoch_loss)
#             metrics[phase]['accuracy'].append(epoch_acc)
#
#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#
#     print(f'Best val Acc: {best_acc:.4f}')
#     model.load_state_dict(best_model_wts)
#
#     # Plotting
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(metrics['train']['loss'], label='Train Loss')
#     plt.plot(metrics['val']['loss'], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training vs Validation Loss')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     t_acc = metrics['train']['accuracy']
#     v_acc = metrics['val']['accuracy']
#     plt.plot(metrics['val']['accuracy'], label='Train Accuracy')
#     plt.plot(metrics['val']['accuracy'], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training vs Validation Accuracy')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#     return model
#
# def evaluate_model(model, test_loader):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#     model.eval()
#
#     running_corrects = 0
#     conf_matrix = torchmetrics.ConfusionMatrix(task = 'multiclass',num_classes=len(test_loader.dataset.classes)).to(device)
#
#     for inputs in test_loader:
#         clips, labels = inputs[0].to(device), inputs[1].to(device)
#
#         with torch.no_grad():
#             outputs = model(clips)
#             _, preds = torch.max(outputs, 1)
#
#         running_corrects += torch.sum(preds == labels.data)
#         conf_matrix.update(preds, labels)
#
#     acc = running_corrects.double() / len(test_loader.dataset)
#     print(f'Test Acc: {acc:.4f}')
#
#     # Plotting the confusion matrix
#     cm = conf_matrix.compute().cpu().numpy()
#     plt.figure(figsize=(10, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title('Confusion Matrix')
#     plt.colorbar()
#     tick_marks = np.arange(len(test_loader.dataset.classes))
#     plt.xticks(tick_marks, test_loader.dataset.classes, rotation=45)
#     plt.yticks(tick_marks, test_loader.dataset.classes)
#
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             plt.text(j, i, format(cm[i, j], 'd'),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig('confusion_matrix.png')  # Save the plot as a file
#     plt.show()

import torchmetrics


def train_model(model, train_loader, optimizer, criterion, device, num_classes):
    model.train()
    train_loss = 0
    # class_wise_accuracy = torchmetrics.Accuracy(task = "multiclass" ,num_classes=num_classes, average=None , top_k=3).to(device)
    micro_accuracy = torchmetrics.Accuracy(task = "multiclass" ,num_classes=num_classes, average='micro' ).to(device)

    for batch in tqdm(train_loader , desc="Training", leave=False):
        # Assuming each batch returns data and targets
        data, targets = batch[0].to(device), batch[2].to(device)
        # data = data.to(device)
        # targets = target.to(device)
        optimizer.zero_grad()

        # Convert targets to one-hot format for MSE Loss
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()  # Convert to float for MSE loss

        outputs = model(data)
        # loss = criterion(outputs, targets)
        loss = criterion(outputs, targets_one_hot)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Update metrics
        # class_wise_accuracy.update(outputs, targets)
        # micro_accuracy.update(outputs, targets)
        # Convert model outputs to predicted class indices for MSE loss
        predicted_classes = outputs.argmax(dim=1)
        micro_accuracy.update(predicted_classes, targets)

    avg_loss = train_loss / len(train_loader)
    # class_wise_acc = class_wise_accuracy.compute()
    acc = micro_accuracy.compute()
    return avg_loss,  acc #class_wise_acc,


def evaluate_model(model, val_loader, criterion, device, num_classes):
    model.eval()
    eval_loss = 0
    # class_wise_accuracy = torchmetrics.Accuracy(task = "multiclass" ,num_classes=num_classes, average=None , top_k=3).to(device)
    micro_accuracy = torchmetrics.Accuracy(task = "multiclass" ,num_classes=num_classes, average='micro').to(device)
    confusion_matrix = torchmetrics.ConfusionMatrix(task = "multiclass" ,num_classes=num_classes).to(device)

    with torch.no_grad():
        for batch in tqdm(val_loader , desc= "Validation" , leave=False):
            data, targets = batch[0].to(device), batch[2].to(device)
            outputs = model(data)

            # Convert targets to one-hot format for MSE Loss
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()  # Convert to float for MSE loss

            # loss = criterion(outputs, targets)
            loss = criterion(outputs, targets_one_hot)
            eval_loss += loss.item()

            # Update metrics
            # class_wise_accuracy.update(outputs, targets)
            # micro_accuracy.update(outputs, targets)
            # Convert model outputs to predicted class indices for MSE loss
            predicted_classes = outputs.argmax(dim=1)
            micro_accuracy.update(predicted_classes, targets)

            confusion_matrix.update(outputs.argmax(dim=1), targets)

    avg_loss = eval_loss / len(val_loader)
    # class_wise_acc = class_wise_accuracy.compute()
    acc = micro_accuracy.compute()
    cm = confusion_matrix.compute()
    return avg_loss,acc, cm # class_wise_acc,


# Example usage
if __name__ == "__main__":
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Invoke garbage collector manually
    gc.collect()

    seed =1
    set_seed(seed)


    activity_classes = ['Making pancake', 'Making a cup of coffee in coffee maker', 'Stocking up pantry', 'Dining',
          'Cleaning dishes',  'Using handheld smart devices',
         'Organizing the kitchen',  'Making a salad', 'Cleaning the kitchen',
         'Making a cup of instant coffee', 'Sleeping','Playing video game',  'Exercising', 'Using handheld smart devices', 'Reading'
            ,'Writing',  'Working on a computer', 'Watching TV', 'Carrying object' , 'Making pancake',
            'Making a cup of coffee in coffee maker', 'Stocking up pantry', 'Dining','Cleaning dishes',
            'Using handheld smart devices', 'Making a salad', 'Making a cup of instant coffee']

    data_dir = '/mnt/data-tmp/ghazal/DARai_DATA/timeseries_data/EMG_Modality_Top/Insole-RT/Insole-Total'
    train_dataset = TimeSeriesDataset(
        data_dir, mode="train",
        sampling_rate=400,
        include_classes=activity_classes, max_subseq=2 , subseq_length= 6)
    val_dataset = TimeSeriesDataset(
        data_dir, mode="validation",
        sampling_rate=400,
        include_classes=activity_classes, max_subseq=2 , subseq_length= 6)
    test_dataset = TimeSeriesDataset(
        data_dir, mode="test",
        sampling_rate=400,
        include_classes=activity_classes, max_subseq=2 , subseq_length= 6)

    sampling_rate = train_dataset.sampling_rate
    seq_len = train_dataset.sub_sequence_length
    num_classes = len(train_dataset.classes)
    num_sub_seq = train_dataset.max_subseq_per_sample

    train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=8) #, shuffle=True ) #, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, sampler=ImbalancedDatasetSampler(val_dataset),batch_size=8) #, shuffle=False ) #, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset,sampler=ImbalancedDatasetSampler(test_dataset), batch_size=8) #, shuffle=False) # , collate_fn=custom_collate_fn)

    # model = TimeSeriesTransformerClassifier(input_dim=512, num_heads=8, num_layers=6, dim_feedforward=2048,
    #                                         sequence_length=seq_len, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformerClassifier(
        input_dim=1,  # Reduced from 512
        num_heads=4,  # Reduced from 8
        num_layers=4,  # Reduced from 6
        model_dim=64,  # Reduced from 2048
        dropout_rate= 0.5 ,
        sequence_length=seq_len,
        num_subsequences= num_sub_seq,
        num_classes=num_classes,
        device= device
    )


    model = model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # Replace the loss function
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 ,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # Train the model
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    epochs = 10
    print(device)
    for epoch in range(epochs):  # Number of epochs
        start_time = time.time()
        t_loss, t_acc = train_model(model, train_loader, optimizer, criterion, device,num_classes)
        v_loss ,v_acc, v_cm = evaluate_model(model, val_loader, criterion, device,num_classes)
        scheduler.step()
        print(f'Epoch {epoch + 1}, Train Loss: {t_loss}, Train Acc: {t_acc}, Val Loss: {v_loss}, Val Acc: {v_acc}')
        print(f'Epoch {epoch + 1} Confusion Matrix:\n', v_cm)
        # print(f'Epoch {epoch + 1}, Train classwise Acc: {t_class_wise_acc}, Val classwise acc: {v_class_wise_acc}')
        # train_loss.append(t_loss)
        # val_loss.append(v_loss)
        # train_acc.append(t_acc)
        # val_acc.append(v_acc)
        train_loss.append(t_loss if isinstance(t_loss, float) else t_loss.item())
        val_loss.append(v_loss if isinstance(v_loss, float) else v_loss.item())
        train_acc.append(t_acc if isinstance(t_acc, float) else t_acc.item())
        val_acc.append(v_acc if isinstance(v_acc, float) else v_acc.item())

        end_time = time.time()
        print(f"epoch {epoch +1} took {end_time - start_time}")
    # Directory for saving figures
    save_dir = "/archived_codes/Timeseries_Transformer/figs"
    plot_name = f'ep {epochs}_seed {seed}_{os.path.basename(data_dir)}'
    # plot_metrics(train_loss, val_loss, train_acc, val_acc)
    # print("Confusion Matrix:\n", v_cm)
    plot_metrics(train_loss, val_loss, train_acc, val_acc, save_dir ,plot_name)
    save_confusion_matrix(v_cm, 'Validation Confusion Matrix', save_dir , plot_name)

    test_loss,test_acc, test_cm = evaluate_model(model, val_loader, criterion, device,num_classes)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")
    # print(f"Test Classwise Accuracy: {test_class_wise_acc}")
    print("Test Confusion Matrix:\n", test_cm)
    save_confusion_matrix(test_cm, 'Test Confusion Matrix', save_dir , plot_name)