import os.path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from multistream_model import MultiStreamModel
from multistream_dataset import Custom3DDataset
import matplotlib.pyplot as plt
import torchmetrics
import  seaborn as sns

# Assuming Custom3DDataset and MultiStreamModel classes are defined as provided

# Define transformations
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])
env = ""
data_dir ='/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset'

if env == "livingroom":
    include_classes = ['Sleeping', 'Playing video game', 'Exercising', 'Using handheld smart devices', 'Reading'
        , 'Writing', 'Working on a computer', 'Watching TV', 'Carrying object']

elif env == "kitchen":
    include_classes = ['Making pancake', 'Making a cup of coffee in coffee maker', 'Stocking up pantry', 'Dining',
                       'Cleaning dishes', 'Using handheld smart devices',
                       'Organizing the kitchen', 'Making a salad', 'Cleaning the kitchen',
                       'Making a cup of instant coffee']

elif env == "limited set":
    include_classes = ['Sleeping', 'Playing video game', 'Exercising', 'Using handheld smart devices', 'Reading'
        , 'Writing', 'Working on a computer', 'Watching TV', 'Carrying object', 'Making pancake',
                       'Making a cup of coffee in coffee maker', 'Stocking up pantry', 'Dining', 'Cleaning dishes',
                       'Using handheld smart devices', 'Making a salad', 'Making a cup of instant coffee']
else:
    include_classes = []

# Initialize datasets and dataloaders
train_dataset = Custom3DDataset(root_dir=os.path.join(data_dir , "train"), include_classes=include_classes, sequence_length=16, transform=transform)
valid_dataset = Custom3DDataset(root_dir=os.path.join(data_dir , "validation"), include_classes=include_classes, sequence_length=16,
                                transform=transform)
test_dataset = Custom3DDataset(root_dir=os.path.join(data_dir , 'test'), include_classes=include_classes, sequence_length=16, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Initialize the model
num_classes = len(train_dataset.class_names)
model = MultiStreamModel(num_classes=num_classes)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function
def train_model(model, train_loader, criterion, optimizer, device ,  epoch ,num_epochs):
    # print(f'epoch { epoch}/{ num_epochs}')
    print("Training ...")
    model.train()
    running_loss = 0.0
    for frames1, frames2, labels, sample_ids in tqdm(train_loader , desc = "Training"):
        frames1, frames2, labels = frames1.to(device), frames2.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(frames1, frames2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


# Evaluation function with majority voting
def evaluate_model(model, data_loader, device, num_classes):
    print("Evaluation ...")
    model.eval()
    all_predictions = defaultdict(list)
    all_labels = {}
    with torch.no_grad():
        for frames1, frames2, labels, sample_ids in tqdm(data_loader , desc= "Validation"):
            frames1, frames2, labels = frames1.to(device), frames2.to(device), labels.to(device)
            outputs = model(frames1, frames2)
            _, preds = torch.max(outputs, 1)
            for i, sample_id in enumerate(sample_ids):
                all_predictions[sample_id.item()].append(preds[i].item())
                all_labels[sample_id.item()] = labels[i].item()

    # Majority voting
    final_predictions = []
    final_labels = []
    for sample_id, preds in all_predictions.items():
        final_pred = Counter(preds).most_common(1)[0][0]
        final_predictions.append(final_pred)
        final_labels.append(all_labels[sample_id])

    # Calculate accuracy
    correct = sum(p == l for p, l in zip(final_predictions, final_labels))
    total = len(final_labels)
    accuracy = correct / total

    # Calculate confusion matrix
    confusion_matrix = torchmetrics.ConfusionMatrix( task= "multiclass", num_classes=num_classes)
    confusion_matrix_tensor = confusion_matrix(torch.tensor(final_predictions), torch.tensor(final_labels))

    return accuracy, confusion_matrix_tensor

# Training and evaluation
num_epochs = 10
best_valid_accuracy = 0.0
for epoch in tqdm(range(num_epochs) , desc= "epochs"):
    train_loss = train_model(model, train_loader, criterion, optimizer, device , epoch ,num_epochs )
    valid_accuracy, valid_confusion_matrix = evaluate_model(model, valid_loader, device, num_classes)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')
    print(f'Validation Confusion Matrix:\n{valid_confusion_matrix}')

# Save the model if validation accuracy improves
    if valid_accuracy > best_valid_accuracy:
        best_valid_accuracy = valid_accuracy
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_accuracy': valid_accuracy,
        }, os.path.join("/home/ghazal/Activity_Recognition_benchmarking/multiview_3D_sequence/checkpoints" , f'multiview_best_model_ep {num_epochs}.pth'))


# Test the model
test_accuracy, test_confusion_matrix = evaluate_model(model, test_loader, device, num_classes)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Confusion Matrix:\n{test_confusion_matrix}')

# Plot the confusion matrix
def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix.cpu().numpy(), annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("./confusion matrix.png")
    plt.show()

plot_confusion_matrix(test_confusion_matrix, train_dataset.class_names)