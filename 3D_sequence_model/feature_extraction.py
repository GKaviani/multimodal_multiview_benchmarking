import os
import time
import copy
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.video import r3d_18
from dataset import Custom3DDataset  , train_transforms , test_transforms
from utils import set_seed, plot_training_validation_loss_accuracy
import torch
import torchvision.models.video as models
import numpy as np

# Load the pre-trained r3d_18 model
model = models.r3d_18(pretrained=True)

# Print the model architecture
print(model)

class ResNet3D_Features(nn.Module):
    def __init__(self, original_model):
        super(ResNet3D_Features, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x

# Create the feature extraction model
feature_extractor = ResNet3D_Features(model)
feature_extractor.eval()
# Load the 3D ResNet model
model = models.r3d_18(pretrained=False)
model_path = '/home/ghazal/Activity_Recognition_benchmarking/checkpoints/best_3d_resnet_model_single-random_seed 1_ep 10_B 16 T 16_both_cam_1.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

features_list = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)

dataset = CustomDataset(frame_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for frames in dataloader:
        frames = frames.to(device)  # Move input to the appropriate device
        features = feature_extractor(frames)  # Extract features
        features = features.squeeze().cpu().numpy()  # Remove unnecessary dimensions and move to CPU
        features_list.append(features)

# Save the extracted features
features_array = np.array(features_list)
np.save('/path/to/save/features.npy', features_array)
