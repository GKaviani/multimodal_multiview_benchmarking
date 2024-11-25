import torch
import torch.nn as nn
import torchvision.models.video as models
from torchvision.models.video import r3d_18 , mvit_v2_s

class MultiStreamModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiStreamModel, self).__init__()
        self.stream1 = r3d_18(weights="DEFAULT")
        self.stream2 = r3d_18(weights="DEFAULT")  # Stream for cam_2

        # Modify the final layer of each stream
        self.stream1.fc = nn.Identity()
        self.stream2.fc = nn.Identity()

        # Fusion layer
        self.fc = nn.Linear(2 * 512, num_classes)  # Assuming 512 is the feature size after the final layer of ResNet3D

    def forward(self, x1, x2):
        # Forward pass through each stream
        features1 = self.stream1(x1)
        features2 = self.stream2(x2)

        # Concatenate features
        combined_features = torch.cat((features1, features2), dim=1)

        # Classification
        output = self.fc(combined_features)
        return output

if __name__ == "__main__":
    # Instantiate the model
    num_classes = 10
    model = MultiStreamModel(num_classes)
    print(model)
