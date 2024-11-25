import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.projection = nn.Conv1d(in_channels=input_dim, out_channels=model_dim, kernel_size=6, padding=2)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout_rate),
            num_layers=num_layers
        )
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # print(f'x.size {x.size()}')
        # batch_size, num_clips, clip_length = x.size()
        # x = x.view(-1, 1, clip_length)  # Reshape for Conv1D: [batch_size * num_clips, 1, clip_length]
        batch_size, num_clips, clip_length, input_dim = x.size()

        # Reshape for Conv1D: [batch_size * num_clips, input_dim, clip_length]
        x = x.view(batch_size * num_clips, input_dim, clip_length)

        x = self.projection(x)  # [batch_size * num_clips, model_dim, clip_length]
        x = x.permute(2, 0, 1)  # Transformer expects [seq_length, batch_size, model_dim]
        x = self.transformer(x)  # [clip_length, batch_size * num_clips, model_dim]
        x = x.mean(dim=0)  # Aggregate along the sequence length: [batch_size * num_clips, model_dim]
        x = x.view(batch_size, num_clips, -1)  # [batch_size, num_clips, model_dim]
        x = x.mean(dim=1)  # Aggregate along the num_clips: [batch_size, model_dim]
        output = self.fc(x)  # [batch_size, num_classes]
        return output
