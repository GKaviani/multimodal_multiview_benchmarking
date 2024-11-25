import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, num_subsequences, input_dim]
        weights = torch.softmax(self.attention(x), dim=1)
        output = torch.sum(weights * x, dim=1)
        return output

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=4, hidden_dim=64 , conv_output_dim = 128, dropout_prob=0.5):
        super(TransformerModel, self).__init__()
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.attention_pooling = AttentionPooling(input_dim)
        # self.fc = nn.Linear(input_dim, num_classes)
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=conv_output_dim, kernel_size=3, padding=1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=conv_output_dim, nhead=num_heads,
                                                        dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.attention_pooling = AttentionPooling(conv_output_dim)
        self.fc = nn.Linear(conv_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x shape: [batch_size, num_subsequences, sub_sequence_length, input_dim]
        batch_size, num_subsequences, sub_sequence_length, input_dim = x.shape
        x = x.view(batch_size * num_subsequences, sub_sequence_length, input_dim)  # Flatten subsequences into batch
        #no conv1d >>
        # x = x.transpose(0, 1)  # Transformer expects [sequence_length, batch_size, input_dim]
        # Apply 1D convolutional layer >>
        x = x.transpose(1, 2)  # Change shape to [batch_size * num_subsequences, input_dim, sub_sequence_length]
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # Change back to [batch_size * num_subsequences, sub_sequence_length, conv_output_dim]
        # Apply dropout
        x = self.dropout(x)
        x = x.transpose(0, 1)  # Transformer expects [sequence_length, batch_size, input_dim]
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Convert back to [batch_size * num_subsequences, sub_sequence_length, input_dim]
        x = x.mean(dim=1)  # Aggregate over the sub_sequence_length
        # x = x.view(batch_size, num_subsequences, input_dim)  # Restore original batch and subsequence dimensions
        x = x.view(batch_size, num_subsequences, -1)  # Restore original batch and subsequence dimensions
        x = self.attention_pooling(x)
        x = self.fc(x)
        return x