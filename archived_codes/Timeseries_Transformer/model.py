import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# class TimeSeriesTransformerClassifier(nn.Module):
#     def __init__(self, input_dim, num_heads, num_layers, dim_feedforward, sequence_length, num_classes):
#         super(TimeSeriesTransformerClassifier, self).__init__()
#
#         # 1D Convolution to project the input sequence
#         self.conv1d = nn.Conv1d(in_channels=1, out_channels=input_dim, kernel_size=8, stride=1, padding=0)
#
#         # Adjusting sequence length after convolution
#         self.adjusted_length = sequence_length - 7  # Assuming no padding and stride of 1
#
#         # Transformer Encoder Layer
#         encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # Positional encoding
#         self.positional_encoding = nn.Parameter(torch.randn(self.adjusted_length, input_dim))
#
#         # Classifier head
#         self.fc = nn.Linear(input_dim, num_classes)
#
#         # Pooling layer to aggregate encoder outputs
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
#     #
#     # def forward(self, x):
#     #     # x is of shape (batch_size, sequence_length)
#     #     x = x.unsqueeze(1)  # Reshape to (batch_size, 1, sequence_length) for Conv1D
#     #     x = self.conv1d(x)  # Shape becomes (batch_size, input_dim, adjusted_length)
#     #     x = F.relu(x)
#     #     x = x.transpose(1, 2)  # Reshape to (batch_size, adjusted_length, input_dim)
#     #
#     #     # Adding positional encoding
#     #     x = x + self.positional_encoding
#     #
#     #     x = self.transformer_encoder(x)
#     #
#     #     # Pool and flatten before the classifier
#     #     x = self.global_avg_pool(x.transpose(1, 2)).squeeze(2)  # Now shape is (batch_size, input_dim)
#     #     x = self.fc(x)
#     #     return x
#     def forward(self, x):
#         # x shape is initially [batch_size, num_subseq, seq_length]
#         # Flatten batch and sub-sequence dimensions
#         original_shape = x.shape  # Store original shape
#         x = x.view(-1, x.size(-1))  # Flatten: [batch_size * num_subseq, seq_length]
#         x = x.unsqueeze(1)  # Add channel dimension: [batch_size * num_subseq, 1, seq_length]
#
#         # Apply conv1d
#         x = self.conv1d(x)  # Shape becomes [batch_size * num_subseq, input_dim, adjusted_length]
#         x = F.relu(x)
#
#         # Restore batch and sub-sequence dimensions
#         # Calculate new sequence length after convolution
#         new_seq_len = x.size(-1)
#         x = x.view(original_shape[0], original_shape[1], -1,
#                    new_seq_len)  # Reshape to [batch_size, num_subseq, input_dim, new_seq_len]
#         x = x.transpose(2,
#                         3)  # Swap input_dim and new_seq_len for transformer: [batch_size, num_subseq, new_seq_len, input_dim]
#
#         # Flatten sub-sequence dimension back into batch dimension if needed
#         x = x.reshape(-1, new_seq_len, x.size(-1))  # Now [batch_size * num_subseq, new_seq_len, input_dim]
#
#         # Adding positional encoding (adjust dimensions as needed)
#         x = x + self.positional_encoding[:new_seq_len, :]
#
#         # Transformer Encoder
#         x = self.transformer_encoder(x)
#
#         # Pool and flatten before the classifier
#         x = self.global_avg_pool(x.transpose(1, 2)).squeeze(2)  # Global pool: [batch_size * num_subseq, input_dim]
#         x = self.fc(x)  # Classifier: [batch_size * num_subseq, num_classes]
#
#         # Average predictions across sub-sequences for each item in the batch
#         x = x.view(original_shape[0], original_shape[1], -1).mean(dim=1)  # [batch_size, num_classes]
#
#         return x
#     ##### forward version that aggregates all output prediction of sub_sequences of a sample together.####
#     # def forward(self, x):
#     #     batch_size, num_subseq, seq_length = x.shape
#     #     x = x.view(batch_size * num_subseq, 1, seq_length)  # Flatten sub-seqs to batch dimension for processing
#     #
#     #     # Process as before
#     #     x = self.conv1d(x)  # [batch_size * num_subseq, input_dim, adjusted_length]
#     #     x = F.relu(x)
#     #     x = x.transpose(1, 2)  # [batch_size * num_subseq, adjusted_length, input_dim]
#     #     x = x + self.positional_encoding
#     #     x = self.transformer_encoder(x)
#     #     x = self.global_avg_pool(x.transpose(1, 2)).squeeze(2)  # [batch_size * num_subseq, input_dim]
#     #
#     #     x = self.fc(x)  # [batch_size * num_subseq, num_classes]
#     #     x = x.view(batch_size, num_subseq, -1).mean(dim=1)  # Average across sub-seqs
#     #
#     #     return x
#     ##### forward version that aggregates all output prediction of sub_sequences of a sample together.####

import torch
import torch.nn as nn
import torch.nn.functional as F

#
# class TimeSeriesTransformerClassifier(nn.Module):
#     def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout_rate, sequence_length,
#                  num_subsequences, device):
#         super(TimeSeriesTransformerClassifier, self).__init__()
#         self.device = device
#         self.num_subsequences = num_subsequences
#         self.sequence_length = sequence_length
#         self.total_seq_len = num_subsequences * sequence_length  # Total length after concatenation
#
#         # Projection layer: 1D Convolution
#         self.projection = nn.Conv1d(in_channels=1, out_channels=model_dim, kernel_size=6, padding=2)
#
#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout_rate)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # Positional encoding
#         self.positional_encoding = PositionalEncoding(model_dim, max_len=num_subsequences * sequence_length,
#                                                       device=device)
#
#
#         # Classifier layer
#         self.classifier = nn.Linear(model_dim, num_classes)
#
#     def forward(self, x):
#         # x shape: [batch_size, num_subsequences, seq_len]
#         # Reshape to treat each group of subsequences as part of the same sequence
#         x = x.to(self.device)
#         x = x.view(-1, 1, self.total_seq_len)  # Flatten subsequences into one long sequence
#
#         # Apply 1D convolution
#         x = self.projection(x)  # [batch_size, model_dim, conv_output_length]
#         x = x.permute(0, 2, 1)  # Change to [batch_size, conv_output_length, model_dim]
#
#         # Add positional encoding
#         x = self.positional_encoding(x)
#
#         # Transformer processing
#         x = self.transformer_encoder(x)
#
#         # Classifier processing
#         x = x.mean(dim=1)  # Aggregate features across the sequence
#         output = self.classifier(x)
#         return output
#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000, device=None):
#         super(PositionalEncoding, self).__init__()
#
#         self.device = device
#         self.encoding = torch.zeros(max_len, d_model, device=device)
#         position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
#         self.encoding[:, 0::2] = torch.sin(position * div_term)
#         self.encoding[:, 1::2] = torch.cos(position * div_term)
#         self.encoding = self.encoding.unsqueeze(0)
#
#     def forward(self, x):
#         # x shape: [batch_size, seq_len, model_dim]
#         x = x.to(self.device)
#         return x + self.encoding[:, :x.size(1)]
#
#supports multiple dim input:
class TimeSeriesTransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout_rate, sequence_length,
                 num_subsequences, device):
        super(TimeSeriesTransformerClassifier, self).__init__()
        self.device = device
        self.num_subsequences = num_subsequences
        self.sequence_length = sequence_length
        self.total_seq_len = num_subsequences * sequence_length  # Total length after concatenation

        # Projection layer: 1D Convolution for multiple input dimensions (6D)
        self.projection = nn.Conv1d(in_channels=input_dim, out_channels=model_dim, kernel_size=6, padding=2)

        self.norm = nn.LayerNorm(model_dim)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(model_dim, max_len=num_subsequences * sequence_length,
                                                      device=device)

        # Classifier layer
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # x shape: [batch_size, num_subsequences, seq_len, input_dim]
        # Reshape to treat each group of subsequences as part of the same sequence
        x = x.to(self.device)
        batch_size = x.size(0)
        x = x.view(batch_size, self.total_seq_len, -1)  # Flatten subsequences into one long sequence
        x = x.permute(0, 2, 1)  # Change to [batch_size, input_dim, total_seq_len]

        # Apply 1D convolution
        x = self.projection(x)  # [batch_size, model_dim, conv_output_length]
        x = x.permute(0, 2, 1)  # Change to [batch_size, conv_output_length, model_dim]
        x = self.norm(x)  # [batch_size, conv_output_length, model_dim]
        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer processing
        x = self.transformer_encoder(x)

        # Classifier processing
        x = x.mean(dim=1)  # Aggregate features across the sequence
        output = self.classifier(x)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device=None):
        super(PositionalEncoding, self).__init__()

        self.device = device
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        # x shape: [batch_size, seq_len, model_dim]
        x = x.to(self.device)
        return x + self.encoding[:, :x.size(1)]

# Example model initialization

# Example usage
if __name__ == "__main__":
    model = TimeSeriesTransformerClassifier(input_dim=1, num_heads=8, num_layers=6, num_classes=10 , model_dim= 128 , sequence_length= 3000)
    print(model)
