import math
import torch

import torch.nn as nn

from .utils import Builder, filter_kwargs_for_module


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


# Transformer
class Transformer(nn.Module):
    def __init__(self,
                 timestep: int,
                 dimentions: int,
                 num_features: int,
                 num_layers: int = 2,
                 nhead: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.3,
                 pick_time_index: int = -1,
                 **kwargs):
        super(Transformer, self).__init__()

        __kwargs = filter_kwargs_for_module(TimeSeriesTransformer, **kwargs)

        self.transformer = TimeSeriesTransformer(timestep=timestep,
                                                 dimentions=dimentions,
                                                 d_model=num_features,
                                                 nhead=nhead,
                                                 num_layers=num_layers,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout,
                                                 **__kwargs)

        if (pick_time_index > timestep - 1) or (pick_time_index < -timestep):
            pick_time_index = -1
        self.pick_time_index = pick_time_index

    def forward(self, x):
        x = self.transformer(x)
        x = x[:, self.pick_time_index, :]
        return x


# Transformer Builder
class TransformerBuilder(Builder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        transformer_kwargs = filter_kwargs_for_module(Transformer, **kwargs)
        self.backbone = Transformer(**transformer_kwargs)


class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 timestep: int,
                 dimentions: int,
                 d_model=256,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=512,
                 dropout=0.3,
                 kernel_size=3):
        super(TimeSeriesTransformer, self).__init__()

        self.d_model = d_model
        self.input_shape = (timestep, dimentions)
        # *********************** Transformer only
        # self.embedding = nn.Linear(dimentions, d_model)
        # ********************************* single conv1d
        # self.conv1d = nn.Conv1d(in_channels=dimentions, out_channels=d_model, kernel_size=kernel_size,
        #                         padding=kernel_size // 2)
        # # Batch normalization layer
        # self.batch_norm = nn.BatchNorm1d(d_model)
        #
        # # ReLU activation
        # self.relu = nn.ReLU()
        ################################
        # Three ConvBlocks
        self.conv_block_1 = Conv1DBlock(in_channels=dimentions, out_channels=d_model // 4, kernel_size=kernel_size,
                                        padding=kernel_size // 2)
        self.conv_block_2 = Conv1DBlock(in_channels=d_model // 4, out_channels=d_model // 2, kernel_size=kernel_size,
                                        padding=kernel_size // 2)
        self.conv_block_3 = Conv1DBlock(in_channels=d_model // 2, out_channels=d_model, kernel_size=kernel_size,
                                        padding=kernel_size // 2)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, x):
        # ********************************************************************* original:embedding
        # x = self.embedding(x)
        # ********************************************************************* change
        # Convolutional layer expects (batch_size, dimensions, timestep), so we need to permute

        x = x.permute(0, 2, 1)
        # -------------------------------------------
        # # Apply 1 convolutional layer
        # x = self.conv1d(x)
        # # Apply batch normalization
        # x = self.batch_norm(x)
        # # Apply ReLU activation
        # x = self.relu(x)
        #-------------------------------------------
        # Apply the three convolutional blocks
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        # Permute back to (batch_size, timestep, d_model) for transformer
        x = x.permute(0, 2, 1)
        # ********************************************************************* change
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0, 1))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


