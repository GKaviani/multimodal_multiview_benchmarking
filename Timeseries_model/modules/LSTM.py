import torch.nn as nn

from .utils import Builder, filter_kwargs_for_module


# Conv1d Projection + LSTM
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

class LSTM(nn.Module):
    def __init__(self,
                 dimentions: int,
                 num_features: int,
                 num_layers: int=1,kernel_size=3,
                 **kwargs):
        super(LSTM, self).__init__()

        __kwargs = filter_kwargs_for_module(LSTM, **kwargs)
        self.conv_block_1 = Conv1DBlock(in_channels=dimentions, out_channels=dimentions // 4, kernel_size=kernel_size,
                                        padding=kernel_size // 2)
        self.conv_block_2 = Conv1DBlock(in_channels=dimentions // 4, out_channels=dimentions // 2, kernel_size=kernel_size,
                                        padding=kernel_size // 2)
        self.conv_block_3 = Conv1DBlock(in_channels=dimentions // 2, out_channels=dimentions, kernel_size=kernel_size,
                                        padding=kernel_size // 2)


        self.lstm = nn.LSTM(input_size=dimentions,
                            hidden_size=num_features,
                            num_layers=num_layers,
                            batch_first=True,
                            **__kwargs)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        # ******************************************************************** change
        # Convolutional layer expects ((batch, channels, sequence length)
        x = x.transpose(1, 2)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        # Transpose back to match LSTM input format (batch, sequence length, channels)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = x[:, -1, :]
        return x


# Normal LSTM Builder
class LSTMBuilder(Builder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        lstm_kwargs = filter_kwargs_for_module(LSTM, **kwargs)
        self.backbone = LSTM(**lstm_kwargs)


