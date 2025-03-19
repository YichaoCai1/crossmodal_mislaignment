"""
Definition of encoder architectures.
"""
import torch
from torch import nn
from typing import List, Union
from typing_extensions import Literal


def get_mlp(n_in: int, n_out: int,
            layers: List[int],
            layer_normalization: Union[None, Literal["bn"], Literal["gn"]] = None,
            act_inf_param=0.01):
    """
    Creates an MLP.

    This code originates from the following projects:
    - https://github.com/brendel-group/cl-ica
    - https://github.com/ysharma1126/ssl_identifiability

    Args:
        n_in: Dimensionality of the input data
        n_out: Dimensionality of the output data
        layers: Number of neurons for each hidden layer
        layer_normalization: Normalization for each hidden layer.
            Possible values: bn (batch norm), gn (group norm), None
    """
    modules: List[nn.Module] = []

    def add_module(n_layer_in: int, n_layer_out: int, last_layer: bool = False):
        modules.append(nn.Linear(n_layer_in, n_layer_out))
        # perform normalization & activation not in last layer
        if not last_layer:
            if layer_normalization == "bn":
                modules.append(nn.BatchNorm1d(n_layer_out))
            elif layer_normalization == "gn":
                modules.append(nn.GroupNorm(1, n_layer_out))
            modules.append(nn.LeakyReLU(negative_slope=act_inf_param))

        return n_layer_out

    if len(layers) > 0:
        n_out_last_layer = n_in
    else:
        assert n_in == n_out, "Network with no layers must have matching n_in and n_out"
        modules.append(layers.Lambda(lambda x: x))

    layers.append(n_out)

    for i, l in enumerate(layers):
        n_out_last_layer = add_module(n_out_last_layer, l, i == len(layers)-1)

    return nn.Sequential(*modules)


class TextEncoder2D(nn.Module):
    """2D-ConvNet to encode text data."""

    def __init__(self, input_size, output_size, sequence_length,
                 embedding_dim=128, fbase=25):
        super(TextEncoder2D, self).__init__()
        if sequence_length < 24 or sequence_length > 31:
            raise ValueError(
                "TextEncoder2D expects sequence_length between 24 and 31")
        self.fbase = fbase
        self.embedding = nn.Linear(input_size, embedding_dim)
        self.convnet = nn.Sequential(
            # input size: 1 x sequence_length x embedding_dim
            nn.Conv2d(1, fbase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase),
            nn.ReLU(True),
            nn.Conv2d(fbase, fbase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase * 2),
            nn.ReLU(True),
            nn.Conv2d(fbase * 2, fbase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase * 4),
            nn.ReLU(True),
            # size: (fbase * 4) x 3 x 16
        )
        self.ldim = fbase * 4 * 3 * 16
        self.linear = nn.Linear(self.ldim, output_size)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.convnet(x)
        x = x.view(-1, self.ldim)
        x = self.linear(x)
        return x


class TransformerTextEncoder(nn.Module):
    """Improved Transformer encoder with positional encoding and pooling."""
    def __init__(self, input_size, output_size, sequence_length,
                 embedding_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerTextEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(input_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, sequence_length, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True  # Ensures inputs are batch-first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embedding_dim, output_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)  # (batch_size, sequence_length, embedding_dim)
        x = x.mean(dim=1)  # Pooling by averaging over sequence length
        x = self.linear(x)
        return x


class FlexibleTextEncoder2D(nn.Module):
    def __init__(self, input_size, output_size, sequence_length,
                 embedding_dim=128, fbase=25):
        super(FlexibleTextEncoder2D, self).__init__()
        if sequence_length < 4 or sequence_length > 164:
            raise ValueError(
                "FlexibleTextEncoder2D expects sequence_length between 4 and 164")

        self.embedding = nn.Linear(input_size, embedding_dim)
        self.fbase = fbase

        layers = []
        in_channels = 1
        out_channels = fbase

        current_h = sequence_length
        current_w = embedding_dim

        while current_h >= 4 and current_w >= 4:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            current_h = (current_h + 2*1 - 4) // 2 + 1
            current_w = (current_w + 2*1 - 4) // 2 + 1

            in_channels = out_channels
            out_channels = min(out_channels * 2, fbase * 8)

        self.convnet = nn.Sequential(*layers)

        # Dummy forward pass to get exact dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, sequence_length, input_size)
            dummy_embedded = self.embedding(dummy_input).unsqueeze(1)
            conv_out = self.convnet(dummy_embedded)
            self.ldim = conv_out.view(1, -1).size(1)

        self.linear = nn.Linear(self.ldim, output_size)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# Validation code to ensure correctness
def validate_encoder():
    input_size = 64
    output_size = 128
    embedding_dim = 128

    # Test a range of allowed sequence lengths
    for seq_len in range(4, 165):
        model = FlexibleTextEncoder2D(input_size, output_size, seq_len, embedding_dim)
        x = torch.randn(2, seq_len, input_size)
        try:
            y = model(x)
            assert y.shape == (2, output_size), f"Incorrect output shape: {y.shape} for seq_len={seq_len}"
        except Exception as e:
            print(f"Error for sequence length {seq_len}: {e}")
            return False

    print("Validation successful for all allowed sequence lengths (4-164).")
    return True

if __name__ == '__main__':
    validate_encoder()