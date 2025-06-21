import torch
import torch.nn as nn
from typing import List
from torchmultimodal.models.late_fusion import LateFusion
from torchmultimodal.modules.fusions.attention_fusion import AttentionFusionModule
from torchmultimodal.modules.layers.mlp import MLP


class CNNEncoder(nn.Module):
    """A CNN encoder.

    Stacks n layers of (Conv2d, MaxPool2d, BatchNorm2d), where n is determined
    by the length of the input args.

    Args:
        input_dims (List[int]): List of input dimensions.
        output_dims (List[int]): List of output dimensions. Should match
            input_dims offset by one.
        kernel_sizes (List[int]): Kernel sizes for convolutions. Should match
            the sizes of cnn_input_dims and cnn_output_dims.

    Inputs:
        x (Tensor): Tensor containing a batch of images.
    
    """

    def __init__(
        self, input_dims: List[int], output_dims: List[int], kernel_sizes: List[int]
    ):
        super().__init__()
        conv_layers: List[nn.Module] = []
        assert len(input_dims) == len(output_dims) and len(output_dims) == len(
            kernel_sizes
        ), "input_dims, output_dims, and kernel_sizes should all have the same length"
        assert (
            input_dims[1:] == output_dims[:-1]
        ), "output_dims should match input_dims offset by one"
        for in_channels, out_channels, kernel_size in zip(
            input_dims,
            output_dims,
            kernel_sizes,
        ):
            padding_size = kernel_size // 2

            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding_size
            )

            max_pool2d = nn.MaxPool2d(2, stride=2)
            batch_norm_2d = nn.BatchNorm2d(out_channels)

            conv_layers.append(
                nn.Sequential(conv, nn.LeakyReLU(), max_pool2d, batch_norm_2d)
            )

        conv_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


class LSTMEncoder(nn.Module):
    """An LSTM encoder. Stacks an LSTM on an embedding layer.

    Args:
        vocab_size (int): The size of the vocab for embeddings.
        embedding_dim (int): The size of each embedding vector.
        input_size (int): The number of features in the LSTM input.
        hidden_size (int): The number of features in the hidden state.
        bidirectional (bool): Whether to use bidirectional LSTM.
        batch_first (bool): Whether to provide batches as (batch, seq, feature)
            or (seq, batch, feature).

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        input_size: int,
        hidden_size: int,
        bidirectional: bool,
        batch_first: bool,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, x = self.lstm(self.embedding(x))
        # N x B x H => B x X x H where N = num_layers * num_directions
        x = x[0].transpose(0, 1)

        # N should be 2 so we can merge in that dimension
        assert x.size(1) == 2, "hidden state (final) should have 1st dim as 2"

        x = torch.cat([x[:, 0, :], x[:, 1, :]], dim=-1)
        return x


def build_multimodal_model(
    text_vocab_size,
    num_classes=2,
    lstm_hidden_size=100,
    mlp_hidden_dims=[128],
    mlp_dropout=0.3
):
    # Image Encoder (CNN)
    image_encoder = CNNEncoder(
        input_dims=[3, 64, 128, 128, 64, 64],
        output_dims=[64, 128, 128, 64, 64, 10],
        kernel_sizes=[7, 5, 5, 5, 5, 1],
    )

    # Text Encoder (LSTM)
    text_encoder = LSTMEncoder(
        vocab_size=text_vocab_size,
        embedding_dim=50,
        input_size=50, # Should match embedding_dim
        hidden_size=lstm_hidden_size,
        bidirectional=True,
        batch_first=True,
    )

    # Attention Fusion Module
    # CNN output is 10 * 3 * 3 = 90. LSTM output is lstm_hidden_size * 2 (bidirectional)
    channel_to_encoder_dim = {"image": 90, "text": lstm_hidden_size * 2}
    fusion_module = AttentionFusionModule(
        channel_to_encoder_dim=channel_to_encoder_dim,
        encoding_projection_dim=64,
    )

    # Classifier
    classifier = MLP(
        in_dim=64, # Should match fusion_module.encoding_projection_dim
        out_dim=num_classes,
        hidden_dims=mlp_hidden_dims,
        activation=nn.ReLU, # Default, can be parameterized if needed
        dropout=mlp_dropout,
        normalization=nn.BatchNorm1d,
    )

    # Late Fusion Model
    model = LateFusion(
        encoders=nn.ModuleDict({"image": image_encoder, "text": text_encoder}),
        fusion_module=fusion_module,
        head_module=classifier,
    )

    return model