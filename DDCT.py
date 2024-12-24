######################################################### DDCT #########################################################
# Name : Double Domain Converter Transformer
# Time : 2024.12.24
# Author : Junyu Pan
# Affiliation : Shanghai Jiao Tong University
# Conference : ICASSP 2025
#######################################################################################################################

import torch
import torch.nn.functional as F
from torch import nn
import math
from functions import BasicBlock, ReverseLayerF


class Convert_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_decoder = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=5, kernel_size=1)
        )

    def forward(self, x):
        # x.shape=(n,5,64)
        return self.encoder_decoder(x)


class Convert_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=320, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=320),
        )

    def forward(self, x):
        # x.shape=(n,5,64)
        return self.encoder_decoder(x).reshape(-1, 5, 64)


class cnn_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.padding = nn.ZeroPad3d(padding=(1, 1, 0, 0, 0, 0))
        self.conv_1 = BasicBlock(in_channels=5, out_channels=64)

        self.conv_2 = BasicBlock(in_channels=64, out_channels=32)
        self.conv_3 = BasicBlock(in_channels=32, out_channels=16)
        self.extractor = nn.Sequential(
            self.conv_1,
            nn.MaxPool1d(kernel_size=2),
            self.conv_2,
            nn.AvgPool1d(kernel_size=2),
            self.conv_3,
            nn.BatchNorm1d(16)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_pad = self.padding(x)
        x_feature = self.extractor(x_pad)
        x_flatten = self.flatten(x_feature)
        return x_flatten


def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value)


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query, key, value):
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)
        return scaled_dot_product_attention(query, key, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.dim_in = dim_in
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.dim_q = dim_q
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, self.dim_q, self.dim_k) for _ in range(self.n_heads)]
        )
        self.linear = nn.Linear(self.n_heads * self.dim_k, dim_in)

    def forward(self, query, key, value):
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )


def position_encoding(seq_len: int, dim_model: int, device: torch.device = None):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048):
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        # nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input)
    )


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: torch.Tensor):
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model=512, num_heads=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout
        )
        self.feedforward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: torch.Tensor):
        src = self.attention(src, src, src)
        return self.feedforward(src)


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            dim_model: int = 28,
            num_heads: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.device = device

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension, device=self.device)
        for layer in self.layers:
            src = layer(src)
        return src


class DDCT(nn.Module):
    def __init__(self, num_class: int, num_domain: int, num_layers: int, num_heads: int,
                 dim_feedforward: int, convert_type: str, device: torch.device):
        super(DDCT, self).__init__()
        self.padding = nn.ZeroPad3d(padding=(1, 1, 0, 0, 0, 0))
        self.num_class = num_class
        self.num_domain = num_domain
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.device = device
        if convert_type == 'cnn':
            self.convert_to_source = Convert_cnn()
            self.convert_to_target = Convert_cnn()
        elif convert_type == 'mlp':
            self.convert_to_source = Convert_mlp()
            self.convert_to_target = Convert_mlp()
        self.extractor = TransformerEncoder(num_layers=num_layers, dim_model=5 * 64,
                                            num_heads=num_heads, dim_feedforward=dim_feedforward, device=device)
        self.source_classifier = nn.Sequential(
            nn.Linear(in_features=5 * 64, out_features=num_domain),
            nn.ReLU()
        )
        self.target_classifier = nn.Sequential(
            nn.Linear(in_features=5 * 64, out_features=num_domain),
            nn.ReLU()
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(in_features=2 * 5 * 64, out_features=num_class),
            nn.ReLU()
        )

    def forward(self, data, alpha):
        # data.size: (n,5,62)
        # alpha: value for ReverseLayer
        # class_out: emotion label
        # source_rebuilt: output of source data after target converter and source converter in turn.
        #                 Only use in source domain learning stage.
        # target rebuilt: output of target data after source converter and target converter in turn.
        #                 Only use in target domain learning stage.
        # source_out: output of source classifier which judge input data whether is in source domain.
        # target_out: output of target classifier which judge input data whether is in target domain.
        data = self.padding(data)
        n = data.shape[0]
        data_source = self.convert_to_source(data)
        data_target = self.convert_to_target(data)
        data_source_reverse = ReverseLayerF.apply(data_source.view(n, -1), alpha)
        data_target_reverse = ReverseLayerF.apply(data_target.view(n, -1), alpha)
        source_rebuilt = self.convert_to_source(data_target)
        target_rebuilt = self.convert_to_target(data_source)
        source_out = self.source_classifier(data_source_reverse)
        target_out = self.target_classifier(data_target_reverse)
        data_source_feature = torch.cat([data_source.view(n, 1, -1), data_target.view(n, 1, -1)], dim=1)
        feature = self.extractor(data_source_feature)
        class_out = self.class_classifier(feature.view(n, -1))
        return class_out, source_rebuilt, target_rebuilt, source_out, target_out


if __name__ == '__main__':
    x = torch.randn(32, 5, 62)
    model = DDCT(num_class=2, num_domain=2, num_layers=2, num_heads=2, dim_feedforward=512, convert_type='mlp',
                 device=torch.device("cpu"))
    class_out, source_rebuilt, target_rebuilt, source_out, target_out = model(x, 0)
    print(class_out.shape)  # (32,2)
    print(source_rebuilt.shape)  # (32,5,64)
    print(target_rebuilt.shape)  # (32,5,64)
    print(source_out.shape)  # (32,2)
    print(target_out.shape)  # (32,2)
