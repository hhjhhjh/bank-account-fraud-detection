"""FT-Transformer for the shared, preprocessed scalar inputs (see README for scope)."""

import math

import torch
from torch import Tensor, nn


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, input_dim: int, d_token: int):
        super().__init__()
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.empty(input_dim, d_token))
        self.bias = nn.Parameter(torch.empty(input_dim, d_token))
        for parameter in (self.weight, self.bias):
            nn.init.uniform_(parameter, -d_token**-0.5, d_token**-0.5)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"Expected features with shape (batch, {self.input_dim})")
        return features.unsqueeze(-1) * self.weight + self.bias


class ReGLU(nn.Module):
    def forward(self, features: Tensor) -> Tensor:
        values, gates = features.chunk(2, dim=-1)
        return values * gates.relu()


class FTTransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_hidden, dropout, residual_dropout, *, first):
        super().__init__()
        # The original FT-Transformer omits pre-normalization for the first attention.
        self.attention_norm = nn.Identity() if first else nn.LayerNorm(d_token)
        self.attention = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        with torch.no_grad():
            for projection in self.attention.in_proj_weight.chunk(3):
                nn.init.kaiming_uniform_(projection, a=math.sqrt(5))
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_token), nn.Linear(d_token, 2 * ffn_hidden),
            ReGLU(), nn.Dropout(dropout), nn.Linear(ffn_hidden, d_token),
        )
        self.attention_residual_dropout = nn.Dropout(residual_dropout)
        self.ffn_residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.attention_residual_dropout(attended)
        return tokens + self.ffn_residual_dropout(self.ffn(tokens))


class FTTransformer(nn.Module):
    def __init__(
        self, input_dim: int, d_token: int, n_heads: int, num_blocks: int,
        ffn_hidden: int, dropout: float, residual_dropout: float = 0.0,
    ):
        super().__init__()
        for name, value in dict(
            input_dim=input_dim, d_token=d_token, n_heads=n_heads,
            num_blocks=num_blocks, ffn_hidden=ffn_hidden,
        ).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if d_token % n_heads:
            raise ValueError("d_token must be divisible by n_heads")
        if not 0 <= dropout < 1 or not 0 <= residual_dropout < 1:
            raise ValueError("dropout and residual_dropout must be in [0, 1)")
        self.tokenizer = NumericalFeatureTokenizer(input_dim, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.uniform_(self.cls_token, -d_token**-0.5, d_token**-0.5)
        self.blocks = nn.ModuleList([
            FTTransformerBlock(d_token, n_heads, ffn_hidden, dropout, residual_dropout, first=index == 0)
            for index in range(num_blocks)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, features: Tensor) -> Tensor:
        tokens = self.tokenizer(features)
        tokens = torch.cat((tokens, self.cls_token.expand(len(features), -1, -1)), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, -1]).squeeze(-1)
