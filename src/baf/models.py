from itertools import pairwise

from torch import Tensor, nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        for in_features, out_features in pairwise((input_dim, *hidden_dims)):
            layers.extend([nn.Linear(in_features, out_features), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)
