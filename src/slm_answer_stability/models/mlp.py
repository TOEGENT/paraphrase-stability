"""Small probability regressor for frozen question and answer embeddings."""

from __future__ import annotations


def build_model(embedding_dim: int, hidden_dim: int, dropout: float):
    from torch import nn

    if embedding_dim < 1 or hidden_dim < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid MLP dimensions or dropout")
    return nn.Sequential(
        nn.Linear(2 * embedding_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
