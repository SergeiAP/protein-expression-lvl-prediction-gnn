import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.utils.data import DataLoader
from torch_geometric.data.data import Data

from typing import Callable

import numpy as np

import matplotlib.pyplot as plt


# Inspired by:   
# 1. https://colab.research.google.com/drive/1N3LvAO0AXV4kBPbTMX866OwJM9YS6Ji2?usp=sharing#scrollTo=fl5W1gg5Jhzz
# 2. https://colab.research.google.com/drive/1udeUfWJzvMlLO7sGUDGsHo8cRPMicajl?usp=sharing#scrollTo=wTR4wQG31Vtk


class GNNEncoder(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 out_channels: int,
                 agg_method: str,
                 dropout: float):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv((-1, -1), hidden_channels, aggr=agg_method)
        self.conv2 = SAGEConv((-1, -1), out_channels, aggr=agg_method)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, in_layers: int, hidden_layers: int):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_layers, hidden_layers)
        self.lin2 = torch.nn.Linear(hidden_layers, 1)

    def forward(self, h):
        h = self.lin1(h).relu()
        h = self.lin2(h)
        return h.view(-1)


class Model(torch.nn.Module):
    """GraphSAGE"""
    def __init__(self,
                sage_hidden: int,
                linear_hidden: int,
                dropout: float,
                seed: int,
                agg_method: str = "mean"):
        super().__init__()
        torch.manual_seed(seed)
        self.dropout = dropout
        self.encoder = GNNEncoder(sage_hidden, sage_hidden, agg_method, dropout)
        self.decoder = EdgeDecoder(sage_hidden, linear_hidden)

        self.total_loss = []
        self.total_score = []
        self.val_loss = []
        self.val_score = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        res = self.decoder(h)
        return res

    def fit(self,
            loader: DataLoader,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.modules.loss,
            scorer: Callable):

        self.total_loss = []
        self.total_score = []
        self.val_loss = []
        self.val_score = []

        self.train()
        for epoch in range(1, epochs+1):

            total_loss = 0
            total_score = 0
            val_loss = 0
            val_score = 0

            # Train on batches
            for batch in loader:
                out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += float(loss)
                total_score += float(scorer(out[batch.train_mask], 
                                            batch.y[batch.train_mask]))
                # Validation
                val_loss += float(criterion(out[batch.val_mask], 
                                            batch.y[batch.val_mask]))
                val_score += float(scorer(out[batch.val_mask], 
                                          batch.y[batch.val_mask]))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.total_loss.append(total_loss / len(loader))
            self.total_score.append(total_score / len(loader))
            self.val_loss.append(val_loss / len(loader))
            self.val_score.append(val_score / len(loader))

            # Print metrics every 10 epochs
            if epoch % 1 == 0:
                print(f'Epoch {epoch:>3} | Train Loss: {self.total_loss[-1]:.3f} '
                        f'| Train Score: {self.total_score[-1]:.3f} | Val Loss: '
                        f'{self.val_loss[-1]:.3f} | Val Score: '
                        f'{self.val_score[-1]:.3f}')


def test_model(model: torch.nn.Module,
               data: Data,
               mask: torch.Tensor,
               scorer: Callable) -> tuple[torch.tensor, float]:
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    out = model(data.x, data.edge_index)
    score = scorer(out[mask], data.y[mask])
    return out[mask], float(score)


def my_plot(epoch: int, train_loss: list[float], val_loss: list[float]):
    """Plot epochs results"""
    epochs = np.linspace(1, epoch, epoch).astype(int)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.legend()
    plt.show()


def mse_loss(output: torch.tensor, target: torch.tensor) -> torch.tensor:
    """the target is transformed by log, therefore get it by exp"""
    loss = torch.mean((torch.exp(output) - torch.exp(target))**2)
    return loss
