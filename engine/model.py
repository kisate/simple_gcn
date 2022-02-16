import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import NNConv
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    Convolutional layer.

    Applies continuous kernel-based convolutional operators from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper to the input.

    Gets weight vector from edge features just by applying a linear layer.

    Args:
        edge_dim (int): Size of edge features.
        atom_dim (int): Size of atom features.
        out_dim (int): Size of output vector for each atom.
    """
    def __init__(self, edge_dim: int, atom_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(edge_dim, atom_dim * out_dim)
        self.nnconv = NNConv(atom_dim, out_dim, self.lin)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        return self.nnconv(x, edge_index, edge_attr)


class GCN(torch.nn.Module):
    """
    Graph convolutional network.

    Consists of embedding layers and
    continuous kernel-based convolutional operators from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.

    Args:
        atom_emb_dim (int): Size of atomic number embeddings.
        edge_emb_dim (int): Size of bond type embeddings.
        atom_hidden_dim (int): Size of atom hidden feature.
        output_dim (int): Size of output vector for each node.
        max_atomic_number (int): Maximum atomic number in input (100 should be enough).
        edge_types_num (int): Number of bond types in rdkit.
        n_inner_layers (int): Number of convolutional layers besides the first and the last.
    """
    def __init__(self, atom_emb_dim: int, edge_emb_dim: int, atom_hidden_dim: int, output_dim: int,
                 max_atomic_number: int = 100, edge_types_num: int = 22, n_inner_layers: int = 0):
        super().__init__()
        self.atom_emb = nn.Embedding(max_atomic_number + 1, atom_emb_dim, 0)
        self.edge_emb = nn.Embedding(edge_types_num, edge_emb_dim, 0)
        
        self.layer1 = ConvLayer(edge_emb_dim, atom_emb_dim, atom_hidden_dim)

        self.inner_layers = nn.ModuleList([ConvLayer(edge_emb_dim, atom_hidden_dim, atom_hidden_dim) for _ in range(n_inner_layers)])

        self.layer2 = ConvLayer(edge_emb_dim, atom_hidden_dim, output_dim)

    def forward(self, inputs: Data):
        x, edge_index, edge_attr = inputs.x, inputs.edge_index, inputs.edge_attr
        x = self.atom_emb(x).squeeze(1)
        edge_attr = self.edge_emb(edge_attr).squeeze(1)

        x = self.layer1(x, edge_index, edge_attr)
        x = F.relu(x)

        for layer in self.inner_layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)

        x = self.layer2(x, edge_index, edge_attr)

        return x
