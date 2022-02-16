from typing import List, Dict

import torch
from rdkit.Chem import Mol
from torch_geometric.data import Data

import matplotlib.pyplot as plt

def rdkit_to_pyg(mol: Mol, expt: float) -> Data:
    """
    Helper method to convert rdkit molecule to pyg format

    :param mol: Mol from rdkit
    :param expt: target float
    :return: pyg Data object
    """
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # Make adjacency list and edge types
    edges = []
    edge_types = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        bond_type = int(bond.GetBondType())

        edges.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
        edge_types.extend([[bond_type], [bond_type]])

    edge_index = torch.LongTensor(edges).t().contiguous()

    if not edge_index.any():
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    edge_features = torch.LongTensor(edge_types)
    node_features = torch.LongTensor(atoms).reshape(-1, 1)

    target = torch.FloatTensor([[expt]])

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, target=target)


def plot_metrics(training_metrics: List[Dict], val_metrics: List[Dict]):
    metric_names = list(training_metrics[0].keys())
    combined_training_metrics = {name: [x[name] for x in training_metrics] for name in metric_names}
    combined_val_metrics = {name: [x[name] for x in val_metrics] for name in metric_names}

    fig, ax = plt.subplots(1, len(metric_names), figsize=(13, 5))

    for idx, name in enumerate(metric_names):
        val_x_axis = torch.linspace(0, len(combined_training_metrics[name]), steps=len(combined_val_metrics[name]))

        ax[idx].plot(combined_training_metrics[name], label="training")
        ax[idx].plot(val_x_axis, combined_val_metrics[name], label="val")
        ax[idx].set_title(name)
        ax[idx].legend()

    plt.show()
