import torch
from rdkit.Chem import Mol
from torch_geometric.data import Data


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