import argparse
from typing import List

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from rdkit.Chem import PandasTools
from torch_geometric.data import Data

from engine.model import GCN
from engine.training import TrainingModule, TrainingDataModule
from engine.utils import rdkit_to_pyg


def read_data(data_path: str) -> List[Data]:
    df = pd.read_csv(data_path)
    PandasTools.AddMoleculeColumnToFrame(df, "smiles", "Molecule")
    return [
        rdkit_to_pyg(mol, expt) for mol, expt in df[["Molecule", "expt"]].values
    ]


def train(network: GCN, training_data: List[Data], gpus: int = 1) -> GCN:
    training_module = TrainingModule(network)
    datamodule = TrainingDataModule(training_data)

    early_stopping = EarlyStopping('val_loss', mode="min", patience=5)

    trainer = Trainer(gpus=gpus, callbacks=early_stopping, check_val_every_n_epoch=100)
    trainer.fit(training_module, datamodule)

    return network


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="SAMPL.csv")
    args = parser.parse_args()

    model = GCN(8, 8, 32, 1)

    data = read_data(args.data_path)

    train(model, data)
