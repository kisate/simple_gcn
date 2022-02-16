import argparse

import pandas as pd
from pytorch_lightning import Trainer
from rdkit.Chem import PandasTools

from engine.model import GCN
from engine.training import TrainingModule, TrainingDataModule
from engine.utils import rdkit_to_pyg
from pytorch_lightning.callbacks import EarlyStopping


def train(data_path: str, gpus: int = 1):
    df = pd.read_csv(data_path)
    PandasTools.AddMoleculeColumnToFrame(df, "smiles", "Molecule")

    data = [
        rdkit_to_pyg(mol, expt) for mol, expt in df[["Molecule", "expt"]].values
    ]

    model = GCN(8, 8, 32, 1)

    training_module = TrainingModule(model)
    datamodule = TrainingDataModule(data)

    early_stopping = EarlyStopping('val_r2', mode="max", patience=5)

    trainer = Trainer(gpus=gpus, callbacks=early_stopping, check_val_every_n_epoch=100)
    trainer.fit(training_module, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="SAMPL.csv")
    args = parser.parse_args()

    train(args.data_path)
