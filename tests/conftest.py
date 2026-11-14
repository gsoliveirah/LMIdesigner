"""Fixtures compartilhadas (dataset e modelo pequenos, escopo de sessão)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import TrainingConfig
from src.data_pipeline.calphad_gen import CalphadDataGenerator
from src.data_pipeline.preprocessing import ProcessedDataset, preprocess
from src.models.pinn import PhysicsInformedNet
from src.models.train import HyperParams, PINNTrainer


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return CalphadDataGenerator(seed=7).generate(n_alloys=150, n_process_per_alloy=3)


@pytest.fixture(scope="session")
def dataset(synthetic_df: pd.DataFrame) -> ProcessedDataset:
    return preprocess(synthetic_df, save_dir=None)


@pytest.fixture(scope="session")
def trained_model(dataset: ProcessedDataset) -> PhysicsInformedNet:
    trainer = PINNTrainer(TrainingConfig(max_epochs=80, patience=80, batch_size=128))
    return trainer.fit(dataset.X, dataset.Y, HyperParams(hidden_dim=64, n_layers=2),
                       P_tr=dataset.phases).model
