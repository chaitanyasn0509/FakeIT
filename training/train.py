"""Command-line training entry point for reconstruction models."""

from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl
import typer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import (
    CSVLogger,
    TensorBoardLogger,
    WandbLogger,
)

from common.config import load_config
from common.logging import configure_logging
from models.factory import create_model
from training.datamodule import ReconstructionDataModule

app = typer.Typer(help="Train FakeIT cloud removal models.")


@app.command()
def train(
    config_path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
    ),
    model_name: str | None = typer.Option(
        None,
        "--model",
    ),
    resume: Path | None = typer.Option(
        None,
        "--resume",
    ),
):
    """Train a reconstruction model."""

    configure_logging()

    config = load_config(config_path)

    if model_name:
        config["model"]["name"] = model_name

    if resume:
        config["training"]["resume_from_checkpoint"] = str(resume)

    pl.seed_everything(
        int(config["project"].get("seed", 42)),
        workers=True,
    )

    datamodule = ReconstructionDataModule(config)

    model = create_model(config)

    trainer = create_trainer(config)

    print("=" * 60)
    print("Starting Training...")
    print("=" * 60)

    print("Starting training...")

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config["training"].get("resume_from_checkpoint"),
    )

    print("Training completed.")

    print("=" * 60)
    print("Training Finished")
    print("=" * 60)


def create_trainer(config: dict) -> pl.Trainer:

    training = config["training"]

    checkpoint_dir = Path(
        training.get("checkpoint_dir", "checkpoints")
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False,
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=20,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    loggers = [

        TensorBoardLogger(
            save_dir=training.get("log_dir", "runs"),
            name=config["model"]["name"],
        ),

        CSVLogger(
            save_dir=training.get("log_dir", "runs"),
            name="csv_logs",
        ),
    ]

    if os.getenv("WANDB_API_KEY"):

        loggers.append(
            WandbLogger(
                project=os.getenv(
                    "WANDB_PROJECT",
                    "FakeIT",
                ),
                name=config["model"]["name"],
            )
        )

    trainer = pl.Trainer(

        accelerator=training.get(
            "accelerator",
            "auto",
        ),

        devices=training.get(
            "devices",
            "auto",
        ),

        precision=training.get(
            "precision",
            "16-mixed",
        ),

        max_epochs=int(
            training.get(
                "max_epochs",
                50,
            )
        ),

        callbacks=callbacks,

        logger=loggers,

        log_every_n_steps=1,

        enable_checkpointing=True,

        enable_progress_bar=True,

        deterministic=False,
    )

    return trainer


def main():
    app()


if __name__ == "__main__":
    main()