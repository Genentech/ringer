#!/usr/bin/env python

import argparse
import json
import logging
import multiprocessing
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader
from transformers.models.bert import configuration_bert

import ringer
from ringer.models import bert_for_diffusion
from ringer.utils import data_loading, utils, variance_schedules

POSITION_EMBEDDING_TYPES = Literal[
    "absolute", "relative_key", "relative_key_query", "cyclic_relative_key"
]

assert torch.cuda.is_available(), "Requires CUDA to train"
torch.manual_seed(6489)
torch.backends.cudnn.benchmark = False


@pl.utilities.rank_zero_only
def record_args(func_args: Dict[str, Any], results_dir: Path, overwrite: bool = False) -> None:
    # Create results directory
    if results_dir.exists():
        if overwrite:
            logging.warning(f"Removing old results directory: {results_dir}")
            shutil.rmtree(results_dir)
        else:
            raise IOError(f"'{results_dir}' already exists")
    results_dir.mkdir()

    func_args_serializable = func_args.copy()
    for k, v in func_args_serializable.items():
        if isinstance(v, Path):
            func_args_serializable[k] = str(v)

    with open(results_dir / "training_args.json", "w") as sink:
        logging.info(f"Writing training args to {sink.name}")
        json.dump(func_args_serializable, sink, indent=4)
    for k, v in func_args.items():
        logging.info(f"Training argument: {k}={v}")


def build_callbacks(
    out_dir: Union[str, Path], early_stop_patience: Optional[int] = None, swa: bool = False
) -> List[pl.Callback]:
    # Create the logging dir
    out_dir = Path(out_dir)
    best_validation_dir = out_dir / "models/best_by_valid"
    best_train_dir = out_dir / "models/best_by_train"
    (out_dir / "logs/lightning_logs").mkdir(parents=True, exist_ok=True)
    best_validation_dir.mkdir(parents=True, exist_ok=True)
    best_train_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=best_validation_dir,
            save_top_k=5,
            save_weights_only=True,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="train_loss",
            dirpath=best_train_dir,
            save_top_k=5,
            save_weights_only=True,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    if early_stop_patience is not None and early_stop_patience > 0:
        logging.info(f"Using early stopping with patience {early_stop_patience}")
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val_loss",
                patience=early_stop_patience,
                verbose=True,
                mode="min",
            )
        )

    if swa:
        # Stochastic weight averaging
        callbacks.append(pl.callbacks.StochasticWeightAveraging())
    logging.info(f"Model callbacks: {callbacks}")

    return callbacks


def train(
    # Output
    out_dir: Union[str, Path] = "results",
    # Data loading and noising process
    data_dir: Union[
        str, Path
    ] = "",  # Directory containing pickle files, can be relative to ../data
    split_sizes: Sequence[float] = (0.8, 0.1, 0.1),
    internal_coordinates_definitions: data_loading.INTERNAL_COORDINATES_DEFINITIONS = "angles",
    use_atom_features: bool = True,  # Condition model on atom sequence
    atom_feature_fingerprint_radius: int = 3,  # Morgan fingerprint radius for atom side chains
    atom_feature_fingerprint_size: int = 32,  # Morgan fingerprint size for atom side chains
    atom_feature_embed_size: int = 24,  # Transform atom features to this size before concatenating with angle features
    max_conf: int = 30,
    timesteps: int = 50,
    variance_schedule: variance_schedules.SCHEDULES = "cosine",
    variance_scale: float = 1.0,
    # Model architecture
    time_encoding: bert_for_diffusion.TIME_ENCODING = "gaussian_fourier",
    num_hidden_layers: int = 3,
    hidden_size: int = 24,
    intermediate_size: int = 96,
    num_heads: int = 3,
    position_embedding_type: POSITION_EMBEDDING_TYPES = "cyclic_relative_key",
    dropout_p: float = 0.1,
    decoder: bert_for_diffusion.DECODER_HEAD = "mlp",
    # Training strategy
    batch_size: int = 64,
    loss: bert_for_diffusion.LOSS_KEYS = "smooth_l1",
    l2_norm: float = 0.0,  # AdamW default has 0.01 L2 regularization, but BERT trainer uses 0.0
    l1_norm: float = 0.0,
    circle_reg: float = 0.0,
    gradient_clip: float = 1.0,  # From BERT trainer
    lr: float = 5e-5,  # Default lr for huggingface BERT trainer
    lr_scheduler: bert_for_diffusion.LR_SCHEDULE = "LinearWarmup",
    min_epochs: Optional[int] = None,
    max_epochs: int = 2000,
    warmup_epochs: int = 100,
    early_stop_patience: int = 0,  # Set to 0 to disable early stopping
    use_swa: bool = False,  # Stochastic weight averaging can improve training genearlization
    # Miscellaneous
    exhaustive_validation_t: bool = False,  # Exhaustively enumerate t for validation/test
    use_data_cache: bool = True,
    data_cache_dir: Optional[Union[str, Path]] = None,
    unsafe_cache: bool = False,
    ncpu: int = multiprocessing.cpu_count(),
    ngpu: int = -1,  # -1 for all GPUs
    write_validation_preds: bool = False,  # Write validation predictions to disk at each epoch
    profile: bool = False,
    overwrite: bool = False,  # Overwrite results dir
    wandb_config: Optional[Dict[str, str]] = None,
) -> None:
    """Main training loop."""
    # Record the args given to the function before we create more vars
    func_args = locals()

    assert data_dir
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    record_args(func_args, out_dir, overwrite=overwrite)

    # Get datasets and wrap them in data_loaders
    dsets = data_loading.get_datasets(
        data_dir=data_dir,
        internal_coordinates_definitions=internal_coordinates_definitions,
        splits=["train", "validation"],
        split_sizes=split_sizes,
        use_atom_features=use_atom_features,
        atom_feature_fingerprint_radius=atom_feature_fingerprint_radius,
        atom_feature_fingerprint_size=atom_feature_fingerprint_size,
        max_conf=max_conf,
        timesteps=timesteps,
        variance_schedule=variance_schedule,
        variance_scale=variance_scale,
        exhaustive_t=exhaustive_validation_t,
        use_cache=use_data_cache,
        cache_dir=data_cache_dir,
        unsafe_cache=unsafe_cache,
        num_proc=ncpu,
    )

    # Given total (effective) batch size, calculate batch size per GPU
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count() if ngpu == -1 else ngpu
        batch_size_per_device = max(int(batch_size / device_count), 1)
        pl.utilities.rank_zero_info(
            f"Given batch size: {batch_size} --> per-GPU batch size with {device_count} GPUs: {batch_size_per_device}"
        )
    else:
        batch_size_per_device = batch_size

    data_loaders = {
        split: DataLoader(
            dataset=dset,
            batch_size=batch_size_per_device,
            shuffle=split == "train",
            num_workers=ncpu,
            pin_memory=True,
        )
        for split, dset in dsets.items()
    }

    # Record the means in the output directory
    with open(out_dir / "training_mean_offset.json", "w") as sink:
        json.dump(dsets["train"].dset.means_dict, sink, indent=4)
    with open(out_dir / "training_mean_distances.json", "w") as sink:
        json.dump(dsets["train"].dset.atom_type_means["distance"], sink, indent=4)
    with open(out_dir / "training_mean_angles.json", "w") as sink:
        json.dump(dsets["train"].dset.atom_type_means["angle"], sink, indent=4)
    with open(out_dir / "training_std_angles.json", "w") as sink:
        json.dump(dsets["train"].dset.atom_type_stdevs["angle"], sink, indent=4)

    # Shape of the input is (batch_size, timesteps, features)
    sample_item = dsets["train"][0]
    sample_input = sample_item["corrupted"]
    model_n_inputs = sample_input.shape[-1]
    logging.info(f"Auto detected {model_n_inputs} inputs")

    if use_atom_features:
        sample_atom_features = sample_item["atom_features"]
        atom_feature_size = sample_atom_features.shape[-1]
        logging.info(f"Auto detected atom feature size: {atom_feature_size}")
    else:
        atom_feature_size = None

    logging.info(f"Using loss function: {loss}")
    config = configuration_bert.BertConfig(
        max_position_embeddings=dsets["train"].pad,
        num_attention_heads=num_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        position_embedding_type=position_embedding_type,
        hidden_dropout_prob=dropout_p,
        attention_probs_dropout_prob=dropout_p,
        use_cache=False,
    )
    model = bert_for_diffusion.BertForDiffusion(
        config=config,
        ft_is_angular=dsets["train"].feature_is_angular,
        ft_names=dsets["train"].feature_names,
        time_encoding=time_encoding,
        decoder=decoder,
        atom_feature_size=atom_feature_size,
        atom_feature_embed_size=atom_feature_embed_size,
        lr=lr,
        loss=loss,
        l2=l2_norm,
        l1=l1_norm,
        circle_reg=circle_reg,
        epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        write_preds_to_dir=out_dir / "validation_preds" if write_validation_preds else None,
    )
    config.save_pretrained(out_dir)

    callbacks = build_callbacks(
        out_dir=out_dir, early_stop_patience=early_stop_patience, swa=use_swa
    )

    # Get accelerator and distributed strategy
    accelerator = "cpu"
    strategy = None
    if torch.cuda.is_available():
        accelerator = "cuda"
        if torch.cuda.device_count() > 1:
            # https://github.com/Lightning-AI/lightning/discussions/6761https://github.com/Lightning-AI/lightning/discussions/6761
            strategy = DDPStrategy(find_unused_parameters=False)

    logging.info(f"Using {accelerator} with strategy {strategy}")

    loggers = [pl.loggers.CSVLogger(save_dir=out_dir / "logs")]

    # Set up WandB logging
    if wandb_config is not None:
        wandb_logger = pl.loggers.WandbLogger(**wandb_config)
        if pl.utilities.rank_zero_only.rank == 0:
            wandb_logger.experiment.config.update(func_args)
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        default_root_dir=out_dir,
        gradient_clip_val=gradient_clip,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=min(50, len(data_loaders["train"])),  # Log >= once per epoch
        accelerator=accelerator,
        strategy=strategy,
        gpus=ngpu,
        enable_progress_bar=False,
        move_metrics_to_cpu=False,  # Saves memory
        profiler="simple" if profile else None,
    )
    trainer.fit(
        model=model,
        train_dataloaders=data_loaders["train"],
        val_dataloaders=data_loaders["validation"],
    )


@utils.unwrap_typer_args
def train_from_config(
    config: str = typer.Argument(..., help="JSON file containing training parameters"),
    out_dir: str = typer.Option("results", help="Directory to write model training outputs to"),
    wandb_run: str = typer.Option(None, help="Run name for WandB logging"),
    ncpu: int = typer.Option(multiprocessing.cpu_count(), help="Number of workers"),
    ngpu: int = typer.Option(-1, help="Number of GPUs to use (-1 for all)"),
    unsafe_cache: bool = typer.Option(
        False, help="Don't check data filenames and cache hashes before loading data"
    ),
    profile: bool = False,
    overwrite: bool = typer.Option(False, help="Overwrite output directory"),
) -> None:
    curr_time = datetime.now().strftime("%y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"training_{curr_time}.log"),
            logging.StreamHandler(),
        ],
    )

    config_path = Path(config)
    if not config_path.exists():
        # Assume it's a path relative to the configs folder in the top-level directory
        config_path = ringer.CONFIG_DIR / config_path
        if not config_path.exists():
            raise ValueError(f"Config '{config_path}' doesn't exist")

    with open(config_path) as source:
        config_args = json.load(source)

    wandb_config = None
    if wandb_run is not None:
        wandb_config_path = ringer.CONFIG_DIR / "wandb/wandb.json"
        with open(wandb_config_path) as source:
            wandb_config = json.load(source)
        wandb_config["name"] = wandb_run

    config_args = utils.update_dict_nonnull(
        config_args,
        {
            "out_dir": out_dir,
            "overwrite": overwrite,
            "ncpu": ncpu,
            "ngpu": ngpu,
            "unsafe_cache": unsafe_cache,
            "profile": profile,
            "wandb_config": wandb_config,
        },
    )

    train(**config_args)


def main() -> None:
    typer.run(train_from_config)


if __name__ == "__main__":
    main()
