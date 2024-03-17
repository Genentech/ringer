from __future__ import annotations

import functools
import inspect
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F
from transformers.models.bert import configuration_bert, modeling_bert
from transformers.optimization import get_linear_schedule_with_warmup

from .. import data
from ..utils import featurization, losses
from .components import embeddings, output

LR_SCHEDULE = Optional[Literal["LinearWarmup"]]
TIME_ENCODING = Literal["gaussian_fourier", "sinusoidal"]
LOSS_KEYS = Literal["l1", "smooth_l1"]
DECODER_HEAD = Literal["mlp", "linear"]


class BertForDiffusionBase(modeling_bert.BertPreTrainedModel):
    """BERT designed to be used with continuous inputs instead of tokens.

    Reference: https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/bert/modeling_bert.py#L870

    Decoder: linear = single linear decoding of per-position embeddings
             mlp = two-layer MLP to decode per-position embeddings

    This is the base model object and does _not_ include the pytorch lightning code
    """

    # Define loss functions and their wrapped angular versions
    nonangular_loss_fn_dict = {
        "l1": F.l1_loss,
        "smooth_l1": F.smooth_l1_loss,
    }
    angular_loss_fn_dict = {
        "l1": losses.radian_l1_loss,
        "smooth_l1": functools.partial(losses.radian_smooth_l1_loss, beta=torch.pi / 10),
    }

    def __init__(
        self,
        config,
        ft_is_angular: Sequence[bool] = (False, True, True),
        ft_names: Optional[Sequence[str]] = None,
        time_encoding: TIME_ENCODING = "gaussian_fourier",
        decoder: DECODER_HEAD = "mlp",
        atom_feature_size: Optional[int] = None,
        atom_feature_embed_size: Optional[int] = None,
    ) -> None:
        super().__init__(config)

        self.config = config
        if self.config.is_decoder:
            raise NotImplementedError
        self.ft_is_angular = ft_is_angular
        n_inputs = len(ft_is_angular)
        self.n_inputs = n_inputs

        if ft_names is not None:
            self.ft_names = ft_names
        else:
            self.ft_names = [f"ft{i}" for i in range(n_inputs)]
        assert (
            len(self.ft_names) == n_inputs
        ), f"Got {len(self.ft_names)} names, expected {n_inputs}"

        # Needed to project the low dimensional input to hidden dim
        self.inputs_to_hidden_dim = nn.Linear(
            in_features=n_inputs, out_features=config.hidden_size
        )
        if atom_feature_size is not None:
            self.use_atom_embeddings = False  # Atom features have enough information
            # The total hidden dim will be hidden_size + atom_feature_embed_size
            assert atom_feature_embed_size is not None
            self.atom_features_to_hidden_dim = nn.Linear(
                in_features=atom_feature_size, out_features=atom_feature_embed_size
            )
            hidden_size = config.hidden_size + atom_feature_embed_size
            logging.info(
                f"Expanding hidden size from {config.hidden_size} to {hidden_size} to accommodate atom features"
            )
            config.hidden_size = hidden_size
        else:
            self.use_atom_embeddings = True

        self.embeddings = embeddings.BertEmbeddings(
            config, use_atom_embeddings=self.use_atom_embeddings
        )
        self.encoder = modeling_bert.BertEncoder(config)

        # Set up the network to project token representation to the number of outputs
        if decoder == "linear":
            self.token_decoder = nn.Linear(config.hidden_size, n_inputs)
        elif decoder == "mlp":
            self.token_decoder = output.AnglesPredictor(config.hidden_size, n_inputs)
        else:
            raise ValueError(f"Unrecognized decoder: {decoder}")

        # Set up the time embedder
        if time_encoding == "gaussian_fourier":
            self.time_embed = embeddings.GaussianFourierProjection(config.hidden_size)
        elif time_encoding == "sinusoidal":
            self.time_embed = embeddings.SinusoidalPositionEmbeddings(config.hidden_size)
        else:
            raise ValueError(f"Unknown time encoding: {time_encoding}")
        pl.utilities.rank_zero_info(f"Using time embedding: {self.time_embed}")

        # Initialize weights and apply final processing
        self.init_weights()

        # Epoch counters and timers
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()

    @classmethod
    def from_dir(
        cls,
        dir_name: Union[str, Path],
        ft_is_angular: Optional[Sequence[bool]] = None,
        load_weights: bool = True,
        idx: int = -1,
        best_by: Literal["train", "valid"] = "valid",
        copy_to: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> BertForDiffusionBase:
        """Builds this model out from directory.

        idx indicates which model to load if multiple are given.
        """
        dir_name = Path(dir_name)
        train_args_fname = dir_name / "training_args.json"
        with open(train_args_fname, "r") as source:
            train_args = json.load(source)
        config = configuration_bert.BertConfig.from_json_file(dir_name / "config.json")

        ft_names = data.DATASET_CLASSES[
            train_args["internal_coordinates_definitions"]
        ].feature_names
        logging.info(f"Feature names: {ft_names}")
        if ft_is_angular is None:
            ft_is_angular = data.DATASET_CLASSES[
                train_args["internal_coordinates_definitions"]
            ].feature_is_angular
            logging.info(f"Auto constructed ft_is_angular: {ft_is_angular}")

        model_args = dict(
            config=config,
            ft_is_angular=ft_is_angular,
            ft_names=ft_names,
            time_encoding=train_args["time_encoding"],
            decoder=train_args["decoder"],
            **kwargs,
        )

        if "use_atom_features" in train_args and train_args["use_atom_features"]:
            # Featurize dummy mol to get atom feature size
            dummy_mol = Chem.MolFromSmiles("C1CC1")
            dummy_features = featurization.featurize_macrocycle_atoms(
                dummy_mol,
                macrocycle_idxs=[0, 1, 2],
                use_peptide_stereo=False,  # Doesn't change the dimensionality
                include_side_chain_fingerprint=True,
                radius=train_args["atom_feature_fingerprint_radius"],
                size=train_args["atom_feature_fingerprint_size"],
            )

            # If we trained with atom features, hidden_size was expanded in config, so we need to
            # reduce it back down here so that it gets expanded to the correct size in __init__
            config.hidden_size -= train_args["atom_feature_embed_size"]

            model_args.update(
                dict(
                    atom_feature_size=dummy_features.shape[-1],
                    atom_feature_embed_size=train_args["atom_feature_embed_size"],
                )
            )

        if load_weights:

            def epoch_getter(path: Union[str, Path]) -> int:
                path = Path(path)
                epoch = int(re.findall(r"epoch=[0-9]+", path.name).pop().split("=")[-1])
                return epoch

            ckpt_dir_name = f"best_by_{best_by}"
            ckpt_dir = dir_name / "models" / ckpt_dir_name
            # Sort checkpoints by epoch -- last item is latest epoch
            ckpt_names = sorted(ckpt_dir.glob("*.ckpt"), key=epoch_getter)
            logging.info(f"Found {len(ckpt_names)} checkpoints")
            ckpt_name = ckpt_names[idx]
            logging.info(f"Loading weights from {ckpt_name}")
            if hasattr(cls, "load_from_checkpoint"):
                # Defined for pytorch lightning module
                retval = cls.load_from_checkpoint(checkpoint_path=ckpt_name, **model_args)
            else:
                retval = cls(**model_args)
                loaded = torch.load(ckpt_name, map_location=torch.device("cpu"))
                retval.load_state_dict(loaded["state_dict"])
            retval.train_epoch_counter = epoch_getter(ckpt_name)
        else:
            retval = cls(**model_args)
            logging.info(f"Loaded uninitialized model from {dir_name}")

        # If specified, copy out the requisite files to the given directory
        if copy_to is not None:
            logging.info(f"Copying minimal model file set to: {copy_to}")
            copy_to = Path(copy_to)
            copy_to.mkdir(parents=True, exist_ok=True)
            with open(copy_to / "training_args.json", "w") as sink:
                json.dump(train_args, sink, indent=4)
            config.save_pretrained(copy_to)
            if load_weights:
                # Create the directory structure
                ckpt_dir_copy = copy_to / "models" / ckpt_dir_name
                ckpt_dir_copy.mkdir(parents=True, exist_ok=True)
                shutil.copy(ckpt_name, ckpt_dir_copy)

        return retval

    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,  # Tensor of shape batch_length with time indices
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        atom_ids: Optional[torch.Tensor] = None,
        atom_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        input_shape = inputs.size()
        batch_size, seq_length, *_ = input_shape
        logging.debug(f"Detected batch {batch_size} and seq length {seq_length}")

        if seq_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Input tensor with sequence length {seq_length} is longer than the maximum length {self.config.max_position_embeddings}"
            )

        # If position IDs are not given, auto-generate them
        if position_ids is None:
            # [1, seq_length]
            position_ids = torch.arange(seq_length).expand(batch_size, -1).type_as(timestep)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads. This code is taken
        # from hugggingface modeling_utils
        assert attention_mask is not None
        assert (
            attention_mask.dim() == 2
        ), f"Attention mask expected in shape (batch_size, seq_length), got {attention_mask.shape}"
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.type_as(attention_mask)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        assert len(inputs.shape) == 3  # [batch x seq_len x num_feat]
        inputs_upscaled = self.inputs_to_hidden_dim(inputs)  # [batch x seq_len x input_hidden_dim]

        if atom_features is not None:
            assert len(atom_features.shape) == 3  # [batch x seq_len x num_atom_feat]
            atom_features_upscaled = self.atom_features_to_hidden_dim(
                atom_features
            )  # [batch x seq_len x atom_feat_embed_dim]

            inputs_upscaled = torch.cat(
                (inputs_upscaled, atom_features_upscaled), dim=-1
            )  # [batch x seq_len x hidden_dim]

        # Pass through embeddings
        inputs_upscaled = self.embeddings(
            inputs_upscaled, position_ids=position_ids, atom_ids=atom_ids
        )

        # timestep is (batch, 1), squeeze to (batch,)
        # embedding gets to (batch, embed_dim) -> unsqueee to (batch, 1, dim)
        time_encoded = self.time_embed(timestep.squeeze(dim=-1)).unsqueeze(1)
        inputs_with_time = inputs_upscaled + time_encoded
        encoder_outputs = self.encoder(
            inputs_with_time,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        sequence_output = encoder_outputs[0]
        per_token_decoded = self.token_decoder(sequence_output)

        if output_attentions or output_hidden_states:
            return (per_token_decoded,) + encoder_outputs[1:]

        return per_token_decoded


class BertForDiffusion(BertForDiffusionBase, pl.LightningModule):
    """Wraps our model as a pl LightningModule for easy training."""

    def __init__(
        self,
        lr: float = 5e-5,
        loss: Union[Callable, LOSS_KEYS] = "smooth_l1",
        use_feat_mask: bool = True,  # if available
        l2: float = 0.0,
        l1: float = 0.0,
        circle_reg: float = 0.0,
        epochs: int = 1,
        warmup_epochs: int = 0,
        lr_scheduler: LR_SCHEDULE = None,
        write_preds_to_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        BertForDiffusionBase.__init__(self, **kwargs)

        # Store information about learning rates and loss
        self.learning_rate = lr
        if isinstance(loss, str):
            logging.info(
                f"Mapping loss {loss} to list of losses corresponding to angular {self.ft_is_angular}"
            )
            self.loss_func = [
                self.angular_loss_fn_dict[loss]
                if is_angular
                else self.nonangular_loss_fn_dict[loss]
                for is_angular in self.ft_is_angular
            ]
        else:
            logging.warning(
                f"Using pre-given callable loss: {loss}. This may not handle angles correctly!"
            )
            self.loss_func = loss
        pl.utilities.rank_zero_info(f"Using loss: {self.loss_func}")
        if isinstance(self.loss_func, (tuple, list)):
            assert (
                len(self.loss_func) == self.n_inputs
            ), f"Got {len(self.loss_func)} loss functions, expected {self.n_inputs}"
        self.use_feat_mask = use_feat_mask

        self.l1_lambda = l1
        self.l2_lambda = l2
        self.circle_lambda = circle_reg
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr_scheduler = lr_scheduler

        # Set up the output directory for writing predictions
        self.write_preds_to_dir = write_preds_to_dir
        self.write_preds_counter = 0
        if self.write_preds_to_dir is not None:
            self.write_preds_to_dir = Path(self.write_preds_to_dir)
            self.write_preds_to_dir.mkdir(parents=True, exist_ok=True)

    def _get_loss_terms(
        self, batch: Dict[str, torch.Tensor], write_preds: Optional[Path] = None
    ) -> torch.Tensor:
        """Returns the loss terms for the model.

        Length of the returned list is equivalent to the number of features we are fitting to.
        """
        known_noise = batch["known_noise"]
        predicted_noise = self.forward(
            batch["corrupted"],
            batch["t"],
            attention_mask=batch["attn_mask"],
            position_ids=batch["position_ids"],
            atom_ids=batch["atom_ids"],
            atom_features=batch["atom_features"] if "atom_features" in batch else None,
        )
        assert (
            known_noise.shape == predicted_noise.shape
        ), f"{known_noise.shape} != {predicted_noise.shape}"

        # Indexes into batch then indexes along sequence length
        # attn_mask has shape (batch, seq_len) --> where gives back
        # two lists of values, one for each dimension
        # known_noise has shape (batch, seq_len, num_fts)
        unmask_idx = torch.where(batch["attn_mask"])
        assert len(unmask_idx) == 2
        loss_terms = []
        for i in range(known_noise.shape[-1]):
            # feat_mask masks feature dimension and sequence dimension,
            # so it can replace attention mask
            if self.use_feat_mask and "feat_mask" in batch:
                unmask_idx = torch.where(batch["feat_mask"][..., i])
            loss_fn = self.loss_func[i] if isinstance(self.loss_func, list) else self.loss_func
            logging.debug(f"Using loss function {loss_fn}")
            # Determine whether the loss accepts circle_penalty
            # https://stackoverflow.com/questions/23228664/how-to-check-which-arguments-a-function-method-takes
            loss_args = inspect.getfullargspec(loss_fn)
            if "circle_penalty" in loss_args.args or "circle_penalty" in loss_args.kwonlyargs:
                logging.debug(f"Loss function {loss_fn} accepts circle_penalty")
                loss = loss_fn(
                    predicted_noise[unmask_idx[0], unmask_idx[1], i],
                    known_noise[unmask_idx[0], unmask_idx[1], i],
                    circle_penalty=self.circle_lambda,
                )
            else:
                logging.debug(f"Loss function {loss_fn} does not accept circle_penalty")
                loss = loss_fn(
                    predicted_noise[unmask_idx[0], unmask_idx[1], i],
                    known_noise[unmask_idx[0], unmask_idx[1], i],
                )
            loss_terms.append(loss)

        if write_preds is not None:
            with open(write_preds, "w") as f:
                d_to_write = {
                    "known_noise": known_noise.cpu().numpy().tolist(),
                    "predicted_noise": predicted_noise.cpu().numpy().tolist(),
                    "attn_mask": batch["attn_mask"].cpu().numpy().tolist(),
                    "losses": [loss.item() for loss in loss_terms],
                }
                json.dump(d_to_write, f)

        return torch.stack(loss_terms)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step, runs once per batch."""
        loss_terms = self._get_loss_terms(batch)

        if "weights" in batch:
            weights = batch["weights"][0]  # Same for each item in batch
            nan_mask = ~loss_terms.isnan()
            loss_terms_masked = loss_terms[nan_mask]
            weights_masked = weights[nan_mask]
            avg_loss = torch.dot(weights_masked, loss_terms_masked) / torch.sum(weights_masked)
        else:
            avg_loss = torch.nanmean(loss_terms)

        # L1 loss implementation
        if self.l1_lambda > 0:
            l1_penalty = sum(torch.linalg.norm(p, 1) for p in self.parameters())
            avg_loss += self.l1_lambda * l1_penalty

        assert len(loss_terms) == len(self.ft_names)
        loss_dict = {
            f"train_loss_{val_name}": val for val_name, val in zip(self.ft_names, loss_terms)
        }
        loss_dict["train_loss"] = avg_loss
        self.log_dict(loss_dict)  # Don't seem to need rank zero or sync dist

        return avg_loss

    def training_epoch_end(self, outputs: Sequence[Dict[str, torch.Tensor]]) -> None:
        """Log the average training loss over the epoch."""
        losses = torch.stack([o["loss"] for o in outputs])
        mean_loss = torch.nanmean(losses)
        t_delta = time.time() - self.train_epoch_last_time
        pl.utilities.rank_zero_info(
            f"Train loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f} ({t_delta:.2f} seconds)"
        )
        # Increment counter and timers
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        with torch.no_grad():
            loss_terms = self._get_loss_terms(
                batch,
                write_preds=None
                if self.write_preds_to_dir is None
                else self.write_preds_to_dir / f"{self.write_preds_counter}_preds.json",
            )
            self.write_preds_counter += 1

        if "weights" in batch:
            weights = batch["weights"][0]  # Same for each item in batch
            nan_mask = ~loss_terms.isnan()
            loss_terms_masked = loss_terms[nan_mask]
            weights_masked = weights[nan_mask]
            avg_loss = torch.dot(weights_masked, loss_terms_masked) / torch.sum(weights_masked)
        else:
            avg_loss = torch.nanmean(loss_terms)

        # Log each of the loss terms
        assert len(loss_terms) == len(self.ft_names)
        loss_dict = {
            f"val_loss_{val_name}": self.all_gather(val)
            for val_name, val in zip(self.ft_names, loss_terms)
        }
        loss_dict["val_loss"] = avg_loss
        # With rank zero it seems that we don't need to use sync_dist
        self.log_dict(loss_dict, rank_zero_only=True)

        return {"val_loss": avg_loss}

    def validation_epoch_end(self, outputs: Sequence[Dict[str, torch.Tensor]]) -> None:
        """Log the average validation loss over the epoch."""
        # Note that this method is called before zstraining_epoch_end().
        losses = torch.stack([o["val_loss"] for o in outputs])
        mean_loss = torch.nanmean(losses)
        pl.utilities.rank_zero_info(
            f"Valid loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f}"
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Return optimizer.

        Limited support for some optimizers.
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optim}
        if self.lr_scheduler:
            if self.lr_scheduler == "LinearWarmup":
                # https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
                # Transformers typically do well with linear warmup
                pl.utilities.rank_zero_info(
                    f"Using linear warmup with {self.warmup_epochs}/{self.epochs} warmup epochs"
                )
                retval["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optim,
                        num_warmup_steps=self.warmup_epochs,
                        num_training_steps=self.epochs,
                    ),
                    "frequency": 1,
                    "interval": "epoch",  # Call after 1 epoch
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval
