# Standard library
import argparse
import json
import logging
import os
import psutil
import random
import time
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

# Third party
import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)

# Import Muon optimizer not yet part of PyTorch, see
#   https://github.com/pytorch/pytorch/issues/148819
#from muon import SingleDeviceMuonWithAuxAdam
# Using the default muon impl from muon_adaptive allows us to validate
# v.s. previous runs with the official Keller James impl.
from muon_adaptive import SingleDeviceMuonWithAuxAdamAdaptive

# Local
import tplr


class Timer:
    """Context manager for timing code blocks."""

    _timings = defaultdict(list)
    _active_timers = {}
    _disable = False

    def __init__(self, name, logger=None, disabled=False, enabled=False):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.disabled = disabled or (Timer._disable and not enabled)

    def __enter__(self):
        if self.disabled:
            return self

        self.start_time = time.perf_counter()
        Timer._active_timers[self.name] = self.start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disabled or self.start_time is None:
            return

        end_time = time.perf_counter()
        duration = end_time - self.start_time
        Timer._timings[self.name].append(duration)

        if self.logger and self.name in Timer._active_timers:
            self.logger.debug(f"{self.name}: {duration:.6f}s")

        if self.name in Timer._active_timers:
            del Timer._active_timers[self.name]

    @classmethod
    def get_stats(cls, name=None):
        """Get timing statistics for a specific timer or all timers."""
        if name is not None:
            times = cls._timings.get(name, [])
            if not times:
                return {}
            return {
                "total": sum(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "last": times[-1],
            }
        else:
            return {name: cls.get_stats(name) for name in cls._timings.keys()}

    @classmethod
    def reset(cls):
        """Reset all timings."""
        cls._timings = defaultdict(list)
        cls._active_timers = {}

    @classmethod
    def disable(cls, disabled=True):
        """Disable all timers."""
        cls._disable = disabled

    @classmethod
    def summarize(cls, logger=None):
        """Summarize all timings."""
        result = {}
        for name, times in cls._timings.items():
            if not times:
                continue

            stats = cls.get_stats(name)
            msg = (
                f"{name} - total: {stats['total']:.3f}s, "
                f"mean: {stats['mean']:.3f}s, "
                f"max: {stats['max']:.3f}s"
            )

            result[name] = stats

            if logger:
                logger.info(msg)

        return result


def dict_parser_type(value):
    """Helper function to parse a JSON string into a dict for argparse."""
    try:
        value = value.replace("'", '"')
        loaded_dict = json.loads(value)
        return loaded_dict
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid JSON format for dictionary: {value}")


class DistributedLLMTrainer:
    """Distributed LLM Trainer."""

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="AdamW DDP Baseline")
        parser.add_argument(
            "--project", type=str, default="boom", help="Wandb project."
        )
        parser.add_argument("--run_name", type=str, default="", help="Wandb run name.")
        parser.add_argument(
            "--device", type=str, default="cuda", help="Device to use for training"
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--trace", action="store_true", help="Enable trace logging")
        parser.add_argument(
            "--hparams_file", type=str, default="hparams.json", help="hparams file."
        )
        parser.add_argument(
            "--use_compile",
            action="store_true",
            help="Use torch.compile to optimize model execution",
        )

        # DDP specific args
        # parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
        parser.add_argument(
            "--num_gpus",
            type=int,
            default=torch.cuda.device_count(),
            help="Number of GPUs to use for distributed training",
        )

        # Optimizer args
        parser.add_argument(
            "--micro_batch_size",
            type=int,
            default=-1,
            help="Micro batches for data loader",
        )
        parser.add_argument(
            "--batch_size", type=int, default=64, help="Batch size for grad accum"
        )
        parser.add_argument(
            "--sequence_length",
            type=int,
            default=2048,
            help="sequence length for training",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.1,
            help="Weight decay for regularization",
        )
        parser.add_argument(
            "--warmup_steps",
            type=float,
            default=0.07,
            help="Number of warmup steps for learning rate scheduler",
        )

        # Strategy args
        parser.add_argument(
            "--strategy",
            type=str,
            default="diloco",
            choices=[
                "adam_baseline",
                "diloco_baseline",
                "demo_baseline",
                "demo_diloco",
                "ccloco",
                "ccloco_muon",
                "custom",
            ],
            help="Training strategy to use",
        )
        parser.add_argument(
            "--inner_optimizer",
            type=str,
            default=None,
            choices=["adamw", "muon"],
            help="inner optimizer to use. None means simple gradient accumulation",
        )
        parser.add_argument(
            "--outer_optimizer",
            type=str,
            default="ccloco",
            choices=["adamw", "demo", "ccloco", "nesterov"],
            help="Outer optimizer to use for training (adamw or ccloco or nesterov)",
        )

        ## Inner optimizer
        parser.add_argument(
            "--inner_steps",
            type=int,
            default=10,
            help="Local steps before communication (H)",
        )
        parser.add_argument(
            "--inner_learning_rate",
            type=float,
            default=6e-4,
            help="Learning rate for inner optimizer",
        )
        ## Muon-specific hyperparameters
        parser.add_argument(
            "--muon_inner_learning_rate",
            type=float,
            default=0.02,
            help="Learning rate specifically for Muon optimizer (default: 0.02)",
        )
        parser.add_argument(
            "--muon_momentum",
            type=float,
            default=0.95,
            help="Momentum coefficient for Muon optimizer (default: 0.95)",
        )
        parser.add_argument(
            "--muon_weight_decay",
            type=float,
            default=0.01,
            help="Weight decay for Muon optimizer (default: 0.01)",
        )
        parser.add_argument(
            "--muon_scaling_mode",
            type=str,
            default="muon",
            choices=["muon", "moonlight"],
            help="Scaling mode for Muon optimizer (only 'muon' and 'moonlight' supported)"
        )
        parser.add_argument(
            "--muon_rms_target",
            type=float,
            default=0.25,
            help="Target RMS for moonlight scaling mode"
        )
        parser.add_argument(
            "--track_muon_rms",
            action="store_true",
            help="Track RMS statistics for Muon updates"
        )
        parser.add_argument(
            "--muon_head_lr_scale",
            type=float,
            default=0.5,
            help="Learning rate scaling factor for head parameters when using Muon (default: 0.5)",
        )
        parser.add_argument(
            "--muon_embed_lr_scale",
            type=float,
            default=0.5,
            help="Learning rate scaling factor for embedding parameters when using Muon (default: 0.5)",
        )
        parser.add_argument(
            "--muon_scalar_lr_scale",
            type=float,
            default=0.2,
            help="Learning rate scaling factor for scalar parameters when using Muon (default: 0.2)",
        )

        ## Outer optimizer
        parser.add_argument(
            "--outer_learning_rate",
            type=float,
            default=0.7,
            help="Learning rate for outer optimizer",
        )
        parser.add_argument(
            "--outer_momentum",
            type=float,
            default=0.0,
            help="Momentum for outer optimizer",
        )
        parser.add_argument(
            "--outer_nesterov", action="store_true", help="Nesterov for outer optimizer"
        )
        parser.add_argument(
            "--outer_use_sign", action="store_true", help="Use sign for outer optimizer"
        )

        ## CCLoco specific args
        parser.add_argument(
            "--error_decay",
            type=float,
            default=0.999,
            help="Error decay for CCLoco optimizer",
        )
        parser.add_argument(
            "--top_k", type=int, default=32, help="Top k for CCLoco optimizer"
        )
        parser.add_argument(
            "--chunk_size", type=int, default=64, help="Chunk size for CCLoco optimizer"
        )
        parser.add_argument(
            "--use_dct",
            action="store_true",
            help="Use DCT transform in CCLoco optimizer",
        )
        parser.add_argument(
            "--use_quantization",
            action="store_true",
            help="Use quantization for CCLoco optimizer",
        )
        parser.add_argument(
            "--quantization_bins",
            type=int,
            default=256,
            help="Number of quantization bins",
        )
        parser.add_argument(
            "--quantization_range",
            type=int,
            default=6,
            help="Quantization range in standard deviations",
        )

        # Dataset args
        parser.add_argument(
            "--token_budget",
            type=int,
            default=15728640,
            help="Token budget for training. If negative, is set from hparams file.",
        )
        parser.add_argument(
            "--shards_path",
            type=str,
            default="~/datasets/edu_fineweb_score2_10B_tokenized_llama2",
            help="Path to the dataset shards.",
        )
        parser.add_argument(    
            "--shard_token_size",
            type=int,
            default=1024**3,
            help="Number of tokens in each shard (must match shard creation)",
        )
        parser.add_argument(
            "--max_steps",
            type=int,
            default=-1,
            help="Maximum number of training steps (None for unlimited)",
        )
        parser.add_argument(
            "--seed", type=int, default=42, help="Seed for deterministic page selection"
        )
        parser.add_argument(
            "--data_in_gpu", action="store_true", help="Keep whole dataset in GPU."
        )

        # Checkpoint args
        parser.add_argument(
            "--save_path",
            type=str,
            default="./checkpoints",
            help="Path to save model checkpoints",
        )
        parser.add_argument(
            "--save_interval",
            type=int,
            default=500,
            help="Save checkpoint every N windows",
        )
        parser.add_argument(
            "--load_checkpoint",
            type=str,
            default=None,
            help="Path to checkpoint file to resume training from",
        )

        # Timing args
        parser.add_argument(
            "--timing_log",
            type=str,
            default="timings.log",
            help="File to write timing information to",
        )

        config = parser.parse_args()

        # Setup the predefined strategy
        tplr.logger.info(f"Strategy: {config.strategy}:")
        if config.strategy == "adam_baseline":
            tplr.logger.info(
                f"[Strat] Hardcoding inner optimizer to None (simple grad accumulation) and outer optimizers to 'adamw'."
            )
            config.inner_optimizer = None
            config.outer_optimizer = "adamw"
        elif config.strategy == "diloco_baseline":
            tplr.logger.info(
                f"[Strat] Hardcoding inner optimizer to 'adamw' and outer optimizers to 'SGD+Nesterov'."
            )
            config.inner_optimizer = "adamw"
            config.outer_optimizer = "nesterov"
        elif config.strategy == "demo_baseline":
            tplr.logger.info(
                f"[Strat] Hardcoding inner optimizer to None (simple grad accumulation) and outer optimizers to 'demo' with DCT and sign."
            )
            config.inner_optimizer = None
            config.outer_optimizer = "demo"
            config.use_dct = True
            config.outer_use_sign = True
        elif config.strategy == "demo_diloco":
            tplr.logger.info(
                f"[Strat] Hardcoding inner optimizer to 'adamw' and outer optimizers to 'demo' with DCT and sign."
            )
            config.inner_optimizer = "adamw"
            config.outer_optimizer = "demo"
            config.use_dct = True
            config.outer_use_sign = (
                False  # With use_sign=True, demo and diloco do not work together well.
            )
        elif config.strategy == "ccloco":
            tplr.logger.info(
                f"[Strat] Hardcoding inner optimizer to 'adamw' and outer optimizers to 'ccloco'."
            )
            config.inner_optimizer = "adamw"
            config.outer_optimizer = "ccloco"
            config.use_dct = False
            config.outer_use_sign = False
        elif config.strategy == "ccloco_muon":
            tplr.logger.info(
                f"[Strat] Hardcoding inner optimizer to 'muon' and outer optimizers to 'ccloco'."
            )
            config.inner_optimizer = "muon"
            config.outer_optimizer = "ccloco"
            config.use_dct = False
            config.outer_use_sign = False
        else:
            if config.strategy != "custom":
                tplr.logger.warning(
                    f"Unknown strategy '{config.strategy}', defaulting to 'custom'"
                )
            tplr.logger.info(
                f"[Strat] Using custom strategy with inner optimizer '{config.inner_optimizer}' and outer optimizer '{config.outer_optimizer}' with DCT={config.use_dct} and sign={config.outer_use_sign}."
            )
            config.strategy = "custom"

        if config.micro_batch_size < 0:
            config.micro_batch_size = config.batch_size

        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()

        return config

    def __init__(self):
        tplr.logger.debug("Starting AdamW baseline initialization...")

        self.config = DistributedLLMTrainer.config()

        self.is_diloco = self.config.inner_optimizer is not None

        self._set_seed_and_backend(self.config.seed)

        self._setup_distributed()

        self._calculate_steps()

        self._initialize_model_and_tokenizer()

        self._setup_optimizers_and_schedulers()

        self._initialize_state_and_metrics()

        self._initialize_dataloader()

        self._setup_wandb_and_logging()

        self._initialize_strategy()

        # summary info
        if self.global_rank == 0:
            # Calculate expected training time and tokens
            tokens_per_step = (
                self.config.batch_size
                * self.world_size
                * self.config.sequence_length
                * self.config.inner_steps
            )
            total_tokens = tokens_per_step * self.config.max_steps

            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB

            tplr.logger.info("\n" + "=" * 80)
            tplr.logger.info(f"TRAINING CONFIGURATION SUMMARY:")
            tplr.logger.info(f"→ Hardware: {self.world_size} GPU(s)")
            tplr.logger.info(
                f"→ Model memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved (excluding batches)"
            )
            tplr.logger.info(
                f"→ Training strategy: {self.config.strategy.upper()} with {self.config.inner_steps} inner steps"
            )

            if self.is_diloco:
                tplr.logger.info(
                    f"→ Inner optimizer: {self.config.inner_optimizer} (lr={self.config.inner_learning_rate}, weight_decay={self.config.weight_decay}, inner_steps={self.config.inner_steps})"
                )

            tplr.logger.info(
                f"→ Outer optimizer: {self.config.outer_optimizer} (lr={self.config.outer_learning_rate}, weight_decay={self.outer_weight_decay})"
            )
            tplr.logger.info(
                f"→ Batch hierarchy: {self.config.micro_batch_size} (micro) → {self.config.batch_size} (accum)"
            )
            tplr.logger.info(
                f"→ Sequence length: {self.config.sequence_length} tokens per sample"
            )

            # Add token computation information
            inner_effective_tokens = (
                self.config.batch_size
                * self.world_size
                * self.config.inner_steps
                * self.config.sequence_length
            )
            tplr.logger.info(
                f"→ Inner cycle: {inner_effective_tokens:,} tokens processed per full inner cycle across all GPUs"
            )
            tplr.logger.info(
                f"→ Training plan: {self.config.max_steps:,} steps, targeting {total_tokens:,} tokens total (given target: {self.config.token_budget:,})"
            )
            tplr.logger.info(
                f"→ Scheduler plan: {self.warmup_steps:,} warmup steps, {self.cosine_steps:,} cosine steps, {self.total_scheduler_steps:,} total scheduler steps"
            )
            tplr.logger.info(
                f"→ Data: {len(self.train_loader.dataset)} samples with {self.config.sequence_length:,} tokens each (seq_len)"
            )

            if self.config.use_compile:
                tplr.logger.info(
                    f"→ Optimization: Using torch.compile for model execution"
                )

            tplr.logger.info("=" * 80 + "\n")

    def _set_seed_and_backend(self, seed):
        """Sets the seed and torch.backend."""
        tplr.logger.info(f"Setting global seed to {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _setup_distributed(self):
        """Set up the distributed training environment."""
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        if self.world_size > 1:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
        else:
            self.local_rank = 0
            self.global_rank = 0

        Timer.disable(not (self.config.debug and self.global_rank == 0))

        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            init_process_group(
                backend="nccl", rank=self.global_rank, world_size=self.world_size
            )
            tplr.logger.info(
                f"Initialized DDP: rank {self.global_rank}/{self.world_size - 1} on device {self.local_rank}"
            )

        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

    def _calculate_steps(self):
        """Calculate training steps."""
        if not self.is_diloco:
            self.config.inner_steps = 1

        # Calculate max_steps
        if self.config.max_steps == -1:
            self.config.max_steps = self.config.token_budget // (
                self.config.batch_size
                * self.config.sequence_length
                * self.config.inner_steps
                * self.world_size
            )

        # Calculate total steps for LR schedulers
        self.total_scheduler_steps = self.config.token_budget // (
            self.config.batch_size * self.config.sequence_length * self.world_size
        )

    def _initialize_model_and_tokenizer(self):
        """Initialize the model and tokenizer."""
        hparams_file = os.path.expandvars(os.path.expanduser(self.config.hparams_file))
        with open(hparams_file, "r") as fp:
            hparams = json.load(fp)

        tokenizer_name = hparams.pop("tokenizer_name")
        model_config = LlamaConfig(**hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.model = LlamaForCausalLM(model_config)

        self.hparams = SimpleNamespace(
            model_config=model_config, tokenizer=self.tokenizer
        )

        if self.config.debug and self.global_rank == 0:
            total = sum(p.numel() for p in self.model.parameters())
            train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            tplr.logger.info(f"using model config: {model_config}")
            tplr.logger.info(f"→ Total params:     {total:,}")
            tplr.logger.info(f"→ Trainable params: {train:,}")

        self.model.to(self.device)

        if self.world_size > 1:
            for p in self.model.parameters():
                dist.broadcast(p.data, src=0)
            if self.global_rank == 0:
                tplr.logger.info("Synchronized model parameters across all processes")

        if self.config.use_compile:
            self.model = torch.compile(self.model, dynamic=True)

        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

    def _initialize_dataloader(self):
        """Initialize the data loader."""
        if self.global_rank == 0:
            # Log memory before dataset creation
            ram_before = psutil.virtual_memory()
            tplr.logger.info(
                f"RAM before dataset creation: {ram_before.used / 1024**3:.2f}GB used, "
                f"{ram_before.available / 1024**3:.2f}GB available"
            )

        train_dataset = tplr.ShardedGPUDataset(
            shards_path=os.path.expandvars(os.path.expanduser(self.config.shards_path)),
            token_budget=self.config.token_budget,
            sequence_length=self.config.sequence_length,
            rank=self.global_rank,
            world_size=self.world_size,
            device=self.device,
            split="train",
            pin_to_gpu=self.config.data_in_gpu,
            shard_token_size=self.config.shard_token_size,
        )

        self.train_loader = tplr.get_dataloader(
            train_dataset, batch_size=self.config.micro_batch_size, shuffle=True
        )
        if self.global_rank == 0:
            # Log memory after dataset creation
            ram_after = psutil.virtual_memory()
            tplr.logger.info(
                f"RAM after dataset creation: {ram_after.used / 1024**3:.2f}GB used, "
                f"{ram_after.available / 1024**3:.2f}GB available"
            )

    def _create_scheduler(self, optimizer, lr):
        """Create a standard scheduler with warmup and cosine annealing."""
        warmup_steps = self.config.warmup_steps
        # If warmup_steps is given as a fraction of total steps:
        if warmup_steps < 1:
            warmup_steps = self.total_scheduler_steps * warmup_steps

        warmup_steps = int(warmup_steps)
        cosine_steps = max(1, self.total_scheduler_steps - warmup_steps)
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps

        if warmup_steps >= self.total_scheduler_steps:
            raise ValueError(
                f"Warmup steps ({self.config.warmup_steps:,}) must be less than total scheduler steps "
                f"({self.total_scheduler_steps:,})."
            )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=lr * 0.1,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    def _setup_optimizers_and_schedulers(self):
        """Set up optimizers and schedulers for training."""
        # Initialize inner optimizer (for Diloco)
        self.inner_optimizer = None
        if self.is_diloco:
            if self.config.inner_optimizer.lower() == "adamw":
                self.inner_optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.config.inner_learning_rate,
                    weight_decay=self.config.weight_decay,
                    betas=(0.9, 0.95),
                )
                if self.global_rank == 0:
                    tplr.logger.info(
                        f"Using AdamW as inner optimizer with lr={self.config.inner_learning_rate} and weight_decay={self.config.weight_decay}"
                    )
            elif self.config.inner_optimizer.lower() == "muon":
                # Separate parameters for Muon (2D matrices) and Adam (embeddings, scalars)
                hidden_2d_params = []
                embed_params = []
                scalar_params = []
                head_params = []
                param_to_name = {}
                
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue

                    # we do this because the original muon impl doesn't use named_parameters
                    # and we're doing layer specific RMS tracking
                    param_to_name[param] = name

                    if param.ndim < 2:
                        scalar_params.append(param)
                    elif "embed" in name:
                        embed_params.append(param)
                    elif "lm_head" in name or "output" in name:
                        head_params.append(param)
                    else:
                        hidden_2d_params.append(param)
                
                if not embed_params:
                    tplr.logger.exception("The embedding layer must contain the word 'embed' so we can isolate it from Muon")
                
                # Create parameter groups for SingleDeviceMuonWithAuxAdam
                adam_groups = []
                if head_params:
                    adam_groups.append(dict(params=head_params, lr=self.config.inner_learning_rate * self.config.muon_head_lr_scale, weight_decay=self.config.weight_decay))
                if embed_params:
                    adam_groups.append(dict(params=embed_params, lr=self.config.inner_learning_rate * self.config.muon_embed_lr_scale, weight_decay=self.config.weight_decay))
                if scalar_params:
                    #TODO validate this, should we apply weight decay to scalars
                    adam_groups.append(dict(params=scalar_params, lr=self.config.inner_learning_rate * self.config.muon_scalar_lr_scale, weight_decay=0)) 
                
                adam_groups = [dict(**g, betas=(0.9, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
                
                # Ensure we have at least some parameters for Muon
                if not hidden_2d_params:
                    tplr.logger.exception("No hidden 2-dim parameters found for Muon optimizer.")
                
                muon_group = dict(
                    params=hidden_2d_params, 
                    lr=self.config.muon_inner_learning_rate,
                    momentum=self.config.muon_momentum,
                    weight_decay=self.config.muon_weight_decay,
                    use_muon=True
                )
                
                param_groups = adam_groups + [muon_group]
                
                # Create MuonScalingConfig for the optimizer
                from muon_adaptive import MuonScalingConfig
                scaling_config = MuonScalingConfig(
                    mode=self.config.muon_scaling_mode,
                    rms_target=self.config.muon_rms_target,
                    track_rms=self.config.track_muon_rms
                )
                
                self.inner_optimizer = SingleDeviceMuonWithAuxAdamAdaptive(param_groups, scaling_config=scaling_config)

                self.inner_optimizer.set_param_names(param_to_name)
                
                if self.global_rank == 0:
                    tplr.logger.info(
                        f"Using Muon as inner optimizer with muon_scaling_mode={self.config.muon_scaling_mode}, "\
                        f"lr={self.config.muon_inner_learning_rate}, momentum={self.config.muon_momentum}, "\
                        f"weight_decay={self.config.muon_weight_decay}"
                    )
                    if self.config.muon_scaling_mode == 'moonlight':
                        tplr.logger.info(f"  - muon_rms_target: {self.config.muon_rms_target}")
                    
                    tplr.logger.info(f"  - Hidden matrix params: {len(hidden_2d_params)} (Muon)")
                    tplr.logger.info(f"  - Embedding params: {len(embed_params)} (Adam, lr={self.config.inner_learning_rate * self.config.muon_embed_lr_scale})")
                    tplr.logger.info(f"  - Scalar params: {len(scalar_params)} (Adam, lr={self.config.inner_learning_rate * self.config.muon_scalar_lr_scale})")
                    tplr.logger.info(f"  - Head params: {len(head_params)} (Adam, lr={self.config.inner_learning_rate * self.config.muon_head_lr_scale})")
                    tplr.logger.info(f"  - Track RMS: {self.config.track_muon_rms}")
            else:
                raise NotImplementedError(
                    f"Unknown inner optimizer: {self.config.inner_optimizer}"
                )

        # Initialize outer optimizer
        self.outer_weight_decay = (
            0.0 if self.is_diloco else self.config.weight_decay
        )  # In diloco, we apply WD only to inner optimizer.
        if self.config.outer_optimizer.lower() in ["ccloco", "demo"]:
            self.outer_optimizer = tplr.CCLoco(
                self.model.parameters(),
                lr=self.config.outer_learning_rate,
                momentum=self.config.outer_momentum,
                weight_decay=self.outer_weight_decay,
                error_decay=self.config.error_decay,
                top_k=self.config.top_k,
                chunk_size=self.config.chunk_size,
                use_dct=self.config.use_dct,
                use_sign=self.config.outer_use_sign,
                use_quantization=self.config.use_quantization,
                quantization_bins=self.config.quantization_bins,
                quantization_range=self.config.quantization_range,
                process_group=dist.group.WORLD if self.world_size > 1 else None,
            )
        elif self.config.outer_optimizer.lower() == "adamw":
            self.outer_optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.outer_learning_rate,
                weight_decay=self.outer_weight_decay,
                betas=(0.9, 0.95),
                eps=0.1 if self.is_diloco else 1e-8
            )
        elif self.config.outer_optimizer.lower() == "nesterov":
            self.outer_optimizer = SGD(
                self.model.parameters(),
                lr=self.config.outer_learning_rate,
                weight_decay=self.outer_weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise NotImplementedError(
                f"Unknown outer optimizer: {self.config.outer_optimizer}"
            )

        if self.global_rank == 0:
            tplr.logger.info(
                f"Using {self.config.outer_optimizer} outer optimizer with DDP with LR={self.config.outer_learning_rate} and weight_decay={self.outer_weight_decay}"
            )

        # Create scheduler
        optimizer_for_scheduler = (
            self.inner_optimizer if self.is_diloco else self.outer_optimizer
        )
        lr_for_scheduler = (
            self.config.inner_learning_rate
            if self.is_diloco
            else self.config.outer_learning_rate
        )
        scheduler = self._create_scheduler(optimizer_for_scheduler, lr_for_scheduler)
        self.scheduler = scheduler
        if self.is_diloco:
            self.inner_scheduler = scheduler
            self.outer_scheduler = None  # No outer scheduler for Diloco
        else:
            self.inner_scheduler = None  # No inner scheduler for SimpleAccum
            self.outer_scheduler = scheduler

    def _initialize_state_and_metrics(self):
        """Initialize state variables and metrics tracking."""
        if self.global_rank == 0:
            os.makedirs(self.config.save_path, exist_ok=True)

        self.step_counter = 0
        self.global_step = 0

        self.total_tokens_processed = 0
        self.batch_times = []

        if self.config.load_checkpoint is not None:
            self._load_checkpoint(self.config.load_checkpoint)

    def _setup_wandb_and_logging(self):
        """Set up WandB and timing loggers."""
        if self.global_rank == 0:
            self.wandb = wandb.init(
                project=self.config.project,
                name=f"{self.config.run_name}" if self.config.run_name else "runner",
                config=vars(self.config),
                group="loco",
                job_type="loco_training",
            )
        else:
            self.wandb = None

        self.timing_logger = None
        if self.config.debug:
            self.setup_timing_logger()

    def _initialize_strategy(self):
        """Initialize the training strategy."""
        if self.is_diloco:
            self.strategy = tplr.Diloco(
                self.device,
                self.world_size,
                self.global_rank,
                self.tokenizer,
                self.config,
            )
        else:
            self.strategy = tplr.SimpleAccum(
                self.device,
                self.world_size,
                self.global_rank,
                self.tokenizer,
                self.config,
            )

    def setup_timing_logger(self):
        """Set up a separate logger for performance timing information."""
        log_dir = os.path.dirname(self.config.timing_log)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.timing_logger = logging.getLogger("timing")
        self.timing_logger.setLevel(logging.DEBUG)
        self.timing_logger.propagate = False  # Don't propagate to root logger

        if self.timing_logger.handlers:
            self.timing_logger.handlers.clear()
        file_handler = logging.FileHandler(self.config.timing_log, mode="w")

        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)

        self.timing_logger.addHandler(file_handler)

        self.timing_logger.info(
            f"Starting new training run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.timing_logger.info(
            f"Configuration: optimizer={self.config.outer_optimizer}, lr={self.config.outer_learning_rate}, "
            f"world_size={self.world_size}, batch_size={self.config.batch_size}"
        )
        self.timing_logger.info("-" * 80)

    def log_timing(self, message):
        """Helper to log timing information to the timing log file."""
        if self.global_rank == 0 and self.timing_logger is not None:
            self.timing_logger.info(message)

    def run(self):
        """Main training loop."""
        for window in range(self.global_step, self.config.max_steps):
            if self.global_step >= self.config.max_steps:
                tplr.logger.info(
                    f"Reached maximum steps {self.config.max_steps}. Stopping."
                )
                break

            if self.global_rank == 0:
                tplr.logger.info(
                    f"\n{'-' * 40} Window: {window}/{self.config.max_steps} {'-' * 40}"
                )
                if self.config.debug:
                    self.log_timing(f"Window {window} - Starting gradient accumulation")

            # Reset timers for this window
            if self.global_rank == 0:
                Timer.reset()

            with Timer("window_total", enabled=True):
                # Training loop
                if self.global_rank == 0:
                    tplr.logger.info("Start accumulating gradients...")

                self.model.zero_grad()

                # Use strategy for inner step (gradient accumulation)
                with Timer("inner_step"):
                    metrics = self.strategy.inner_step(
                        self.model,
                        self.train_loader,
                        self.inner_optimizer,
                        self.inner_scheduler,
                    )

                # Reduce metrics across workers
                with Timer("reduce_metrics"):
                    metrics_to_reduce = torch.tensor(
                        [
                            metrics["total_loss"],
                            metrics["batch_count"],
                            metrics["batch_tokens"],
                            metrics["loss_after_gather"],
                        ],
                        device=self.device,
                    )

                    if self.world_size > 1:
                        torch.distributed.all_reduce(
                            metrics_to_reduce, op=torch.distributed.ReduceOp.SUM
                        )

                    loss_after_inner = metrics_to_reduce[0].item() / self.world_size
                    batch_count = metrics_to_reduce[1].item()
                    batch_tokens = metrics_to_reduce[2].item()
                    loss_after_gather = metrics_to_reduce[3].item() / self.world_size

                # Use strategy for outer step
                with Timer("outer_step"):
                    self.strategy.outer_step(
                        self.model, self.outer_optimizer, self.scheduler
                    )

            if self.global_rank == 0:
                # Muon RMS Tracking
                if (self.config.inner_optimizer == "muon" and 
                    self.config.track_muon_rms and 
                    hasattr(self.inner_optimizer, 'get_rms_stats')
                ):
                    rms_stats = self.inner_optimizer.get_rms_stats()
                    muon_rms = {}
                    for layer_type, stats in rms_stats.items():
                        for stat_name, value in stats.items():
                            muon_rms[f"muon_rms/{layer_type}/{stat_name}"] = value

                # Calculate tokens per second
                all_stats = Timer.summarize(
                    logger=self.timing_logger if self.config.debug else None
                )
                window_duration = all_stats.get("window_total", {}).get("total", 0)

                tokens_per_second = batch_tokens / window_duration
                tplr.logger.info(
                    f"Window {window}: Processing rate: {tokens_per_second:.2f} tokens/sec"
                )

                timer_metrics = {}
                timer_metrics[f"timing/tokens_per_sec"] = tokens_per_second
                if self.config.debug:
                    self.log_timing(f"Window {window} - Timing summary:")
                    self.log_timing(
                        f"  Total tokens: {batch_tokens}, Tokens/sec: {tokens_per_second:.2f}"
                    )

                    for timer_name, stats in all_stats.items():
                        timer_metrics[f"timing/{timer_name}/total"] = stats.get(
                            "total", 0
                        )
                        timer_metrics[f"timing/{timer_name}/mean"] = stats.get(
                            "mean", 0
                        )
                        timer_metrics[f"timing/{timer_name}/max"] = stats.get("max", 0)

                    self.log_timing("-" * 40)

                tplr.logger.info(
                    f"effective_batch_size: {self.config.batch_size * self.world_size}"
                )
                tplr.logger.info(
                    f"Window {window} completed: {batch_count} batches with {batch_tokens} tokens"
                )

                # Log gradient metrics
                params_without_grad = sum(1 for p in self.model.parameters() if p.grad is None)
                if params_without_grad > 0:
                    tplr.logger.warning(f"Found {params_without_grad} parameters without gradients")
                
                grad_norms = [
                    p.grad.norm().item()
                    for p in self.model.parameters()
                    if p.grad is not None
                ]
                weight_norms = [p.norm().item() for p in self.model.parameters()]

                tplr.logger.info(
                    f"gradient/mean_grad_norm: {sum(grad_norms) / len(grad_norms) if grad_norms else 0:.3f}, "
                    f"gradient/max_grad_norm: {max(grad_norms) if grad_norms else 0:.3f}, "
                    f"gradient/min_grad_norm: {min(grad_norms) if grad_norms else 0:.3f}, "
                    f"gradient/grad_norm_std: {torch.tensor(grad_norms).std().item() if grad_norms else 0:.3f}, "
                    f"gradient/mean_weight_norm: {sum(weight_norms) / len(weight_norms):.3f}"
                )

                # Wandb logging
                metrics_dict = {
                    # Training metrics
                    "baseline/loss_after_inner": loss_after_inner,
                    "baseline/loss_after_gather": loss_after_gather,
                    "baseline/total_tokens": self.total_tokens_processed + batch_tokens,
                    "baseline/batch_tokens": batch_tokens,
                    "baseline/global_step": self.global_step,
                    "baseline/perplexity_after_inner": torch.exp(
                        torch.tensor(loss_after_inner)
                    ).item(),
                    "baseline/perplexity_after_gather": torch.exp(
                        torch.tensor(loss_after_gather)
                    ).item(),
                    "baseline/tokens_per_sec": tokens_per_second,
                    # Resource metrics
                    "misc/gpu_memory_allocated": torch.cuda.memory_allocated()
                    / 1024**2,  # MB
                    "misc/gpu_memory_cached": torch.cuda.memory_reserved()
                    / 1024**2,  # MB
                    # Network metrics
                    "setting/num_gpus": self.world_size,
                    "setting/effective_batch_size": self.world_size
                    * self.config.batch_size
                    * self.config.inner_steps,
                    "setting/learning_rate": self.scheduler.get_last_lr()[0],
                    # Gradient statistics as points
                    "gradient/mean_grad_norm": sum(grad_norms) / len(grad_norms)
                    if grad_norms
                    else 0,
                    "gradient/max_grad_norm": max(grad_norms) if grad_norms else 0,
                    "gradient/min_grad_norm": min(grad_norms) if grad_norms else 0,
                    "gradient/grad_norm_std": torch.tensor(grad_norms).std().item()
                    if grad_norms
                    else 0,
                    "gradient/mean_weight_norm": sum(weight_norms) / len(weight_norms),
                    "gradient/grad_to_weight_ratio": (sum(grad_norms) / len(grad_norms))
                    / (sum(weight_norms) / len(weight_norms))
                    if grad_norms and weight_norms
                    else 0,
                }

                # Add optimizer-specific learning rates
                if self.is_diloco:
                    metrics_dict["setting/inner_learning_rate"] = (
                        self.inner_scheduler.get_last_lr()[0]
                    )
                    metrics_dict["setting/outer_learning_rate"] = (
                        self.config.outer_learning_rate
                    )

                metrics_dict.update(timer_metrics)
                if (self.config.inner_optimizer == "muon" and 
                    self.config.track_muon_rms and 
                    hasattr(self.inner_optimizer, 'get_rms_stats')
                ):
                    metrics_dict.update(muon_rms)

                self.wandb.log(metrics_dict, step=self.global_step)

                # Update total tokens processed
                self.total_tokens_processed += batch_tokens

                # Save checkpoint every save_interval windows
                if (
                    (window + 1) % self.config.save_interval == 0
                    or window == self.config.max_steps - 1
                ):
                    self._save_checkpoint(window)
            else:
                self.total_tokens_processed += batch_tokens

            self.global_step += 1

        if self.global_rank == 0:
            tplr.logger.info(
                f"Training completed with {self.total_tokens_processed} tokens processed."
            )

    def _save_checkpoint(self, window):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            self.config.save_path, f"demo_checkpoint_window_{window}_{timestamp}.pt"
        )

        if isinstance(self.model, DDP):
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.outer_optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
        }

        # Add inner optimizer/scheduler state for Diloco
        if self.is_diloco:
            checkpoint.update(
                {
                    "inner_optimizer_state_dict": self.inner_optimizer.state_dict(),
                    "inner_scheduler_state_dict": self.inner_scheduler.state_dict()
                    if self.inner_scheduler
                    else None,
                }
            )

        # Add training state
        checkpoint.update(
            {
                "window": window,
                "global_step": self.global_step,
            }
        )

        torch.save(checkpoint, path)
        tplr.logger.info(f"Saved checkpoint to {path}")

    def _load_checkpoint(self, checkpoint_path):
        """Load model, optimizer, and training state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            tplr.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        tplr.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer and scheduler states
        self.outer_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"]
            and self.scheduler
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load inner optimizer and scheduler for Diloco
        if self.is_diloco:
            if "inner_optimizer_state_dict" in checkpoint and self.inner_optimizer:
                self.inner_optimizer.load_state_dict(
                    checkpoint["inner_optimizer_state_dict"]
                )
            if "inner_scheduler_state_dict" in checkpoint and self.inner_scheduler:
                self.inner_scheduler.load_state_dict(
                    checkpoint["inner_scheduler_state_dict"]
                )

        # Load training state
        self.global_step = checkpoint.get("global_step", 0)

        tplr.logger.info(f"Resumed training at global step {self.global_step}")

    def cleanup(self):
        """Clean up resources."""
        if self.world_size > 1:
            destroy_process_group()

        if self.wandb is not None:
            self.wandb.finish()

        # Close timing logger
        if self.global_rank == 0 and self.timing_logger is not None:
            for handler in self.timing_logger.handlers:
                handler.close()
                self.timing_logger.removeHandler(handler)


def main():
    """Entry point."""
    trainer = DistributedLLMTrainer()

    try:
        trainer.run()
    except KeyboardInterrupt:
        tplr.logger.info("Training interrupted by user")
    except Exception as e:
        tplr.logger.error(f"Error during training: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
