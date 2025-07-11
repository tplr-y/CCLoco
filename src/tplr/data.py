# Standard library
import gc
import glob
import os
from pathlib import Path
from typing import Literal

# Third party
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Local
import tplr


class ShardedGPUDataset(Dataset):
    """
    Memory-efficient dataset that loads tokenized data from .npy shards using memmap.
    Each DDP worker pre-loads its assigned data portion to RAM or GPU memory.
    """

    def __init__(
        self,
        shards_path: str,
        token_budget: int,
        sequence_length: int,
        rank: int,
        world_size: int,
        device: torch.device,
        split: Literal["train", "validation"] = "train",
        pin_to_gpu: bool = False,
        shard_dtype: str = "uint16",
    ):
        """
        Args:
            shards_path: Directory containing .npy shard files
            token_budget: Total tokens to load across all workers
            sequence_length: Length of each training sequence
            rank: Current worker rank
            world_size: Total number of workers
            device: Target device for data
            split: Data split to use ("train" or "validation")
            pin_to_gpu: Whether to keep data on GPU memory
            shard_dtype: NumPy dtype of tokens in shards (must match shard creation)
        """
        super().__init__()

        if split not in ["train", "validation"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be 'train' or 'validation'."
            )

        self.shards_path = Path(shards_path)
        self.sequence_length = sequence_length
        self.pin_to_gpu = pin_to_gpu

        if not self.shards_path.is_dir():
            raise FileNotFoundError(f"Shards directory not found: {self.shards_path}")

        # Discover shard files and set up data type
        shard_prefix = f"{split}_"
        shard_files = sorted(glob.glob(str(self.shards_path / f"{shard_prefix}*.npy")))
        if not shard_files:
            raise FileNotFoundError(
                f"No shard files with prefix '{shard_prefix}' in {self.shards_path}"
            )

        dtype = np.dtype(shard_dtype)
        itemsize = dtype.itemsize

        # Calculate worker's token slice
        tokens_per_worker = token_budget // world_size
        worker_start = rank * tokens_per_worker
        worker_end = (rank + 1) * tokens_per_worker
        if rank == world_size - 1:
            worker_end = token_budget

        # Load worker's data segments using memmap
        tplr.logger.info(f"[Rank {rank}] Calculating data segments to load...")
        chunks, tokens_seen = [], 0

        for shard_file in shard_files:
            if tokens_seen >= token_budget:
                break

            try:
                num_tokens_in_shard = os.path.getsize(shard_file) // itemsize
                shard_start_global = tokens_seen
                shard_end_global = min(tokens_seen + num_tokens_in_shard, token_budget)

                if shard_end_global > worker_start and shard_start_global < worker_end:
                    shard_mmap = np.memmap(shard_file, dtype=dtype, mode="r")
                    local_start = max(0, worker_start - shard_start_global)
                    local_end = min(len(shard_mmap), worker_end - shard_start_global)

                    if local_start < local_end:
                        segment = shard_mmap[local_start:local_end]
                        # Copy to free mmap handle and convert to torch.long for embeddings
                        chunks.append(torch.from_numpy(segment.copy()).long())

                    del shard_mmap
                tokens_seen = shard_end_global

            except Exception as e:
                raise IOError(f"Error processing shard file {shard_file}: {e}")

        if not chunks:
            raise ValueError(
                f"[Rank {rank}] No tokens loaded. Check token budget and worker assignment."
            )

        # Consolidate and optionally move to GPU
        worker_tokens = torch.cat(chunks)
        del chunks
        gc.collect()

        if self.pin_to_gpu:
            self.worker_tokens = worker_tokens.to(device)
            del worker_tokens
        else:
            self.worker_tokens = worker_tokens

        self.num_samples = len(self.worker_tokens) // self.sequence_length
        tplr.logger.info(
            f"[Rank {rank}] Loaded {len(self.worker_tokens):,} tokens, "
            f"creating {self.num_samples:,} samples of length {self.sequence_length}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        end = start + self.sequence_length
        return self.worker_tokens[start:end]


def get_dataloader(
    dataset: ShardedGPUDataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader for ShardedGPUDataset with appropriate memory settings."""
    if not isinstance(dataset, ShardedGPUDataset):
        raise TypeError("dataset must be an instance of ShardedGPUDataset")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=not dataset.pin_to_gpu,
    )
