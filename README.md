# CCLoco: Scaling Up Top-K Error Feedback with Local Optimizers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This repository provides a PyTorch implementation of **CCLoco**,  Chunk Compressed Low-Communication distributed training. CCLoco mitigates the communication bottleneck by combining chunked gradient compression, Top-K sparsification, and local error feedback, enabling efficient training of large language models.

## Key Features

- **CCLoco Optimizer**: A reference implementation of the core algorithm in `src/tplr/ccloco.py`.
- **Multiple Training Strategies**: Includes baselines for robust comparison:
    - `CCLoco`: Proposed gradient chunk-compression with local optimization.
    - `DiLoCo`: Distributed training with a local optimization.
    - `DeMo`: Gradient compression with DCT, with and without local optimization steps.
    - `AdamW`: Standard distributed data-parallel training.
- **Efficient Data Handling**: A sharded data pipeline (`src/tplr/data.py`) for handling massive datasets with low memory overhead.

## Getting Started

### 1. Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for environment management)
- This codebase has been tested with H100 and H200 GPUs

### 2. Installation

Clone the repository and install the required dependencies using `uv`.

```bash
git clone https://github.com/tplr-ai/CCLoco
cd CCLoco
uv sync
source .venv/bin/activate
```

### 3. Data Preparation

The training script expects a pre-tokenized and sharded dataset. Use the `pretokenize_data.py` script to process a dataset from Hugging Face.

The default configuration uses `mlfoundations/dclm-baseline-1.0-parquet` and expects the output in `~/datasets/dclm_10B_tokenized`.

```bash
export DATA_DIR="~/datasets/"
python pretokenize_data.py --output_dir $DATA_DIR/dclm_10B_tokenized --total_tokens 10e9
```
*Note: Ensure the `--output_dir` matches the `shards_path` in the sweep configuration files (`hparams/1B/sweeps/*.yaml`) or update the YAML files accordingly.*

## Running Experiments

Experiments are managed through `wandb` sweeps. The `run_sweep.sh` script simplifies the process by creating a sweep and launching a `wandb` agent.

First, set your W&B API key:
```bash
export WANDB_API_KEY="..."
```

Then, run any of the predefined experiments using the corresponding sweep file. Each experiment is configured to run on **8 GPUs** by default (`--nproc_per_node=8`). You can adjust the number of GPUs by modifying the `--nproc_per_node` parameter in the sweep configuration files.

### CCLoco (Proposed Method)
```bash
bash ./run_sweep.sh hparams/1B/sweeps/ccloco.yaml
```

### Baselines

**DiLoCo Baseline**: Baseline DiLoCo with Nesterov outer optimizer
```bash
bash ./run_sweep.sh hparams/1B/sweeps/diloco_baseline.yaml
```

**Demo-DiLoCo**: Replaces DiLoCo's outer step with DeMo (without signum)
```bash
bash ./run_sweep.sh hparams/1B/sweeps/demo_diloco.yaml
```

**DeMo Baseline**: Standard DDP with DeMo
```bash
bash ./run_sweep.sh hparams/1B/sweeps/demo_baseline.yaml
```

**AdamW Baseline**: Standard DDP with AdamW
```bash
bash ./run_sweep.sh hparams/1B/sweeps/adam_baseline.yaml
```

## Citation
If you find **CCLoco** useful in your work, please consider citing our work. You can read more about CCLoco in our [blog post](https://templarresearch.substack.com/p/ccloco-scaling-up-top-k-error-feedback).
```bibtex
@misc{sarfi2025ccloco,
      title={CCLoco: Scaling Up Top-K Error Feedback with Local Optimizers}, 
      author={Sarfi, A. et al.},
      howpublished = {\url{https://github.com/tplr-ai/CCLoco}},
      year={2025},
      note={Full report coming soon}
}
```