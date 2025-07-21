__version__ = "0.2.2"

# Import package.
from .ccloco import CCLoco
from .data import ShardedGPUDataset, get_dataloader
from .strategies import SimpleAccum, Diloco
from .logging_utils import *

# hint type for logger
from logging import Logger

logger: Logger
