"""
Grok4 - Advanced AI Framework

A next-generation AI framework for advanced language understanding, reasoning, and generation.
"""

__version__ = "0.1.0"
__author__ = "Grok4 Team"
__email__ = "team@grok4.ai"

from .models.grok4_model import Grok4Model
from .training.trainer import Trainer
from .inference.engine import InferenceEngine
from .utils.config import Config
from .utils.logger import get_logger

__all__ = [
    "Grok4Model",
    "Trainer", 
    "InferenceEngine",
    "Config",
    "get_logger",
    "__version__",
]

# Configure default logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)