"""Browser-first multilingual embedding training toolkit."""

from browser_embedding.config import ExperimentConfig, load_experiment
from browser_embedding.model import BrowserEncoder

__all__ = ["BrowserEncoder", "ExperimentConfig", "load_experiment"]
__version__ = "0.1.0"
