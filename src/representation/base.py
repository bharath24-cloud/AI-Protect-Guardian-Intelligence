from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, Optional
import numpy as np
import torch

class RepresentationModel(ABC):
    """
    Abstract Base Class for Self-Supervised Scientific Representation Learning Models.
    """

    @abstractmethod
    def fit(self, data: np.ndarray, epochs: int = 20, batch_size: int = 32) -> Dict[str, Any]:
        """Trains the self-supervised representation model on scientific features."""
        pass

    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Transforms input scientific features (N, D) into latent representations (N, d)."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Saves model weights and hyperparameters."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Loads model weights and hyperparameters."""
        pass
