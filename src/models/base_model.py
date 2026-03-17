"""Base model class for all Bug Hunter models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ModelOutput:
    """Standard output for all models."""

    logits: Tensor           # (batch_size, num_classes)
    loss: Tensor | None      # scalar
    embeddings: Tensor | None = None  # (batch_size, hidden_dim)
    attention_weights: Tensor | None = None  # model-specific


class BaseVulnDetector(nn.Module, ABC):
    """Abstract base class for vulnerability detection models.

    All models in Bug Hunter should inherit from this class.
    Provides common functionality: mixed precision, memory management,
    gradient checkpointing.
    """

    def __init__(self, num_classes: int = 2, use_amp: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_amp = use_amp
        self._gradient_checkpointing = False

    @abstractmethod
    def forward(self, **kwargs) -> ModelOutput:
        """Forward pass - must be implemented by subclasses."""
        ...

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save VRAM."""
        self._gradient_checkpointing = True
        # VRAM: saves ~40% memory at cost of ~25% speed

    def count_parameters(self) -> dict[str, int]:
        """Count trainable and total parameters."""
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

    def estimate_memory_mb(self, batch_size: int = 8) -> float:
        """Estimate GPU memory usage in MB (rough estimate)."""
        param_mem = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        # Rule of thumb: activations ≈ 2-3x parameters for training
        estimated = param_mem * 3 * batch_size / (1024 ** 2)
        return estimated

    @torch.no_grad()
    def predict(self, **kwargs) -> Tensor:
        """Run inference without gradients."""
        self.eval()
        output = self.forward(**kwargs)
        probs = torch.softmax(output.logits, dim=-1)
        return probs
