"""Small utilities for keeping inference memory bounded on constrained machines."""

from __future__ import annotations

import gc
from typing import Any


def get_device() -> str:
    """Return the best supported PyTorch device, preferring CUDA then Apple MPS."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def flush_memory(*objects: Any) -> None:
    """Release model references, collect Python garbage, and clear accelerator caches.

    Call this after every inference stage.  Deleting local references remains the
    caller's responsibility; this function also accepts objects to make the
    disposal intent explicit at call sites.
    """
    # Drop this function's references before collecting cyclic garbage.
    del objects
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        # Memory cleanup should never hide a successful inference result.
        pass
