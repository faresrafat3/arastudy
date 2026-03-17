"""GPU memory management utilities for RTX 4050 (6GB VRAM)."""

import gc
import logging
from contextlib import contextmanager
from functools import wraps

import torch

logger = logging.getLogger(__name__)

# === Constants for RTX 4050 ===
MAX_VRAM_GB = 6.0
SAFE_VRAM_GB = 5.0  # Leave 1GB headroom
WARNING_THRESHOLD = 0.85  # 85% usage


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"available": False}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    return {
        "available": True,
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(total - allocated, 2),
        "utilization": round(allocated / total, 3),
    }


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage."""
    info = get_gpu_memory_info()
    if info["available"]:
        msg = (
            f"[GPU{' - ' + prefix if prefix else ''}] "
            f"Allocated: {info['allocated_gb']:.2f}GB / "
            f"{info['total_gb']:.1f}GB "
            f"({info['utilization']*100:.1f}%)"
        )
        if info["utilization"] > WARNING_THRESHOLD:
            logger.warning(f"⚠️ HIGH VRAM: {msg}")
        else:
            logger.info(f"📊 {msg}")


def clear_gpu_memory() -> None:
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


@contextmanager
def gpu_memory_monitor(label: str = "operation"):
    """Context manager to monitor GPU memory usage."""
    log_gpu_memory(f"Before {label}")
    try:
        yield
    finally:
        log_gpu_memory(f"After {label}")


def auto_batch_size(
    model: torch.nn.Module,
    sample_input_fn,
    max_batch: int = 64,
    target_memory_gb: float = SAFE_VRAM_GB,
) -> int:
    """Automatically find the maximum batch size that fits in VRAM.

    Args:
        model: The model to test
        sample_input_fn: Function that takes batch_size and returns input
        max_batch: Maximum batch size to try
        target_memory_gb: Target max memory usage

    Returns:
        Optimal batch size
    """
    model.eval()
    optimal_batch = 1

    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        if batch_size > max_batch:
            break

        clear_gpu_memory()
        try:
            with torch.no_grad(), torch.cuda.amp.autocast():
                inputs = sample_input_fn(batch_size)
                _ = model(**inputs)

            info = get_gpu_memory_info()
            if info["allocated_gb"] < target_memory_gb:
                optimal_batch = batch_size
                logger.info(
                    f"✅ batch_size={batch_size}: "
                    f"{info['allocated_gb']:.2f}GB"
                )
            else:
                logger.info(
                    f"❌ batch_size={batch_size}: "
                    f"{info['allocated_gb']:.2f}GB (exceeds target)"
                )
                break
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"OOM at batch_size={batch_size}")
                clear_gpu_memory()
                break
            raise

    model.train()
    clear_gpu_memory()
    logger.info(f"🎯 Optimal batch size: {optimal_batch}")
    return optimal_batch
