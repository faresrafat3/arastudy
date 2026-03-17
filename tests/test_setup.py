"""Quick test to verify the entire setup is working."""

import pytest
import torch


def test_cuda_available():
    """Verify GPU is accessible."""
    assert torch.cuda.is_available(), "CUDA not available!"


def test_gpu_name():
    """Verify correct GPU."""
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}")
    assert "RTX" in name or "GeForce" in name


def test_vram():
    """Verify VRAM is sufficient."""
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    assert total >= 5.0, f"Need at least 5GB VRAM, got {total:.1f}GB"





def test_mixed_precision():
    """Test AMP works correctly."""
    from torch.cuda.amp import autocast

    x = torch.randn(32, 128).cuda()
    linear = torch.nn.Linear(128, 64).cuda()

    with autocast():
        out = linear(x)

    assert out.dtype == torch.float16 or out.dtype == torch.float32
    print(f"✅ AMP output dtype: {out.dtype}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
