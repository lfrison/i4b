"""Zero-copy framework export utilities using DLPack.

JAX GPU arrays can be consumed by PyTorch, TensorFlow, or CuPy without any
CPU round-trip by sharing the underlying device buffer via the DLPack protocol.

Usage example::

    from src.gym_interface.framework_export import jax_to_torch, jax_to_tf, jax_to_cupy

    obs, *_ = env.step(actions)          # JAX GPU array
    obs_torch = jax_to_torch(obs)        # torch.Tensor on CUDA, zero-copy
    obs_tf    = jax_to_tf(obs)           # tf.Tensor on GPU, zero-copy
    obs_cupy  = jax_to_cupy(obs)         # cupy.ndarray on GPU, zero-copy
"""
from __future__ import annotations


def jax_to_torch(arr, device=None):
    """Convert a JAX array to a PyTorch tensor via DLPack (zero-copy on GPU).

    Args:
        arr: JAX array (on any device).
        device: Optional torch device string or ``torch.device`` to move the
            tensor to after conversion (e.g. ``'cuda:0'``).  If ``None`` the
            tensor stays on the same device as ``arr``.

    Returns:
        ``torch.Tensor`` sharing GPU memory with ``arr`` (no CPU staging).

    Raises:
        ImportError: if ``torch`` or ``jax`` are not installed.
    """
    import torch
    # Use the modern __dlpack__ protocol (JAX ≥0.4, PyTorch ≥1.10).
    # This avoids the legacy jax.dlpack path which requires a CPU backend.
    t = torch.from_dlpack(arr)
    if device is not None:
        t = t.to(device)
    return t


def jax_to_tf(arr):
    """Convert a JAX array to a TensorFlow tensor via DLPack (zero-copy on GPU).

    Args:
        arr: JAX array (on any device).

    Returns:
        ``tf.Tensor`` sharing device memory with ``arr``.

    Raises:
        ImportError: if ``tensorflow`` or ``jax`` are not installed.
    """
    import tensorflow as tf
    return tf.experimental.dlpack.from_dlpack(arr)


def jax_to_cupy(arr):
    """Convert a JAX array to a CuPy ndarray via DLPack (zero-copy on GPU).

    Args:
        arr: JAX array (must be on a CUDA device).

    Returns:
        ``cupy.ndarray`` sharing GPU memory with ``arr``.

    Raises:
        ImportError: if ``cupy`` or ``jax`` are not installed.
    """
    import cupy
    return cupy.from_dlpack(arr)
