import gc
import os

import torch

try:
    import psutil

    _PROCESS = psutil.Process(os.getpid())
except Exception:  # psutil optional — logging degrades gracefully without it
    _PROCESS = None


def rss_mb() -> float:
    """Current resident-set (host RAM) size of this process in MB, or -1 if
    psutil isn't available."""
    if _PROCESS is None:
        return -1.0
    return _PROCESS.memory_info().rss / (1024 * 1024)


def log_memory(label: str) -> None:
    """Print a one-line host/GPU memory snapshot."""
    host = rss_mb()
    host_text = f"{host:.0f}MB" if host >= 0 else "n/a"
    if torch.cuda.is_available():
        gpu = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(
            f"    [mem] {label}: host={host_text} | gpu={gpu:.0f}MB alloc, "
            f"{reserved:.0f}MB reserved",
            flush=True,
        )
    else:
        print(f"    [mem] {label}: host={host_text}", flush=True)


def free_gpu_memory() -> None:
    """Release cached but unused GPU memory back to the driver.

    PyTorch's caching allocator does not return freed blocks on its own, so a long
    run over differently-shaped tensors can grow the reserved pool until it OOMs
    while active memory stays modest. Called between samples and iterations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
