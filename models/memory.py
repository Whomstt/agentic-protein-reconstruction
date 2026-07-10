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
    """Print a one-line host/GPU memory snapshot. Used to trace where RAM grows
    over a long agentic run without attaching an external profiler."""
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
    """Releases cached (but currently unused) GPU memory back to the driver.

    PyTorch's caching allocator doesn't proactively release freed blocks, so a
    long run with many differently-shaped tensors (varying fragment counts,
    varying sequence lengths across samples/iterations) can grow the reserved
    memory pool until it OOMs even though the active memory at any instant is
    modest. Call this between samples/iterations to keep the reserved pool
    from climbing unbounded over a long evaluation or sweep.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
