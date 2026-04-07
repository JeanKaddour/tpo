"""Runtime bootstrap helpers for JAX CUDA library discovery."""

from __future__ import annotations

import ctypes
import glob
import os
import sys

_CUDA_SETUP_SENTINEL = "_TPO_CUDA_SETUP"


def _nvidia_lib_dirs() -> list[str]:
    """Return pip-installed NVIDIA library directories if available."""
    try:
        import nvidia
    except ImportError:
        return []

    base = os.path.dirname(nvidia.__file__)
    lib_dirs: list[str] = []
    for name in sorted(os.listdir(base)):
        libdir = os.path.join(base, name, "lib")
        if os.path.isdir(libdir):
            lib_dirs.append(libdir)
    return lib_dirs


def _merge_library_path(existing: str, extra_dirs: list[str]) -> str:
    """Prepend unique library directories while preserving existing order."""
    merged: list[str] = []
    seen: set[str] = set()

    for path in [*extra_dirs, *[p for p in existing.split(":") if p]]:
        if path and path not in seen:
            seen.add(path)
            merged.append(path)

    return ":".join(merged)


def _stdin_invocation() -> bool:
    """Return whether the current interpreter is executing code from stdin."""
    orig_argv = getattr(sys, "orig_argv", None)
    if orig_argv:
        return len(orig_argv) >= 2 and orig_argv[1] == "-"
    return sys.argv == ["-"]


def _reexec_argv() -> list[str] | None:
    """Build an argv that preserves the original Python invocation."""
    orig_argv = getattr(sys, "orig_argv", None)
    if orig_argv:
        if len(orig_argv) >= 2 and orig_argv[1] == "-":
            return None
        return [sys.executable, *orig_argv[1:]]

    if sys.argv == ["-"]:
        return None
    return [sys.executable, *sys.argv]


def _preload_nvidia_libs(lib_dirs: list[str]) -> None:
    """Best-effort preload of packaged NVIDIA shared objects.

    This fallback is mainly for stdin-based invocations, where re-exec cannot
    replay the script contents after mutating the environment.
    """
    for libdir in lib_dirs:
        for path in sorted(glob.glob(os.path.join(libdir, "*.so*"))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


def ensure_cuda_env() -> None:
    """Expose pip-installed NVIDIA libraries before JAX is imported."""
    if os.environ.get(_CUDA_SETUP_SENTINEL):
        return

    lib_dirs = _nvidia_lib_dirs()
    if not lib_dirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    current_dirs = {path for path in existing.split(":") if path}
    missing = [path for path in lib_dirs if path not in current_dirs]
    if not missing:
        os.environ[_CUDA_SETUP_SENTINEL] = "1"
        return

    os.environ["LD_LIBRARY_PATH"] = _merge_library_path(existing, lib_dirs)
    os.environ[_CUDA_SETUP_SENTINEL] = "1"

    argv = _reexec_argv()
    if argv is None or _stdin_invocation():
        _preload_nvidia_libs(lib_dirs)
        return

    os.execvpe(sys.executable, argv, os.environ)


def bootstrap_runtime() -> None:
    """Apply runtime environment tweaks before importing JAX-heavy modules."""

    ensure_cuda_env()
    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "jax"),
    )
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=2")
