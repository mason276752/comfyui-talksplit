#!/usr/bin/env python3
"""One-shot installer for talksplit.

If invoked from inside an active venv (including ComfyUI's), installs into
that environment. Otherwise creates a local .venv. Pre-downloads the
embedding model so first run doesn't stall on a 2GB download.

Usage:
    python3 INSTALL.py                                    # default: BGE-M3
    python3 INSTALL.py --model intfloat/multilingual-e5-small
    python3 INSTALL.py --skip-model                       # deps only
    python3 INSTALL.py --skip-venv                        # use current Python (e.g. ComfyUI portable)
    python3 INSTALL.py --force-venv                       # force a separate .venv
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
VENV = REPO / ".venv"
DEFAULT_MODEL = "BAAI/bge-m3"
MIN_PY = (3, 10)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HuggingFace model to pre-download (default: {DEFAULT_MODEL}).")
    p.add_argument("--python", default=None,
                   help="Python interpreter for venv (default: auto-detect 3.10+).")
    p.add_argument("--skip-venv", action="store_true",
                   help="Install into the current interpreter; skip .venv even if none is active.")
    p.add_argument("--force-venv", action="store_true",
                   help="Create a local .venv even if a venv is already active.")
    p.add_argument("--skip-model", action="store_true",
                   help="Don't pre-download the embedding model.")
    p.add_argument("--extras", default="dev,plot",
                   help='Comma-separated extras to install (default: "dev,plot").')
    args = p.parse_args()

    py, used_local_venv = _prepare_interpreter(args.python, args.skip_venv, args.force_venv)
    _install_package(py, args.extras)
    if not args.skip_model:
        _download_model(py, args.model)

    print("\n[done] Setup complete.")
    if used_local_venv:
        print(f"  source {VENV.relative_to(REPO)}/bin/activate")
    print("  talksplit tests/fixtures/sample_zh.txt")
    return 0


def _prepare_interpreter(requested: str | None, skip_venv: bool, force_venv: bool) -> tuple[str, bool]:
    """Return (python_path, used_local_venv)."""
    use_current = skip_venv or (_in_active_venv() and not force_venv)
    if use_current:
        if sys.version_info < MIN_PY:
            sys.exit(f"[error] Need Python >= {'.'.join(map(str, MIN_PY))}, "
                     f"current is {sys.version.split()[0]}.")
        env = _active_env_path()
        if env:
            print(f"[venv] using active environment at {env}")
        else:
            print(f"[venv] using current interpreter {sys.executable}")
        return sys.executable, False

    base_py = requested or _find_python()
    if VENV.exists():
        print(f"[venv] reusing existing {VENV}")
    else:
        print(f"[venv] creating {VENV} with {base_py}")
        subprocess.check_call([base_py, "-m", "venv", str(VENV)])
    py = str(VENV / "bin" / "python")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    return py, True


def _in_active_venv() -> bool:
    if "VIRTUAL_ENV" in os.environ or "CONDA_PREFIX" in os.environ:
        return True
    return getattr(sys, "real_prefix", None) is not None or sys.prefix != sys.base_prefix


def _active_env_path() -> str | None:
    # Prefer sys.prefix when it differs from base_prefix — authoritative for
    # the running interpreter. VIRTUAL_ENV / CONDA_PREFIX env vars can be stale.
    if sys.prefix != sys.base_prefix:
        return sys.prefix
    return os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")


def _find_python() -> str:
    if sys.version_info >= MIN_PY:
        return sys.executable
    for cand in ("python3.13", "python3.12", "python3.11", "python3.10"):
        path = shutil.which(cand)
        if path:
            return path
    sys.exit(f"[error] No Python >= {'.'.join(map(str, MIN_PY))} found. "
             f"Pass --python /path/to/python.")


def _install_package(py: str, extras: str) -> None:
    target = f".[{extras}]" if extras else "."
    print(f"[deps] installing {target}")
    subprocess.check_call([py, "-m", "pip", "install", "-e", target])


def _download_model(py: str, model: str) -> None:
    print(f"[model] pre-downloading {model}")
    code = (
        "from sentence_transformers import SentenceTransformer;"
        f"m = SentenceTransformer({model!r});"
        "v = m.encode(['probe'], normalize_embeddings=True);"
        "print(f'[model] ready, dim={v.shape[1]}')"
    )
    subprocess.check_call([py, "-c", code])


if __name__ == "__main__":
    raise SystemExit(main())
