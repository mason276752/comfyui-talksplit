"""ComfyUI custom-node entry point.

Loads ``src/split/`` as a renamed internal package (``_talksplit_internal``)
so the inner ``split`` name never collides with the folder this gets cloned
into. Whatever ComfyUI calls our folder, the import works.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_SPLIT = os.path.join(_HERE, "src", "split")
_INTERNAL = "_talksplit_internal"


def _load_internal_pkg():
    if _INTERNAL in sys.modules:
        return sys.modules[_INTERNAL]
    spec = importlib.util.spec_from_file_location(
        _INTERNAL,
        os.path.join(_SRC_SPLIT, "__init__.py"),
        submodule_search_locations=[_SRC_SPLIT],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not locate {_SRC_SPLIT}/__init__.py")
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_INTERNAL] = pkg
    spec.loader.exec_module(pkg)
    return pkg


def _load_nodes_module():
    sub_name = f"{_INTERNAL}.comfy_nodes"
    if sub_name in sys.modules:
        return sys.modules[sub_name]
    _load_internal_pkg()
    spec = importlib.util.spec_from_file_location(
        sub_name, os.path.join(_SRC_SPLIT, "comfy_nodes.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not locate {_SRC_SPLIT}/comfy_nodes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[sub_name] = mod
    spec.loader.exec_module(mod)
    return mod


_nodes = _load_nodes_module()
NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
