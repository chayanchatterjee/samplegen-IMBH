# utils/noise_index.py
from __future__ import annotations
import os, json, time
from dataclasses import dataclass, asdict
from typing import List, Optional

HDF_EXTS = {".h5", ".hdf", ".hdf5"}

@dataclass
class NoiseFile:
    path: str
    size: int
    mtime: float
    gps_start: Optional[float] = None
    duration: Optional[float] = None

def _walk_hdf_paths(root: str) -> List[str]:
    if os.path.isfile(root):
        ext = os.path.splitext(root)[1].lower()
        return [root] if ext in HDF_EXTS else []
    paths, stack = [], [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file() and os.path.splitext(e.name)[1].lower() in HDF_EXTS:
                        paths.append(e.path)
        except (PermissionError, FileNotFoundError):
            continue
    return paths

def _snapshot_sig(files: List[NoiseFile]) -> str:
    total = sum(f.size for f in files)
    latest = max((f.mtime for f in files), default=0.0)
    return f"{len(files)}:{total}:{int(latest)}"

def build_index(root: str) -> dict:
    if not os.path.exists(root):
        raise FileNotFoundError(f"background_data_directory does not exist: {root}")
    paths = _walk_hdf_paths(root)
    files = [NoiseFile(p, os.path.getsize(p), os.path.getmtime(p)) for p in paths]
    return {
        "root": os.path.abspath(root),
        "created": time.time(),
        "signature": _snapshot_sig(files),
        "files": [asdict(f) for f in files],
    }

def _try_build_signature(root: str) -> Optional[str]:
    paths = _walk_hdf_paths(root)
    files = [NoiseFile(p, os.path.getsize(p), os.path.getmtime(p)) for p in paths]
    return _snapshot_sig(files)

def load_index(cache_path: str) -> Optional[dict]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            idx = json.load(f)
        root = idx.get("root")
        if not root or not os.path.exists(root):
            return None
        current_sig = _try_build_signature(root)
        if current_sig and current_sig == idx.get("signature"):
            return idx
    except Exception:
        pass
    return None

def save_index(index: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f)
    os.replace(tmp, cache_path)
