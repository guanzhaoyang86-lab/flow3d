"""Load trusted project checkpoints across Windows and POSIX hosts.

Older training checkpoints contain concrete pathlib objects in train_config.
They are metadata, not local paths to open. A foreign path is therefore restored
as its matching PurePath, preserving its original spelling and path semantics.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import pickle
from types import ModuleType
from typing import Any, BinaryIO

import torch


class _PortablePathUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "pathlib":
            if name == "PosixPath" and os.name == "nt":
                return PurePosixPath
            if name == "WindowsPath" and os.name != "nt":
                return PureWindowsPath
        return super().find_class(module, name)


def _pickle_load(file: BinaryIO, **kwargs: Any) -> Any:
    return _PortablePathUnpickler(file, **kwargs).load()


# torch.load expects a pickle-compatible module. This private adapter changes
# no global pathlib classes and remains compatible with legacy Torch archives.
_PORTABLE_PICKLE = ModuleType("flow_observation.checkpoint_pickle")
_PORTABLE_PICKLE.Unpickler = _PortablePathUnpickler
_PORTABLE_PICKLE.load = _pickle_load


def load_trusted_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> Any:
    """Load a checkpoint produced by this project, preserving its file bytes.

    Like torch.load(weights_only=False), this uses unrestricted pickle. It is
    only intended for the user's own trusted training artifacts, not downloads
    from unknown publishers.
    """
    return torch.load(
        path,
        map_location=map_location,
        pickle_module=_PORTABLE_PICKLE,
        weights_only=False,
    )
