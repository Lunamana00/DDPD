"""Test harness compatibility fixes."""

from __future__ import annotations

import os


if os.name == "nt":
    _original_mkdir = os.mkdir

    def _mkdir_without_restrictive_windows_mode(path, mode=0o777, *args, **kwargs):
        # In this sandbox, Windows directories created with mode 0o700 can become
        # unreadable to the same process. Pytest uses that mode for tmp_path.
        if mode == 0o700:
            mode = 0o777
        return _original_mkdir(path, mode, *args, **kwargs)

    os.mkdir = _mkdir_without_restrictive_windows_mode
