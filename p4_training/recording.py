"""Small, JSON-safe provenance and atomic text-file helpers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def provenance(device: str) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(
                ['git', *args], cwd=ROOT, stderr=subprocess.DEVNULL
            ).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    # Include uncommitted AI code too: a commit alone cannot identify a dirty run.
    digest = hashlib.sha256()
    for directory in ('shared/game_core', 'p1_representation', 'p2_neural_network',
                      'p3_mcts', 'p4_training'):
        for path in sorted((ROOT / directory).rglob('*.py')):
            digest.update(path.relative_to(ROOT).as_posix().encode() + b'\0')
            digest.update(path.read_bytes() + b'\0')
    return {
        'git_commit': git('rev-parse', 'HEAD'),
        'git_status': git('status', '--porcelain'),
        'ai_source_sha256': digest.hexdigest(),
        'python': sys.version, 'platform': platform.platform(),
        'torch': str(torch.__version__), 'numpy': str(np.__version__),
        'device': str(device), 'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
    }


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def json_text(value) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + '\n'
