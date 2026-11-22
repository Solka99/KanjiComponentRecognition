"""Record which code, inputs and hardware produced a result, for the experiment history."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


def _git(*arguments):
    try:
        return subprocess.run(["git", *arguments], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "cpu"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(input_files=()):
    """Git commit (plus whether tracked files had uncommitted changes), command, inputs, hardware."""
    uncommitted = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_uncommitted_changes": None if uncommitted is None else bool(uncommitted),
        "inputs": {str(path): file_sha256(path) for path in input_files},
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else _cpu_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def write_provenance(directory, input_files=()):
    record = provenance(input_files)
    Path(directory, "provenance.json").write_text(json.dumps(record, indent=1))
    return record
