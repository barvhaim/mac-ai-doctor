from __future__ import annotations

import platform
import subprocess

from .models import SystemInfo


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def detect_system() -> SystemInfo:
    os_name = platform.system()
    if os_name != "Darwin":
        return SystemInfo(
            os_name,
            None,
            platform.machine(),
            None,
            False,
            "Automatic unified-memory detection is supported only on macOS; use --memory-gb.",
        )
    chip = _sysctl("machdep.cpu.brand_string")
    model = _sysctl("hw.model")
    raw = _sysctl("hw.memsize")
    memory = int(raw) / 1_000_000_000 if raw and raw.isdigit() else None
    apple = platform.machine() == "arm64" or bool(chip and "Apple" in chip)
    return SystemInfo(
        os_name,
        chip,
        model,
        memory,
        apple,
        "Apple Silicon detected." if apple else "This Mac is not detected as Apple Silicon.",
    )
