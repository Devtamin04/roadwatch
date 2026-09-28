"""Print host configuration relevant to CPU inference.

Reports CPU model, core counts, total RAM, and versions of Python,
onnxruntime, OpenCV, numpy and scipy. Missing packages are reported
rather than raising, so this can run before requirements are installed.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys


def cpu_model() -> str:
    """Return the CPU model name, falling back to platform info."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def physical_cores() -> int | None:
    """Count distinct (physical id, core id) pairs on Linux; None elsewhere."""
    try:
        cores = set()
        phys = core = None
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    if core is not None:
                        cores.add((phys, core))
                    phys = core = None
        return len(cores) or None
    except OSError:
        return None


def total_ram_gb() -> float | None:
    """Total system RAM in GiB, or None if it cannot be determined."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except (ValueError, OSError, AttributeError):
        return None


def package_version(module: str) -> str:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return "NOT INSTALLED"
    return getattr(mod, "__version__", "unknown")


def collect() -> dict[str, str]:
    """Collect environment info as an ordered mapping of label -> value."""
    ram = total_ram_gb()
    phys = physical_cores()
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": cpu_model(),
        "Logical cores": str(os.cpu_count()),
        "Physical cores": str(phys) if phys else "unknown",
        "RAM (GiB)": f"{ram:.1f}" if ram else "unknown",
        "Python": platform.python_version(),
        "onnxruntime": package_version("onnxruntime"),
        "OpenCV": package_version("cv2"),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
    }
    try:
        import onnxruntime as ort

        info["ORT providers"] = ", ".join(ort.get_available_providers())
    except ImportError:
        pass
    return info


def main() -> int:
    info = collect()
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{width}} : {value}")
    missing = [k for k, v in info.items() if v == "NOT INSTALLED"]
    if missing:
        print(f"\nMissing packages: {', '.join(missing)} -> pip install -r requirements.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
