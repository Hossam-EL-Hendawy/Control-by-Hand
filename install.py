#!/usr/bin/env python3
"""Create the project's virtual environment and install its dependencies.

Run with Python 3.10, 3.11, or 3.12. pip automatically avoids reinstalling
packages that already satisfy requirements.txt.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import venv
from pathlib import Path


MIN_PYTHON = (3, 10)
MAX_PYTHON_EXCLUSIVE = (3, 13)
PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"


def supported_python() -> bool:
    version = sys.version_info[:2]
    return MIN_PYTHON <= version < MAX_PYTHON_EXCLUSIVE


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main() -> int:
    system = platform.system()
    print(f"Operating system: {system} ({platform.machine()})")
    print(f"Python interpreter: {sys.executable} ({platform.python_version()})")

    if not supported_python():
        print(
            "\nUnsupported Python version. Install/use Python 3.10, 3.11, or 3.12, "
            "then rerun this installer.\n"
            "macOS/Linux example: python3.12 install.py\n"
            "Windows example:     py -3.12 install.py",
            file=sys.stderr,
        )
        return 2

    if not venv_python().exists():
        print(f"Creating virtual environment: {VENV_DIR.name}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print(f"Using existing virtual environment: {VENV_DIR.name}")

    print("Installing missing or incompatible dependencies from requirements.txt...")
    command = [
        str(venv_python()),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade-strategy",
        "only-if-needed",
        "-r",
        str(PROJECT_DIR / "requirements.txt"),
    ]
    result = subprocess.run(command, cwd=PROJECT_DIR, check=False)
    if result.returncode != 0:
        print("\nInstallation failed. Read the pip output above for the specific cause.", file=sys.stderr)
        return result.returncode

    if system == "Windows":
        activate = ".venv\\Scripts\\Activate.ps1"
    else:
        activate = "source .venv/bin/activate"

    print("\nInstallation complete. Start the project with:")
    print(f"  {activate}")
    print("  python hand_control.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
