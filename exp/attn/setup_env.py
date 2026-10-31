#!/usr/bin/env python3
"""Create a local benchmark venv and reuse an existing CUDA PyTorch installation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-python", default=sys.executable)
    parser.add_argument("--venv", type=Path, default=ROOT / "build/attn-venv")
    args = parser.parse_args()
    probe = subprocess.check_output(
        [
            args.torch_python,
            "-c",
            "import json,torch; from pathlib import Path; "
            "print(json.dumps({'version':torch.__version__,'site':str(Path(torch.__file__).resolve().parents[1])}))",
        ],
        text=True,
    )
    installed = json.loads(probe)
    subprocess.run([args.torch_python, "-m", "venv", str(args.venv)], check=True)
    python = args.venv / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    target = subprocess.check_output(
        [str(python), "-c", "import site;print(site.getsitepackages()[0])"], text=True
    ).strip()
    (Path(target) / "existing_torch_environment.pth").write_text(
        installed["site"] + "\n"
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--cache-dir",
            str(ROOT / "build/attn-cache/pip"),
            "-r",
            str(Path(__file__).with_name("requirements-measured.txt")),
        ],
        check=True,
    )
    print(f"Environment: {python}; reusing PyTorch {installed['version']}")


if __name__ == "__main__":
    main()
