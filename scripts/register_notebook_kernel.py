"""Đăng ký kernelspec Jupyter ``citd-ml`` trỏ vào .venv của dự án.

Notebook ``Do_An_Meta_Labeling_BTCUSD.ipynb`` khai báo kernel này trong metadata.
Kernelspec được ghi vào ``.venv/share/jupyter/kernels/`` (thư mục bị gitignore) nên
không lây sang các dự án khác trên máy.

Chạy sau khi ``uv sync --extra dev``:

    uv run python scripts/register_notebook_kernel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KERNEL_NAME = "citd-ml"
KERNEL_DISPLAY_NAME = "Python 3.12 (citd-ml)"


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    prefix = project_root / ".venv"

    if not prefix.is_dir():
        print(
            f"Không tìm thấy {prefix}. Hãy chạy `uv sync --extra dev` trước.",
            file=sys.stderr,
        )
        return 1

    try:
        import ipykernel  # noqa: F401
    except ModuleNotFoundError:
        print(
            "Thiếu ipykernel. Hãy chạy `uv sync --extra dev` trước.",
            file=sys.stderr,
        )
        return 1

    from ipykernel.kernelspec import install

    spec_dir = install(
        user=False,
        prefix=str(prefix),
        kernel_name=KERNEL_NAME,
        display_name=KERNEL_DISPLAY_NAME,
    )
    spec_path = Path(spec_dir) / "kernel.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    print(f"Đã đăng ký kernelspec: {spec_path}")
    print(f"  argv: {' '.join(spec['argv'])}")
    print()
    print("Mở notebook:")
    print("  uv run jupyter lab Do_An_Meta_Labeling_BTCUSD.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
