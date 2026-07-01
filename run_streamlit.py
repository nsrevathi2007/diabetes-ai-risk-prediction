"""Launch the Streamlit dashboard."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    """Start the Streamlit application."""
    app_path = os.path.join(os.path.dirname(__file__), "frontend", "web", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", "8501", "--server.address", "127.0.0.1"]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
