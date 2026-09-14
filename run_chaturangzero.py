"""Run the Python rules API and Vite UI as one local application."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from shared.game_core.server import create_server


def main() -> int:
    project_root = Path(__file__).resolve().parent
    ui_dir = project_root / "shared" / "ui"
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        print("Node.js and npm are required to start ChaturangZero.", file=sys.stderr)
        return 1

    try:
        server = create_server()
    except OSError as error:
        print(f"Could not start the game core on port 8000: {error}", file=sys.stderr)
        return 1

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print("Game core ready at http://127.0.0.1:8000")
    print("UI ready at http://127.0.0.1:5173")

    command = [npm, "run", "dev", "--", "--host", "127.0.0.1"]
    if "CHATURANGZERO_NO_OPEN" not in os.environ:
        command.append("--open")

    try:
        return subprocess.call(command, cwd=ui_dir)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())

