"""Run the Python rules API and Vite UI as one local application."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from shared.game_core.server import create_server


def _guard_windows_child(child: subprocess.Popen[bytes]) -> int | None:
    """Keep Vite tied to this launcher even if its terminal is closed abruptly."""
    if os.name != "nt":
        return None

    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("per_process_time", ctypes.c_longlong), ("per_job_time", ctypes.c_longlong),
            ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
            ("max_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t), ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "read_operations", "write_operations", "other_operations",
            "read_bytes", "write_bytes", "other_bytes",
        )]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", BasicLimits), ("io", IoCounters),
            ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    if not kernel.AssignProcessToJobObject(job, child._handle):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    return job


def main() -> int:
    project_root = Path(__file__).resolve().parent
    ui_dir = project_root / "shared" / "ui"
    node = os.environ.get("CHATURANGZERO_NODE") or "node"
    vite = ui_dir / "node_modules" / "vite" / "bin" / "vite.js"
    if not vite.is_file():
        print("Vite is missing. Run npm install in shared/ui first.", file=sys.stderr)
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

    command = [node, str(vite), "--host", "127.0.0.1"]
    if "CHATURANGZERO_NO_OPEN" not in os.environ:
        command.append("--open")

    child: subprocess.Popen[bytes] | None = None
    child_job: int | None = None
    try:
        child = subprocess.Popen(
            command, cwd=ui_dir,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        child_job = _guard_windows_child(child)
        return child.wait()
    except KeyboardInterrupt:
        return 0
    except OSError as error:
        print(f"Could not start the UI: {error}", file=sys.stderr)
        return 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if child_job is not None:
            import ctypes
            from ctypes import wintypes
            ctypes.WinDLL("kernel32").CloseHandle(wintypes.HANDLE(child_job))
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())

