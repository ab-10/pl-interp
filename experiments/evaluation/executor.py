"""Sandboxed subprocess execution. Runs generated code with timeout and process isolation."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile


def execute_code(script: str, timeout: int = 3) -> tuple[bool, str, int]:
    """Execute a Python script in a subprocess with timeout.

    Args:
        script: Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        (passed, stderr_output, exit_code)
        - passed: True if exit code is 0.
        - stderr_output: stderr text on failure, "timeout" on timeout, "" on success.
        - exit_code: process exit code, or -1 on timeout.
    """
    if not script or not script.strip():
        return (False, "empty script", 1)

    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="exec_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)

        proc = subprocess.Popen(
            ["python", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError, PermissionError):
                proc.kill()
            proc.wait()
            return (False, "timeout", -1)

        if proc.returncode == 0:
            return (True, "", 0)
        else:
            stderr_text = stderr.strip() if stderr else ""
            return (False, stderr_text, proc.returncode)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
