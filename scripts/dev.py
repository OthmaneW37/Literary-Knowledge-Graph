"""Start the local API and Vite together; Ctrl+C stops only processes we started."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def listening(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.3)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    children = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        if not listening(8000):
            children.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000"],
                cwd=ROOT, creationflags=flags,
            ))
            deadline = time.monotonic() + 20
            while not listening(8000):
                if children[-1].poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("L’API n’a pas pu démarrer. Consulte les erreurs ci-dessus.")
                time.sleep(0.2)
        with urlopen("http://127.0.0.1:8000/api/health", timeout=10) as response:
            if "auth_enabled" not in json.load(response):
                raise RuntimeError("Le port 8000 est utilisé par une autre application.")
        if not listening(5173):
            npm = shutil.which("npm")
            if not npm or not (ROOT / "web/node_modules").is_dir():
                raise RuntimeError("Installe Node.js puis lance npm install dans web avant de réessayer.")
            command = [npm, "run", "dev"]
            if os.name == "nt":
                command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", npm, "run", "dev"]
            children.append(subprocess.Popen(command, cwd=ROOT / "web", creationflags=flags))
        print("NarrativeLens : http://127.0.0.1:5173 — Ctrl+C pour arrêter.", flush=True)
        while children:
            if any(child.poll() is not None for child in children):
                raise RuntimeError("Un des serveurs s’est arrêté. Consulte les erreurs ci-dessus.")
            time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        for child in reversed(children):
            if child.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=flags)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()


if __name__ == "__main__":
    raise SystemExit(main())
