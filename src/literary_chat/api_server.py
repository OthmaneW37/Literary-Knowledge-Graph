from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
