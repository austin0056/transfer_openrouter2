"""Zeabur / PaaS entry: honor PORT env (default 8000)."""

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WEB_CONCURRENCY", min(os.cpu_count() or 1, 4)))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, workers=workers)


if __name__ == "__main__":
    main()
