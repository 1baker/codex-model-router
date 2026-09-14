"""Keep the current ModelLabs proxy healthy without replacing live peers."""

from __future__ import annotations

import os
import sys
import time

from model_host_launcher import ensure_proxy


def main() -> None:
    interval = float(os.environ.get("MODELLABS_PROXY_SUPERVISOR_INTERVAL", "5"))
    if interval < 1:
        raise ValueError("MODELLABS_PROXY_SUPERVISOR_INTERVAL must be at least one second.")
    while True:
        try:
            ensure_proxy()
        except Exception as exc:
            print(f"ModelLabs proxy supervisor: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
