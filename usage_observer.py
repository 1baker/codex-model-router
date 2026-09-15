"""Capture exact app-server usage for a managed thread's initial turn."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets

from host_control import HOST_URL, _read_token, _rpc
from telemetry import record as record_metric, usage_from


async def initialize(ws: websockets.ClientConnection) -> None:
    await _rpc(ws, "initialize", {"clientInfo": {"name": "modellabs-usage-observer", "version": "0.1"},
                                  "capabilities": {"experimentalApi": True}}, 1)
    await ws.send(json.dumps({"method": "initialized", "params": {}}))


async def observe(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + 20
    ws = None
    while time.monotonic() < deadline:
        try:
            ws = await websockets.connect(HOST_URL, additional_headers={"Authorization": f"Bearer {_read_token()}"},
                                          open_timeout=2, close_timeout=1, max_size=8 * 1024 * 1024)
            await initialize(ws)
            await _rpc(ws, "thread/resume", {"threadId": args.thread_id}, 2)
            break
        except Exception:
            if ws is not None:
                await ws.close()
            ws = None
            await asyncio.sleep(0.1)
    if ws is None:
        record_metric("usage_observer_unavailable", thread_id=args.thread_id, turn_id=args.turn_id,
                      model=args.model, effort=args.effort)
        return
    try:
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=15 * 60))
            params = message.get("params") or {}
            event_turn_id = params.get("turnId") or (params.get("turn") or {}).get("id")
            if params.get("threadId") != args.thread_id or event_turn_id != args.turn_id:
                continue
            if message.get("method") == "rawResponse/completed":
                record_metric("response_usage", thread_id=args.thread_id, turn_id=args.turn_id,
                              model=args.model, effort=args.effort, usage=usage_from(params))
            elif message.get("method") == "turn/completed":
                turn = params.get("turn") or {}
                record_metric("turn_completed", thread_id=args.thread_id, turn_id=args.turn_id,
                              model=args.model, effort=args.effort, task_class=args.task_class,
                              elapsed_ms=turn.get("durationMs"), status=turn.get("status"), usage=None)
                return
    finally:
        await ws.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--turn-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--task-class", required=True)
    asyncio.run(observe(parser.parse_args()))


if __name__ == "__main__":
    main()
