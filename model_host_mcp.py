"""Global Codex MCP tool for active-turn model control on the shared app-server."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from host_control import ModelHostError, switch_current_turn_model


mcp = FastMCP("codex_model_control")


@mcp.tool(
    name="switch_current_turn_model",
    annotations={"title": "Switch current Codex turn model", "readOnlyHint": False,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def switch_current_turn_model_tool(thread_id: str, model: str, effort: str | None = None) -> str:
    """Switch later inference steps of the *current* active chat turn.

    Supply the exact current CODEX_THREAD_ID, a model ID from this host, and an
    optional supported reasoning effort. Never pass a different chat's ID.
    The tool fails closed if the thread is not active on the shared model host.
    An applied result is not proof that a later inference occurred.
    """
    try:
        result = await switch_current_turn_model(thread_id, model, effort)
    except ModelHostError as exc:
        result = {"status": "not_applied", "reason": str(exc)}
    return json.dumps(result, separators=(",", ":"))


if __name__ == "__main__":
    mcp.run(transport="stdio")
