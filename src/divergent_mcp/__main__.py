"""Entrypoint for the divergent MCP server (stdio)."""

from __future__ import annotations

import argparse
import asyncio

from .server import mcp, tool_names


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="divergent-mcp",
        description="Serve divergent ideation tools over MCP. Reads OPENROUTER_API_KEY.",
    )
    parser.add_argument("--stdio", action="store_true", help="Serve over stdio (the default)")
    parser.add_argument("--list-tools", action="store_true", help="Print tool names and exit")
    args = parser.parse_args()

    if args.list_tools:
        print("\n".join(asyncio.run(tool_names())))
        return
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
