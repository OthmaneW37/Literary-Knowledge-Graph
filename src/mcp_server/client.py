"""Allowlisted MCP stdio client. Tool names and executable are never model supplied."""
import asyncio
import json
import os
import sys
from pathlib import Path
from security.budget import consume

TOOLS = {'get_reading_progress', 'get_user_annotations', 'get_book_metadata', 'prepare_annotation', 'save_annotation'}


async def call_tool_async(tool, arguments, data_dir, user_id, secret):
    if tool not in TOOLS:
        raise ValueError('Outil MCP non autorisé.')
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    budget = consume('mcp')
    timeout = min(30, budget.remaining()) if budget else 30
    # The child receives only runtime essentials and its exact user scope.
    env = {key: value for key, value in os.environ.items() if key.upper() in {'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'USERPROFILE'}}
    env.update({'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
                'PYTHONIOENCODING': 'utf-8', 'NARRATIVELENS_DATA_DIR': str(Path(data_dir).resolve()),
                'NARRATIVELENS_USER_ID': user_id, 'APPROVAL_SECRET': secret})
    params = StdioServerParameters(command=sys.executable, args=['-m', 'mcp_server.server'], env=env)
    async with asyncio.timeout(timeout):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments)
                if result.isError:
                    raise ValueError('L’outil MCP a refusé la demande ou ses preuves.')
                structured = getattr(result, 'structuredContent', None)
                if structured is not None:
                    return structured.get('result', structured)
                parts = [item.text for item in result.content if item.type == 'text']
                return json.loads('\n'.join(parts))


def call_tool(tool, arguments, data_dir, user_id, secret):
    try:
        return asyncio.run(call_tool_async(tool, arguments, data_dir, user_id, secret))
    except ExceptionGroup as exc:
        raise ValueError('Le service MCP a refusé la demande ou est indisponible.') from exc
