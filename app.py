"""
ASGI application wrapper for Railway deployment
This file exposes the FastMCP app as an ASGI application
"""

# Import the mcp instance from server module
# When uvicorn imports this module, __name__ will be "app", not "__main__"
# So server.py's __main__ block won't execute
from server import mcp

# Get the ASGI app from FastMCP for streamable HTTP
try:
    app = mcp.streamable_http_app()
except AttributeError:
    # Fallback: create a simple ASGI wrapper
    # This shouldn't happen with recent FastMCP versions
    raise RuntimeError(
        "FastMCP streamable_http_app() not available. "
        "Please ensure you're using mcp>=1.0.0"
    )
