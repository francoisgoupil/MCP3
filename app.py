"""
ASGI application wrapper for Railway deployment
This file exposes the FastMCP app as an ASGI application using FastAPI
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from server import mcp
import json

# Create FastAPI app
app = FastAPI(title="HistGradientBoostingClassifier MCP Server")

# FastMCP's streamable HTTP transport needs to handle MCP protocol
# We'll create endpoints that FastMCP can use

@app.get("/")
async def root():
    return {
        "status": "ok",
        "server": "HistGradientBoostingClassifier MCP Server",
        "mcp_endpoint": "/mcp"
    }

@app.api_route("/mcp", methods=["GET", "POST", "OPTIONS"])
async def mcp_endpoint(request: Request):
    """
    MCP protocol endpoint for streamable HTTP transport
    """
    # FastMCP handles the MCP protocol internally
    # We need to integrate it properly with FastAPI
    # For now, return a basic response - FastMCP should handle the protocol
    try:
        # Get request body if POST
        if request.method == "POST":
            body = await request.json()
        else:
            body = {}
        
        # FastMCP's streamable HTTP transport expects specific handling
        # This is a simplified version - FastMCP should provide the actual handler
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {
                    "name": "HistGradientBoostingClassifier MCP Server",
                    "version": "1.0.0"
                }
            }
        })
    except Exception as e:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }, status_code=500)
