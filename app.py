"""
ASGI application wrapper for Railway deployment
This file properly integrates FastMCP with FastAPI for HTTP transport
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from server import mcp
import json

# Create FastAPI app
app = FastAPI(title="HistGradientBoostingClassifier MCP Server")

# Add CORS middleware for MCP clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "server": "HistGradientBoostingClassifier MCP Server",
        "mcp_endpoint": "/mcp",
        "protocol": "streamable-http"
    }

@app.api_route("/mcp", methods=["GET", "POST", "OPTIONS"])
async def mcp_endpoint(request: Request):
    """
    MCP protocol endpoint for streamable HTTP transport
    FastMCP handles MCP protocol - we route requests to it
    """
    try:
        # Get request method and body
        method = request.method
        
        if method == "OPTIONS":
            return JSONResponse({"status": "ok"})
        
        # Get request body for POST requests
        if method == "POST":
            try:
                body = await request.json()
            except:
                body = {}
        else:
            # GET request - might be SSE connection
            # Return server info for initialization
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": None,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "HistGradientBoostingClassifier MCP Server",
                        "version": "1.0.0"
                    }
                }
            })
        
        # Handle MCP protocol requests
        # FastMCP should process these, but we need to call it properly
        # For now, return a proper MCP response structure
        if "method" in body:
            method_name = body.get("method")
            
            # Handle initialize request
            if method_name == "initialize":
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "HistGradientBoostingClassifier MCP Server",
                            "version": "1.0.0"
                        }
                    }
                })
            
            # Handle notifications/initialized (some clients send this after initialize)
            elif method_name == "notifications/initialized":
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {}
                })

            # Handle tools/list request
            elif method_name == "tools/list":
                try:
                    tool_models = await mcp.list_tools()
                    tools = [tool.model_dump(by_alias=True, exclude_none=True) for tool in tool_models]
                except Exception:
                    tools = []

                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "tools": tools
                    }
                })
            
            # Handle tools/call request
            elif method_name == "tools/call":
                params = body.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                try:
                    result = await mcp.call_tool(tool_name, arguments)
                    if isinstance(result, dict) and "content" in result:
                        content = result["content"]
                    elif isinstance(result, list):
                        content = [
                            block.model_dump(by_alias=True, exclude_none=True)
                            if hasattr(block, "model_dump") else block
                            for block in result
                        ]
                    else:
                        content = [
                            {
                                "type": "text",
                                "text": json.dumps(result) if not isinstance(result, str) else result
                            }
                        ]

                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "result": {
                            "content": content
                        }
                    })
                except Exception as e:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "error": {
                            "code": -32603,
                            "message": f"Error calling tool: {str(e)}"
                        }
                    }, status_code=500)
            
            # Handle other MCP methods
            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method_name}"
                    }
                }, status_code=400)
        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {
                    "code": -32600,
                    "message": "Invalid Request"
                }
            }, status_code=400)
            
    except Exception as e:
        import traceback
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}",
                "data": traceback.format_exc()
            }
        }, status_code=500)
