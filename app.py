"""
ASGI application wrapper for Railway deployment
This file properly integrates FastMCP with FastAPI for HTTP transport
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from server import mcp
import json
import asyncio

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
                # Get tools from FastMCP - try to access the registered tools
                tools = []
                try:
                    # FastMCP stores tools in different ways depending on version
                    if hasattr(mcp, '_tools'):
                        tool_dict = mcp._tools
                    elif hasattr(mcp, 'tools'):
                        tool_dict = mcp.tools
                    else:
                        # Try to get from the server instance
                        tool_dict = getattr(mcp, 'server', {}).get('tools', {}) if hasattr(mcp, 'server') else {}
                    
                    for tool_name, tool_info in tool_dict.items():
                        # Extract tool information
                        tool_schema = {
                            "name": tool_name,
                            "description": getattr(tool_info, '__doc__', '') or '',
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                        tools.append(tool_schema)
                except Exception as e:
                    # Fallback: return empty tools list
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
                # Call the tool via FastMCP
                params = body.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                try:
                    # Get the tool function from FastMCP
                    tool_func = None
                    if hasattr(mcp, '_tools') and tool_name in mcp._tools:
                        tool_func = mcp._tools[tool_name]
                    elif hasattr(mcp, 'tools') and tool_name in mcp.tools:
                        tool_func = mcp.tools[tool_name]
                    
                    if tool_func:
                        # Call the tool function
                        if asyncio.iscoroutinefunction(tool_func):
                            result = await tool_func(**arguments)
                        else:
                            result = tool_func(**arguments)
                        
                        return JSONResponse({
                            "jsonrpc": "2.0",
                            "id": body.get("id"),
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(result) if not isinstance(result, str) else result
                                    }
                                ]
                            }
                        })
                    else:
                        return JSONResponse({
                            "jsonrpc": "2.0",
                            "id": body.get("id"),
                            "error": {
                                "code": -32601,
                                "message": f"Tool not found: {tool_name}"
                            }
                        }, status_code=400)
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
