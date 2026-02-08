# MCP Server Connection Guide

## 🚀 Your MCP Server is Live!

**Deployment URL:** `https://web-production-a620a.up.railway.app`

## Connecting to Your MCP Server

### Option 1: MCP Inspector (Testing)

Test your server interactively:

```bash
npx -y @modelcontextprotocol/inspector
```

Then connect to: `https://web-production-a620a.up.railway.app`

### Option 2: Claude Desktop

Add to your Claude Desktop config file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "histgradientboosting": {
      "url": "https://web-production-a620a.up.railway.app",
      "transport": "streamable-http"
    }
  }
}
```

Restart Claude Desktop after making changes.

### Option 3: MCP Client (Python)

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# For streamable HTTP
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "https://web-production-a620a.up.railway.app/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1
        }
    )
    print(response.json())
```

## Available Tools

Your server exposes these 10 tools:

1. **create_classifier** - Create a new classifier instance
2. **train_model** - Train the classifier on data
3. **predict** - Make class predictions
4. **predict_proba** - Get class probabilities
5. **score_model** - Evaluate model accuracy
6. **get_model_info** - Get model details and parameters
7. **list_models** - List all available models
8. **delete_model** - Remove a model from memory
9. **save_model** - Serialize model to base64
10. **load_model** - Load model from serialized string

## Quick Test

Try creating and training a simple model:

```python
# 1. Create a classifier
create_classifier(
    model_id="test_model",
    learning_rate=0.1,
    max_iter=50
)

# 2. Train it
train_model(
    model_id="test_model",
    X=[[1, 2], [3, 4], [5, 6], [7, 8]],
    y=[0, 0, 1, 1]
)

# 3. Make predictions
predict(
    model_id="test_model",
    X=[[2, 3], [6, 7]]
)
```

## Troubleshooting

### Server Not Responding
- Check Railway logs: Go to your Railway dashboard → View logs
- Verify the deployment is running (should show "Active" status)
- Check that the `Procfile` is correct

### Connection Errors
- Ensure you're using `streamable-http` transport (not `sse` or `stdio`)
- Verify the URL is correct: `https://web-production-a620a.up.railway.app`
- Check Railway logs for any startup errors

### Model Not Found Errors
- Models are stored in-memory and reset when the server restarts
- Create the model first using `create_classifier` before training
- Use `list_models` to see what models are currently available

## Railway Dashboard

Monitor your deployment:
- **URL:** https://railway.app
- View logs, metrics, and manage your deployment
- Check resource usage and scaling options
