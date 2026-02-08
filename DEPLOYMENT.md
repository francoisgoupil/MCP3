# Railway Deployment Guide

## Quick Start

1. **Push to GitHub**: Make sure your code is in a Git repository
2. **Connect to Railway**: 
   - Go to https://railway.app
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
3. **Deploy**: Railway will automatically detect the Python project and deploy
4. **Get URL**: Railway will provide a public URL like `https://your-app.railway.app`

## Files Overview

- **server.py**: Main MCP server with all HistGradientBoostingClassifier tools
- **app.py**: ASGI wrapper for Railway deployment (exposes the FastMCP app)
- **Procfile**: Tells Railway how to start the server
- **requirements.txt**: Python dependencies
- **railway.json**: Railway configuration (optional)
- **runtime.txt**: Python version specification

## How It Works

1. Railway reads the `Procfile` which runs: `uvicorn app:app --host 0.0.0.0 --port $PORT`
2. Uvicorn imports `app.py` which imports `mcp` from `server.py`
3. The FastMCP app is exposed as an ASGI application
4. Railway provides the `PORT` environment variable automatically
5. The server binds to `0.0.0.0` to accept external connections

## Testing Your Deployment

Once deployed, test your MCP server:

```bash
# Using MCP Inspector
npx -y @modelcontextprotocol/inspector

# Then connect to: https://your-app.railway.app
```

Or use any MCP client with streamable-http transport pointing to your Railway URL.

## Troubleshooting

### Server won't start
- Check Railway logs: `railway logs` or view in Railway dashboard
- Ensure `requirements.txt` includes all dependencies
- Verify Python version in `runtime.txt` matches Railway's supported versions

### Port binding issues
- Railway automatically sets `PORT` - don't hardcode it
- Ensure `Procfile` uses `$PORT` variable
- Server must bind to `0.0.0.0`, not `127.0.0.1`

### Import errors
- Verify `mcp` package is installed (in requirements.txt)
- Check that `server.py` doesn't have syntax errors
- Ensure all imports are available in requirements.txt

## Environment Variables

Railway automatically provides:
- `PORT`: Port to listen on (required)
- `HOST`: Can be set to `0.0.0.0` (default in Procfile)

No additional environment variables needed for basic operation.
