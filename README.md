# HistGradientBoostingClassifier MCP Server

A Model Context Protocol (MCP) server that provides tools for training, predicting, and managing sklearn's `HistGradientBoostingClassifier` models.

## Features

This MCP server exposes the following tools:

- **create_classifier**: Create a new HistGradientBoostingClassifier with custom parameters
- **train_model**: Train a classifier on provided data
- **predict**: Make class predictions on new data
- **predict_proba**: Get class probabilities for predictions
- **score_model**: Evaluate model accuracy on test data
- **get_model_info**: Get detailed information about a model
- **list_models**: List all available models
- **delete_model**: Remove a model from memory
- **save_model**: Serialize a model to base64 string
- **load_model**: Load a model from serialized string

## Installation

```bash
pip install -r requirements.txt
```

## Local Development

Run the server locally:

```bash
uvrun --with mcp server.py
```

The server will start on `http://localhost:8000` by default.

## Railway Deployment

### Prerequisites

1. A Railway account (sign up at https://railway.app)
2. Railway CLI installed (optional, can use web interface)
3. Git repository with your code (GitHub, GitLab, or Bitbucket)

### Deploy via Railway Web Interface

1. Go to https://railway.app and create a new project
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository containing this MCP server
4. Railway will automatically detect the Python project and use the `Procfile`
5. The server will be deployed and you'll get a public URL (e.g., `https://your-app.railway.app`)

### Deploy via Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login to Railway
railway login

# Initialize project (in your project directory)
railway init

# Link to existing project or create new one
railway link

# Deploy
railway up
```

### Environment Variables

No environment variables are required for basic operation. Railway automatically provides:
- `PORT`: The port your application should listen on
- The server automatically binds to `0.0.0.0` to accept external connections

### Verifying Deployment

Once deployed, your MCP server will be available at your Railway URL. You can test it by:

1. Visiting `https://your-app.railway.app` in a browser (should show MCP server info or 404, which is normal)
2. Using the MCP Inspector: `npx -y @modelcontextprotocol/inspector` and connecting to your Railway URL
3. Connecting from an MCP client using the streamable-http transport

**Current Deployment URL:** `https://web-production-a620a.up.railway.app`

## Usage

Once deployed, the MCP server will be accessible at your Railway URL. You can connect to it using any MCP-compatible client.

### Example: Using with Claude Desktop

Add to your Claude Desktop MCP configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

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

### Example API Calls

The server exposes tools that can be called via MCP protocol. Here's what each tool does:

**Create a classifier:**
```python
create_classifier(
    model_id="my_model",
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31
)
```

**Train the model:**
```python
train_model(
    model_id="my_model",
    X=[[1, 2], [3, 4], [5, 6]],
    y=[0, 1, 0]
)
```

**Make predictions:**
```python
predict(
    model_id="my_model",
    X=[[2, 3], [4, 5]]
)
```

**Get probabilities:**
```python
predict_proba(
    model_id="my_model",
    X=[[2, 3], [4, 5]]
)
```

## Model Storage

Currently, models are stored in-memory. This means:
- Models persist only during the server's lifetime
- Restarting the server will lose all models
- For production use, consider implementing persistent storage (database, file system, or cloud storage)

## API Reference

### HistGradientBoostingClassifier Parameters

All standard sklearn HistGradientBoostingClassifier parameters are supported:
- `loss`: Loss function (default: 'log_loss')
- `learning_rate`: Learning rate/shrinkage (default: 0.1)
- `max_iter`: Maximum boosting iterations (default: 100)
- `max_leaf_nodes`: Maximum leaves per tree (default: 31)
- `max_depth`: Maximum tree depth (default: None)
- `min_samples_leaf`: Minimum samples per leaf (default: 20)
- `l2_regularization`: L2 regularization (default: 0.0)
- `max_features`: Feature subsampling proportion (default: 1.0)
- `max_bins`: Maximum histogram bins (default: 255)
- `early_stopping`: Enable early stopping (default: 'auto')
- `validation_fraction`: Validation set fraction (default: 0.1)
- `n_iter_no_change`: Early stopping patience (default: 10)
- `random_state`: Random seed (default: None)
- `verbose`: Verbosity level (default: 0)

See the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) for detailed parameter descriptions.

## License

MIT
