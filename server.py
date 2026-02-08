"""
MCP Server for sklearn HistGradientBoostingClassifier

This server provides tools for training, predicting, and managing
HistGradientBoostingClassifier models via the Model Context Protocol.
"""

import json
import pickle
import base64
import os
from typing import Optional, List, Dict, Any, Union
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("HistGradientBoostingClassifier MCP Server", json_response=True)

# In-memory storage for models (in production, use a database or file system)
models: Dict[str, HistGradientBoostingClassifier] = {}


def serialize_model(model: HistGradientBoostingClassifier) -> str:
    """Serialize a model to base64 string."""
    return base64.b64encode(pickle.dumps(model)).decode('utf-8')


def deserialize_model(model_str: str) -> HistGradientBoostingClassifier:
    """Deserialize a model from base64 string."""
    return pickle.loads(base64.b64decode(model_str.encode('utf-8')))


@mcp.tool()
def create_classifier(
    model_id: str,
    loss: str = 'log_loss',
    learning_rate: float = 0.1,
    max_iter: int = 100,
    max_leaf_nodes: int = 31,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
    max_features: float = 1.0,
    max_bins: int = 255,
    early_stopping: Union[str, bool] = 'auto',
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 10,
    random_state: Optional[int] = None,
    verbose: int = 0
) -> str:
    """
    Create a new HistGradientBoostingClassifier instance.
    
    Args:
        model_id: Unique identifier for this model
        loss: Loss function ('log_loss')
        learning_rate: Learning rate (shrinkage)
        max_iter: Maximum number of boosting iterations
        max_leaf_nodes: Maximum number of leaves per tree
        max_depth: Maximum depth of trees (None = no limit)
        min_samples_leaf: Minimum samples per leaf
        l2_regularization: L2 regularization parameter
        max_features: Proportion of features to use for splits
        max_bins: Maximum number of bins for histogram
        early_stopping: Enable early stopping ('auto', True, False)
        validation_fraction: Fraction of data for validation
        n_iter_no_change: Iterations without improvement for early stopping
        random_state: Random seed for reproducibility
        verbose: Verbosity level (0, 1, or 2)
    
    Returns:
        Success message with model_id
    """
    classifier = HistGradientBoostingClassifier(
        loss=loss,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_features=max_features,
        max_bins=max_bins,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
        verbose=verbose
    )
    
    models[model_id] = classifier
    
    return json.dumps({
        "status": "success",
        "message": f"Classifier '{model_id}' created successfully",
        "model_id": model_id,
        "parameters": classifier.get_params()
    })


@mcp.tool()
def train_model(
    model_id: str,
    X: List[List[float]],
    y: List[Union[int, str]],
    sample_weight: Optional[List[float]] = None,
    X_val: Optional[List[List[float]]] = None,
    y_val: Optional[List[Union[int, str]]] = None
) -> str:
    """
    Train a HistGradientBoostingClassifier model.
    
    Args:
        model_id: Identifier of the model to train
        X: Training features (2D array-like)
        y: Training targets (1D array-like)
        sample_weight: Sample weights for training
        X_val: Validation features (optional, for early stopping)
        y_val: Validation targets (optional, for early stopping)
    
    Returns:
        Training results including score and iteration count
    """
    if model_id not in models:
        return json.dumps({
            "status": "error",
            "message": f"Model '{model_id}' not found. Create it first using create_classifier."
        })
    
    classifier = models[model_id]
    
    # Convert to numpy arrays
    X_array = np.array(X)
    y_array = np.array(y)
    
    # Prepare validation data if provided
    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs['X_val'] = np.array(X_val)
        fit_kwargs['y_val'] = np.array(y_val)
    
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = np.array(sample_weight)
    
    # Train the model
    classifier.fit(X_array, y_array, **fit_kwargs)
    
    # Calculate training score
    train_score = classifier.score(X_array, y_array)
    
    result = {
        "status": "success",
        "message": f"Model '{model_id}' trained successfully",
        "model_id": model_id,
        "n_iter": int(classifier.n_iter_),
        "n_trees_per_iteration": int(classifier.n_trees_per_iteration_),
        "train_score": float(train_score),
        "classes": classifier.classes_.tolist() if hasattr(classifier, 'classes_') else None
    }
    
    # Add validation score if available
    if hasattr(classifier, 'validation_score_') and len(classifier.validation_score_) > 0:
        result["validation_score"] = float(classifier.validation_score_[-1])
    
    return json.dumps(result)


@mcp.tool()
def predict(
    model_id: str,
    X: List[List[float]]
) -> str:
    """
    Predict classes for input samples.
    
    Args:
        model_id: Identifier of the trained model
        X: Input features (2D array-like)
    
    Returns:
        Predicted classes
    """
    if model_id not in models:
        return json.dumps({
            "status": "error",
            "message": f"Model '{model_id}' not found."
        })
    
    classifier = models[model_id]
    
    if not hasattr(classifier, 'classes_'):
        return json.dumps({
            "status": "error",
            "message": f"Model '{model_id}' has not been trained yet."
        })
    
    X_array = np.array(X)
    predictions = classifier.predict(X_array)
    
    return json.dumps({
        "status": "success",
        "model_id": model_id,
        "predictions": predictions.tolist(),
        "n_samples": len(predictions)
    })


@mcp.tool()
def predict_proba(
    model_id: str,
    X: List[List[float]]
) -> str:
    """
    Predict class probabilities for input samples.
    
    Args:
        model_id: Identifier of the trained model
        X: Input features (2D array-like)
    
    Returns:
        Class probabilities for each sample
    """
    if model_id not in models:
        return json.dumps({
            "status": "error",
            "message": f"Model '{model_id}' not found."
        })
    
    classifier = models[model_id]
    
    if not hasattr(classifier, 'classes_'):
        return json.dumps({
            "status": "error",
            "message": f"Model '{model_id}' has not been trained yet."
        })
    
    X_array = np.array(X)
    probabilities = classifier.predict_proba(X_array)
    
    return json.dumps({
        "status": "success",
        "model_id": model_id,
        "probabilities": probabilities.tolist(),
        "classes": classifier.classes_.tolist(),
        "n_samples": len(probabilities)
    })


@mcp.tool()
def score_model(
    model_id: str,
    X: List[List[float]],
    y: List[Union[int, str]],
    sample_weight: Optional[List[float]] = None
) -> str:
    """
    Score the model on test data (returns accuracy).
    
    Args:
        model_id: Identifier of the trained model
        X: Test features (2D array-like)
        y: True labels (1D array-like)
        sample_weight: Sample weights (optional)
    
    Returns:
        Accuracy score
    """
    if model_id not in models:
        return json.dumps({
            "status": "error",
            "message": f"Model '{model_id}' not found."
        })
    
    classifier = models[model_id]
    
    if not hasattr(classifier, 'classes_'):
        return json.dumps({
            "status": "error",
            "message": f"Model '{model_id}' has not been trained yet."
        })
    
    X_array = np.array(X)
    y_array = np.array(y)
    
    score = classifier.score(
        X_array, 
        y_array, 
        sample_weight=np.array(sample_weight) if sample_weight else None
    )
    
    return json.dumps({
        "status": "success",
        "model_id": model_id,
        "score": float(score),
        "metric": "accuracy"
    })


@mcp.tool()
def get_model_info(model_id: str) -> str:
    """
    Get information about a model including parameters and attributes.
    
    Args:
        model_id: Identifier of the model
    
    Returns:
        Model information including parameters and training attributes
    """
    if model_id not in models:
        return json.dumps({
            "status": "error",
            "message": f"Model '{model_id}' not found."
        })
    
    classifier = models[model_id]
    
    info = {
        "status": "success",
        "model_id": model_id,
        "parameters": classifier.get_params(),
        "is_trained": hasattr(classifier, 'classes_')
    }
    
    if hasattr(classifier, 'classes_'):
        info["classes"] = classifier.classes_.tolist()
        info["n_iter"] = int(classifier.n_iter_)
        info["n_trees_per_iteration"] = int(classifier.n_trees_per_iteration_)
        info["n_features_in"] = int(classifier.n_features_in_)
        
        if hasattr(classifier, 'train_score_') and len(classifier.train_score_) > 0:
            info["train_score"] = float(classifier.train_score_[-1])
        
        if hasattr(classifier, 'validation_score_') and len(classifier.validation_score_) > 0:
            info["validation_score"] = float(classifier.validation_score_[-1])
    
    return json.dumps(info)


@mcp.tool()
def list_models() -> str:
    """
    List all available models.
    
    Returns:
        List of model IDs and their training status
    """
    model_list = []
    for model_id, classifier in models.items():
        model_info = {
            "model_id": model_id,
            "is_trained": hasattr(classifier, 'classes_')
        }
        if hasattr(classifier, 'classes_'):
            model_info["n_classes"] = len(classifier.classes_)
            model_info["n_iter"] = int(classifier.n_iter_)
        model_list.append(model_info)
    
    return json.dumps({
        "status": "success",
        "models": model_list,
        "count": len(model_list)
    })


@mcp.tool()
def delete_model(model_id: str) -> str:
    """
    Delete a model from memory.
    
    Args:
        model_id: Identifier of the model to delete
    
    Returns:
        Success message
    """
    if model_id not in models:
        return json.dumps({
            "status": "error",
            "message": f"Model '{model_id}' not found."
        })
    
    del models[model_id]
    
    return json.dumps({
        "status": "success",
        "message": f"Model '{model_id}' deleted successfully"
    })


@mcp.tool()
def save_model(model_id: str) -> str:
    """
    Serialize and save a model to a base64 string.
    
    Args:
        model_id: Identifier of the model to save
    
    Returns:
        Base64-encoded serialized model
    """
    if model_id not in models:
        return json.dumps({
            "status": "error",
            "message": f"Model '{model_id}' not found."
        })
    
    classifier = models[model_id]
    model_str = serialize_model(classifier)
    
    return json.dumps({
        "status": "success",
        "model_id": model_id,
        "serialized_model": model_str
    })


@mcp.tool()
def load_model(model_id: str, serialized_model: str) -> str:
    """
    Load a model from a base64-encoded serialized string.
    
    Args:
        model_id: Identifier for the loaded model
        serialized_model: Base64-encoded serialized model
    
    Returns:
        Success message
    """
    try:
        classifier = deserialize_model(serialized_model)
        models[model_id] = classifier
        
        return json.dumps({
            "status": "success",
            "message": f"Model '{model_id}' loaded successfully",
            "is_trained": hasattr(classifier, 'classes_')
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to load model: {str(e)}"
        })


# Expose app for uvicorn (e.g., uvicorn server:app)
# Try to get the ASGI app from FastMCP for streamable HTTP
try:
    if hasattr(mcp, 'streamable_http_app'):
        app = mcp.streamable_http_app()
    else:
        # Fallback: Import the FastAPI app from app.py
        from app import app
except Exception:
    # Ultimate fallback: create a simple HTTP server
    from fastapi import FastAPI
    app = FastAPI(title="HistGradientBoostingClassifier MCP Server")
    
    @app.get("/")
    async def root():
        return {"status": "error", "message": "FastMCP initialization failed"}

if __name__ == "__main__":
    import uvicorn
    
    # Get port and host from environment (Railway provides PORT)
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Run the app with uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
