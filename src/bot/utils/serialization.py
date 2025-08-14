"""
Secure serialization utilities to replace pickle usage.

This module provides safe alternatives to pickle for different data types:
- ML models: joblib for scikit-learn, torch.save/load for PyTorch
- DataFrames: parquet format
- Arrays: numpy native format
- Simple data: JSON
- Complex objects: custom serialization methods

Security Note: This module replaces pickle usage to eliminate arbitrary code
execution vulnerabilities (CVE-2022-48560).
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


class SerializationError(Exception):
    """Custom exception for serialization errors."""

    pass


def save_model(model: Any, filepath: str | Path, model_type: str = "auto") -> None:
    """
    Save ML model using appropriate secure serialization method.

    Args:
        model: The model to save
        filepath: Path to save the model
        model_type: Type of model ("sklearn", "torch", "tensorflow", "auto")

    Raises:
        SerializationError: If model type is unsupported or saving fails
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "auto":
        model_type = _detect_model_type(model)

    try:
        if model_type == "sklearn":
            joblib.dump(model, filepath)
            logger.info(f"Saved sklearn model to {filepath}")

        elif model_type == "torch":
            if not TORCH_AVAILABLE:
                raise SerializationError("PyTorch not available")
            torch.save(model.state_dict(), filepath)
            logger.info(f"Saved PyTorch model to {filepath}")

        elif model_type == "tensorflow":
            if not TF_AVAILABLE:
                raise SerializationError("TensorFlow not available")
            model.save_weights(str(filepath))
            logger.info(f"Saved TensorFlow model to {filepath}")

        else:
            raise SerializationError(f"Unsupported model type: {model_type}")

    except Exception as e:
        raise SerializationError(f"Failed to save model: {e}") from e


def load_model(
    filepath: str | Path, model_class: Any | None = None, model_type: str = "auto"
) -> Any:
    """
    Load ML model using appropriate secure deserialization method.

    Args:
        filepath: Path to the saved model
        model_class: Model class for PyTorch/TensorFlow models
        model_type: Type of model ("sklearn", "torch", "tensorflow", "auto")

    Returns:
        The loaded model

    Raises:
        SerializationError: If loading fails or model type is unsupported
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise SerializationError(f"Model file not found: {filepath}")

    if model_type == "auto":
        model_type = _detect_model_type_from_file(filepath)

    try:
        if model_type == "sklearn":
            model = joblib.load(filepath)
            logger.info(f"Loaded sklearn model from {filepath}")
            return model

        elif model_type == "torch":
            if not TORCH_AVAILABLE:
                raise SerializationError("PyTorch not available")
            if model_class is None:
                raise SerializationError("model_class required for PyTorch models")
            model = model_class()
            model.load_state_dict(torch.load(filepath))
            logger.info(f"Loaded PyTorch model from {filepath}")
            return model

        elif model_type == "tensorflow":
            if not TF_AVAILABLE:
                raise SerializationError("TensorFlow not available")
            if model_class is None:
                raise SerializationError("model_class required for TensorFlow models")
            model = model_class()
            model.load_weights(str(filepath))
            logger.info(f"Loaded TensorFlow model from {filepath}")
            return model

        else:
            raise SerializationError(f"Unsupported model type: {model_type}")

    except Exception as e:
        raise SerializationError(f"Failed to load model: {e}") from e


def save_dataframe(df: pd.DataFrame, filepath: str | Path, format: str = "parquet") -> None:
    """
    Save DataFrame using secure format.

    Args:
        df: DataFrame to save
        filepath: Path to save the DataFrame
        format: Format to use ("parquet", "feather", "csv")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "parquet":
            df.to_parquet(filepath)
        elif format == "feather":
            df.to_feather(filepath)
        elif format == "csv":
            df.to_csv(filepath, index=False)
        else:
            raise SerializationError(f"Unsupported DataFrame format: {format}")

        logger.info(f"Saved DataFrame to {filepath} in {format} format")

    except Exception as e:
        raise SerializationError(f"Failed to save DataFrame: {e}") from e


def load_dataframe(filepath: str | Path, format: str = "auto") -> pd.DataFrame:
    """
    Load DataFrame from secure format.

    Args:
        filepath: Path to the DataFrame file
        format: Format to read ("parquet", "feather", "csv", "auto")

    Returns:
        The loaded DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise SerializationError(f"DataFrame file not found: {filepath}")

    if format == "auto":
        format = _detect_dataframe_format(filepath)

    try:
        if format == "parquet":
            df = pd.read_parquet(filepath)
        elif format == "feather":
            df = pd.read_feather(filepath)
        elif format == "csv":
            df = pd.read_csv(filepath)
        else:
            raise SerializationError(f"Unsupported DataFrame format: {format}")

        logger.info(f"Loaded DataFrame from {filepath} in {format} format")
        return df

    except Exception as e:
        raise SerializationError(f"Failed to load DataFrame: {e}") from e


def save_array(arr: np.ndarray, filepath: str | Path) -> None:
    """
    Save numpy array using secure numpy format.

    Args:
        arr: Array to save
        filepath: Path to save the array
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        np.save(filepath, arr)
        logger.info(f"Saved numpy array to {filepath}")
    except Exception as e:
        raise SerializationError(f"Failed to save array: {e}") from e


def load_array(filepath: str | Path) -> np.ndarray:
    """
    Load numpy array from secure numpy format.

    Args:
        filepath: Path to the array file

    Returns:
        The loaded array
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise SerializationError(f"Array file not found: {filepath}")

    try:
        arr = np.load(filepath)
        logger.info(f"Loaded numpy array from {filepath}")
        return arr
    except Exception as e:
        raise SerializationError(f"Failed to load array: {e}") from e


def save_json(data: dict | list, filepath: str | Path, indent: int = 2) -> None:
    """
    Save data as JSON.

    Args:
        data: Data to save (must be JSON serializable)
        filepath: Path to save the JSON file
        indent: JSON indentation for readability
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent, default=_json_serializer)
        logger.info(f"Saved JSON data to {filepath}")
    except Exception as e:
        raise SerializationError(f"Failed to save JSON: {e}") from e


def load_json(filepath: str | Path) -> dict | list:
    """
    Load data from JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        The loaded data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise SerializationError(f"JSON file not found: {filepath}")

    try:
        with open(filepath) as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {filepath}")
        return data
    except Exception as e:
        raise SerializationError(f"Failed to load JSON: {e}") from e


def _detect_model_type(model: Any) -> str:
    """Detect the type of ML model."""
    model_class_name = model.__class__.__name__
    module_name = model.__class__.__module__

    if "sklearn" in module_name:
        return "sklearn"
    elif "torch" in module_name and hasattr(model, "state_dict"):
        return "torch"
    elif "tensorflow" in module_name or "keras" in module_name:
        return "tensorflow"
    else:
        raise SerializationError(f"Unknown model type: {model_class_name} from {module_name}")


def _detect_model_type_from_file(filepath: Path) -> str:
    """Detect model type from file extension or content."""
    suffix = filepath.suffix.lower()

    if suffix in [".pkl", ".joblib"]:
        return "sklearn"
    elif suffix in [".pt", ".pth"]:
        return "torch"
    elif suffix in [".h5", ".tf"]:
        return "tensorflow"
    else:
        # Try to detect from parent directory or filename
        if "sklearn" in str(filepath):
            return "sklearn"
        elif "torch" in str(filepath) or "pytorch" in str(filepath):
            return "torch"
        elif "tensorflow" in str(filepath) or "keras" in str(filepath):
            return "tensorflow"
        else:
            return "sklearn"  # Default fallback


def _detect_dataframe_format(filepath: Path) -> str:
    """Detect DataFrame format from file extension."""
    suffix = filepath.suffix.lower()

    if suffix == ".parquet":
        return "parquet"
    elif suffix == ".feather":
        return "feather"
    elif suffix == ".csv":
        return "csv"
    else:
        return "parquet"  # Default to parquet


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for special types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "isoformat"):  # datetime objects
        return obj.isoformat()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def migrate_pickle_file(
    old_filepath: str | Path, new_filepath: str | Path, data_type: str = "auto"
) -> bool:
    """
    Migrate an existing pickle file to a secure format.

    Args:
        old_filepath: Path to the pickle file
        new_filepath: Path for the new secure file
        data_type: Type of data ("model", "dataframe", "array", "json", "auto")

    Returns:
        True if migration successful, False otherwise
    """
    old_filepath = Path(old_filepath)
    new_filepath = Path(new_filepath)

    if not old_filepath.exists():
        logger.warning(f"Pickle file not found: {old_filepath}")
        return False

    try:
        # This is the ONLY place we use pickle - for migration purposes
        import pickle

        with open(old_filepath, "rb") as f:
            data = pickle.load(f)

        if data_type == "auto":
            data_type = _detect_data_type(data)

        # Save using secure method
        if data_type == "model":
            save_model(data, new_filepath)
        elif data_type == "dataframe":
            save_dataframe(data, new_filepath)
        elif data_type == "array":
            save_array(data, new_filepath)
        elif data_type == "json":
            save_json(data, new_filepath)
        else:
            logger.error(f"Unknown data type: {data_type}")
            return False

        logger.info(f"Successfully migrated {old_filepath} to {new_filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to migrate {old_filepath}: {e}")
        return False


def _detect_data_type(data: Any) -> str:
    """Detect the type of data for migration."""
    if hasattr(data, "fit") and hasattr(data, "predict"):
        return "model"
    elif isinstance(data, pd.DataFrame):
        return "dataframe"
    elif isinstance(data, np.ndarray):
        return "array"
    elif isinstance(data, (dict, list)):
        return "json"
    else:
        return "json"  # Default fallback
