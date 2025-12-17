"""
Input validation utilities.
"""

from typing import Any, Dict, Optional, Set
import re


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


# Valid ID pattern: alphanumeric, underscores, hyphens, dots
ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')

# Reserved metadata keys
RESERVED_KEYS: Set[str] = {"_id", "_vector", "_score", "_distance", "_timestamp"}

# Maximum limits
MAX_ID_LENGTH = 256
MAX_METADATA_KEY_LENGTH = 64
MAX_METADATA_VALUE_LENGTH = 65536
MAX_METADATA_KEYS = 100


def validate_id(id: str, allow_empty: bool = False) -> str:
    """
    Validate a vector ID.
    
    Args:
        id: The ID to validate
        allow_empty: Whether to allow empty IDs
        
    Returns:
        The validated ID
        
    Raises:
        ValidationError: If ID is invalid
    """
    if not isinstance(id, str):
        raise ValidationError(f"ID must be a string, got {type(id).__name__}")
    
    if not id:
        if allow_empty:
            return id
        raise ValidationError("ID cannot be empty")
    
    if len(id) > MAX_ID_LENGTH:
        raise ValidationError(
            f"ID too long: {len(id)} characters (max {MAX_ID_LENGTH})"
        )
    
    if not ID_PATTERN.match(id):
        raise ValidationError(
            f"Invalid ID '{id}': must contain only alphanumeric characters, "
            "underscores, hyphens, or dots"
        )
    
    return id


def validate_metadata(
    metadata: Optional[Dict[str, Any]],
    allow_none: bool = True,
) -> Dict[str, Any]:
    """
    Validate metadata dictionary.
    
    Args:
        metadata: The metadata to validate
        allow_none: Whether to allow None
        
    Returns:
        The validated metadata (empty dict if None and allowed)
        
    Raises:
        ValidationError: If metadata is invalid
    """
    if metadata is None:
        if allow_none:
            return {}
        raise ValidationError("Metadata cannot be None")
    
    if not isinstance(metadata, dict):
        raise ValidationError(
            f"Metadata must be a dictionary, got {type(metadata).__name__}"
        )
    
    if len(metadata) > MAX_METADATA_KEYS:
        raise ValidationError(
            f"Too many metadata keys: {len(metadata)} (max {MAX_METADATA_KEYS})"
        )
    
    validated = {}
    for key, value in metadata.items():
        # Validate key
        if not isinstance(key, str):
            raise ValidationError(
                f"Metadata key must be string, got {type(key).__name__}"
            )
        
        if len(key) > MAX_METADATA_KEY_LENGTH:
            raise ValidationError(
                f"Metadata key too long: '{key[:20]}...' "
                f"({len(key)} chars, max {MAX_METADATA_KEY_LENGTH})"
            )
        
        if key in RESERVED_KEYS:
            raise ValidationError(f"Reserved metadata key: '{key}'")
        
        # Validate value
        validated_value = _validate_metadata_value(key, value)
        validated[key] = validated_value
    
    return validated


def _validate_metadata_value(key: str, value: Any) -> Any:
    """Validate a single metadata value."""
    # Allow None
    if value is None:
        return None
    
    # Allow basic types
    if isinstance(value, (bool, int, float)):
        return value
    
    # Validate strings
    if isinstance(value, str):
        if len(value) > MAX_METADATA_VALUE_LENGTH:
            raise ValidationError(
                f"Metadata value for '{key}' too long: "
                f"{len(value)} chars (max {MAX_METADATA_VALUE_LENGTH})"
            )
        return value
    
    # Validate lists
    if isinstance(value, list):
        return [_validate_metadata_value(key, v) for v in value]
    
    # Validate nested dicts (one level)
    if isinstance(value, dict):
        return {
            k: _validate_metadata_value(f"{key}.{k}", v)
            for k, v in value.items()
        }
    
    raise ValidationError(
        f"Invalid metadata value type for '{key}': {type(value).__name__}. "
        "Allowed types: str, int, float, bool, None, list, dict"
    )


def validate_dimension(dimension: int, min_dim: int = 1, max_dim: int = 65536) -> int:
    """
    Validate vector dimension.
    
    Args:
        dimension: The dimension to validate
        min_dim: Minimum allowed dimension
        max_dim: Maximum allowed dimension
        
    Returns:
        The validated dimension
        
    Raises:
        ValidationError: If dimension is invalid
    """
    if not isinstance(dimension, int):
        raise ValidationError(
            f"Dimension must be an integer, got {type(dimension).__name__}"
        )
    
    if dimension < min_dim:
        raise ValidationError(
            f"Dimension too small: {dimension} (min {min_dim})"
        )
    
    if dimension > max_dim:
        raise ValidationError(
            f"Dimension too large: {dimension} (max {max_dim})"
        )
    
    return dimension


def validate_k(k: int, max_k: int = 10000) -> int:
    """
    Validate k (number of results).
    
    Args:
        k: Number of results to return
        max_k: Maximum allowed k
        
    Returns:
        The validated k
        
    Raises:
        ValidationError: If k is invalid
    """
    if not isinstance(k, int):
        raise ValidationError(f"k must be an integer, got {type(k).__name__}")
    
    if k < 1:
        raise ValidationError(f"k must be at least 1, got {k}")
    
    if k > max_k:
        raise ValidationError(f"k too large: {k} (max {max_k})")
    
    return k