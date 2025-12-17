"""
Custom exceptions for VectorDB.
"""


class VectorDBError(Exception):
    """Base exception for VectorDB."""
    pass


class CollectionError(VectorDBError):
    """Error related to collection operations."""
    pass


class CollectionNotFoundError(CollectionError):
    """Collection does not exist."""
    pass


class CollectionExistsError(CollectionError):
    """Collection already exists."""
    pass


class VectorError(VectorDBError):
    """Error related to vector operations."""
    pass


class VectorNotFoundError(VectorError):
    """Vector with given ID not found."""
    pass


class VectorExistsError(VectorError):
    """Vector with given ID already exists."""
    pass


class DimensionMismatchError(VectorError):
    """Vector dimension doesn't match collection dimension."""
    pass


class ValidationError(VectorDBError):
    """Input validation error."""
    pass


class IndexError(VectorDBError):
    """Error related to index operations."""
    pass


class IndexNotTrainedError(IndexError):
    """Index requires training before use."""
    pass


class StorageError(VectorDBError):
    """Error related to storage operations."""
    pass


class SerializationError(StorageError):
    """Error during serialization/deserialization."""
    pass