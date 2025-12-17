"""
Collection class for managing vectors with metadata.

A Collection is a container for vectors of the same dimension,
providing CRUD operations, search, and metadata filtering.
"""

from __future__ import annotations

import time
import uuid
import threading
from typing import (
    Dict, 
    List, 
    Optional, 
    Any, 
    Iterator, 
    Callable,
    Union,
    Tuple,
    Set,
)
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from .vector import VectorRecord, VectorBatch, validate_vector
from .exceptions import (
    VectorNotFoundError,
    VectorExistsError,
    DimensionMismatchError,
    ValidationError,
)
from ..distance import (
    get_metric_fn,
    query_distances,
    BatchDistanceCalculator,
    DistanceMetric,
)
from ..utils.validation import validate_id, validate_metadata
from ..utils.normalization import normalize_vector, normalize_batch


class IndexType(str, Enum):
    """Available index types."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"


@dataclass
class CollectionConfig:
    """Configuration for a Collection."""
    
    dimension: int
    metric: str = "euclidean"
    index_type: IndexType = IndexType.FLAT
    normalize_vectors: bool = False
    max_vectors: Optional[int] = None  # None = unlimited
    enable_metadata_index: bool = True
    
    def __post_init__(self):
        if self.dimension < 1:
            raise ValidationError(f"Dimension must be >= 1, got {self.dimension}")
        
        # Validate metric
        try:
            get_metric_fn(self.metric)
        except KeyError:
            raise ValidationError(f"Unknown metric: {self.metric}")


@dataclass
class CollectionStats:
    """Statistics about a collection."""
    
    name: str
    dimension: int
    metric: str
    vector_count: int
    index_type: str
    memory_usage_bytes: int
    created_at: float
    updated_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "metric": self.metric,
            "vector_count": self.vector_count,
            "index_type": self.index_type,
            "memory_usage_bytes": self.memory_usage_bytes,
            "memory_usage_mb": round(self.memory_usage_bytes / (1024 * 1024), 2),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class SearchResult:
    """Result from a similarity search."""
    
    id: str
    distance: float
    score: float  # Normalized score (higher = more similar)
    vector: Optional[NDArray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "distance": self.distance,
            "score": self.score,
            "metadata": self.metadata,
        }
        if include_vector and self.vector is not None:
            result["vector"] = self.vector.tolist()
        return result


class MetadataIndex:
    """
    Simple inverted index for metadata filtering.
    
    Maintains mappings from metadata values to vector IDs
    for efficient filtering.
    """
    
    def __init__(self):
        # {field_name: {value: set of ids}}
        self._index: Dict[str, Dict[Any, Set[str]]] = {}
        self._lock = threading.RLock()
    
    def add(self, id: str, metadata: Dict[str, Any]) -> None:
        """Add metadata for a vector ID."""
        with self._lock:
            for key, value in metadata.items():
                if key not in self._index:
                    self._index[key] = {}
                
                # Handle list values
                values = value if isinstance(value, list) else [value]
                
                for v in values:
                    # Convert to hashable type
                    v_key = self._make_hashable(v)
                    if v_key not in self._index[key]:
                        self._index[key][v_key] = set()
                    self._index[key][v_key].add(id)
    
    def remove(self, id: str, metadata: Dict[str, Any]) -> None:
        """Remove metadata for a vector ID."""
        with self._lock:
            for key, value in metadata.items():
                if key not in self._index:
                    continue
                
                values = value if isinstance(value, list) else [value]
                
                for v in values:
                    v_key = self._make_hashable(v)
                    if v_key in self._index[key]:
                        self._index[key][v_key].discard(id)
                        # Clean up empty sets
                        if not self._index[key][v_key]:
                            del self._index[key][v_key]
    
    def update(
        self, 
        id: str, 
        old_metadata: Dict[str, Any], 
        new_metadata: Dict[str, Any]
    ) -> None:
        """Update metadata for a vector ID."""
        self.remove(id, old_metadata)
        self.add(id, new_metadata)
    
    def query(self, filters: Dict[str, Any]) -> Set[str]:
        """
        Query for IDs matching all filters (AND logic).
        
        Args:
            filters: Dictionary of {field: value} or {field: [values]}
            
        Returns:
            Set of matching IDs
        """
        with self._lock:
            if not filters:
                return set()
            
            result_sets = []
            
            for key, value in filters.items():
                if key not in self._index:
                    return set()  # No matches for unknown field
                
                # Handle operators
                if isinstance(value, dict):
                    matching = self._query_with_operator(key, value)
                else:
                    # Simple equality
                    values = value if isinstance(value, list) else [value]
                    matching = set()
                    for v in values:
                        v_key = self._make_hashable(v)
                        if v_key in self._index[key]:
                            matching.update(self._index[key][v_key])
                
                result_sets.append(matching)
            
            # Intersection of all sets (AND logic)
            if not result_sets:
                return set()
            
            result = result_sets[0]
            for s in result_sets[1:]:
                result = result.intersection(s)
            
            return result
    
    def _query_with_operator(self, key: str, op_dict: Dict[str, Any]) -> Set[str]:
        """Handle query operators like $in, $gt, $lt, etc."""
        matching = set()
        
        for op, value in op_dict.items():
            if op == "$in":
                # Match any of the values
                for v in value:
                    v_key = self._make_hashable(v)
                    if v_key in self._index[key]:
                        matching.update(self._index[key][v_key])
            
            elif op == "$nin":
                # Match none of the values
                excluded = set()
                for v in value:
                    v_key = self._make_hashable(v)
                    if v_key in self._index[key]:
                        excluded.update(self._index[key][v_key])
                # Get all IDs for this field and exclude
                all_ids = set()
                for ids in self._index[key].values():
                    all_ids.update(ids)
                matching = all_ids - excluded
            
            elif op == "$exists":
                if value:
                    # Field exists
                    for ids in self._index[key].values():
                        matching.update(ids)
                # For $exists: false, would need to track all IDs
            
            # Numeric comparisons (simplified - works for comparable types)
            elif op in ("$gt", "$gte", "$lt", "$lte", "$ne"):
                for v_key, ids in self._index[key].items():
                    if self._compare(v_key, op, value):
                        matching.update(ids)
        
        return matching
    
    def _compare(self, a: Any, op: str, b: Any) -> bool:
        """Compare two values with an operator."""
        try:
            if op == "$gt":
                return a > b
            elif op == "$gte":
                return a >= b
            elif op == "$lt":
                return a < b
            elif op == "$lte":
                return a <= b
            elif op == "$ne":
                return a != b
        except TypeError:
            return False
        return False
    
    def _make_hashable(self, value: Any) -> Any:
        """Convert value to hashable type."""
        if isinstance(value, list):
            return tuple(value)
        elif isinstance(value, dict):
            return tuple(sorted(value.items()))
        return value
    
    def clear(self) -> None:
        """Clear the index."""
        with self._lock:
            self._index.clear()
    
    def get_indexed_fields(self) -> List[str]:
        """Get list of indexed field names."""
        return list(self._index.keys())


class Collection:
    """
    A collection of vectors with the same dimension.
    
    Provides CRUD operations, similarity search, and metadata filtering.
    
    Example:
        >>> collection = Collection("my_vectors", dimension=128, metric="cosine")
        >>> 
        >>> # Insert vectors
        >>> collection.add("id1", vector1, {"category": "A"})
        >>> collection.add("id2", vector2, {"category": "B"})
        >>> 
        >>> # Search
        >>> results = collection.search(query_vector, k=10)
        >>> 
        >>> # Search with filter
        >>> results = collection.search(
        ...     query_vector, 
        ...     k=10, 
        ...     filter={"category": "A"}
        ... )
    """
    
    def __init__(
        self,
        name: str,
        dimension: int,
        metric: str = "euclidean",
        normalize: bool = False,
        max_vectors: Optional[int] = None,
        enable_metadata_index: bool = True,
    ):
        """
        Initialize a new collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            metric: Distance metric to use
            normalize: Whether to normalize vectors on insert
            max_vectors: Maximum number of vectors (None = unlimited)
            enable_metadata_index: Whether to index metadata for filtering
        """
        self.name = validate_id(name)
        self.config = CollectionConfig(
            dimension=dimension,
            metric=metric,
            normalize_vectors=normalize,
            max_vectors=max_vectors,
            enable_metadata_index=enable_metadata_index,
        )
        
        # Storage
        self._vectors: Dict[str, VectorRecord] = {}
        self._vector_matrix: Optional[NDArray] = None  # Cached matrix
        self._id_to_index: Dict[str, int] = {}  # ID -> matrix row index
        self._index_to_id: Dict[int, str] = {}  # Matrix row index -> ID
        self._matrix_dirty = True
        
        # Metadata index
        self._metadata_index = MetadataIndex() if enable_metadata_index else None
        
        # Distance calculator
        self._distance_calc = BatchDistanceCalculator(metric=metric)
        
        # Thread safety
        self._lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.RLock()
        
        # Timestamps
        self._created_at = time.time()
        self._updated_at = time.time()
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def dimension(self) -> int:
        """Vector dimension."""
        return self.config.dimension
    
    @property
    def metric(self) -> str:
        """Distance metric."""
        return self.config.metric
    
    def __len__(self) -> int:
        """Number of vectors in collection."""
        return len(self._vectors)
    
    def __contains__(self, id: str) -> bool:
        """Check if vector ID exists."""
        return id in self._vectors
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    def add(
        self,
        id: str,
        vector: Union[NDArray, List[float]],
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Add a vector to the collection.
        
        Args:
            id: Unique identifier
            vector: The embedding vector
            metadata: Optional metadata dictionary
            overwrite: If True, overwrite existing vector with same ID
            
        Returns:
            The vector ID
            
        Raises:
            VectorExistsError: If ID exists and overwrite=False
            DimensionMismatchError: If vector dimension doesn't match
        """
        # Validate inputs
        id = validate_id(id) if id else str(uuid.uuid4())
        metadata = validate_metadata(metadata)
        vector = validate_vector(vector, expected_dimension=self.dimension)
        
        # Normalize if configured
        if self.config.normalize_vectors:
            vector = normalize_vector(vector)
        
        with self._lock:
            # Check if exists
            if id in self._vectors:
                if not overwrite:
                    raise VectorExistsError(f"Vector with ID '{id}' already exists")
                # Update instead
                return self.update(id, vector=vector, metadata=metadata)
            
            # Check max vectors
            if self.config.max_vectors and len(self._vectors) >= self.config.max_vectors:
                raise ValidationError(
                    f"Collection at maximum capacity ({self.config.max_vectors})"
                )
            
            # Create record
            record = VectorRecord(id=id, vector=vector, metadata=metadata)
            self._vectors[id] = record
            
            # Update metadata index
            if self._metadata_index and metadata:
                self._metadata_index.add(id, metadata)
            
            # Mark matrix as dirty
            self._matrix_dirty = True
            self._updated_at = time.time()
            
            return id
    
    def add_batch(
        self,
        vectors: Union[List[Dict[str, Any]], VectorBatch],
        overwrite: bool = False,
        on_error: str = "raise",  # "raise", "skip", "continue"
    ) -> Dict[str, Any]:
        """
        Add multiple vectors at once.
        
        Args:
            vectors: List of dicts with 'id', 'vector', 'metadata' keys,
                    or a VectorBatch object
            overwrite: If True, overwrite existing vectors
            on_error: Error handling: "raise", "skip", "continue"
            
        Returns:
            Dictionary with 'success_count', 'error_count', 'errors'
        """
        results = {
            "success_count": 0,
            "error_count": 0,
            "errors": [],
        }
        
        # Convert VectorBatch to list
        if isinstance(vectors, VectorBatch):
            vectors = [
                {"id": r.id, "vector": r.vector, "metadata": r.metadata}
                for r in vectors
            ]
        
        for item in vectors:
            try:
                self.add(
                    id=item.get("id", str(uuid.uuid4())),
                    vector=item["vector"],
                    metadata=item.get("metadata", {}),
                    overwrite=overwrite,
                )
                results["success_count"] += 1
            except Exception as e:
                results["error_count"] += 1
                error_info = {"id": item.get("id"), "error": str(e)}
                results["errors"].append(error_info)
                
                if on_error == "raise":
                    raise
                elif on_error == "skip":
                    continue
        
        return results
    
    def get(
        self, 
        id: str, 
        include_vector: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get a vector by ID.
        
        Args:
            id: Vector ID
            include_vector: Whether to include the vector array
            
        Returns:
            Dictionary with vector info, or None if not found
        """
        record = self._vectors.get(id)
        if record is None:
            return None
        
        result = {
            "id": record.id,
            "metadata": record.metadata.copy(),
            "timestamp": record.timestamp,
        }
        
        if include_vector:
            result["vector"] = record.vector.copy()
        
        return result
    
    def get_many(
        self, 
        ids: List[str], 
        include_vector: bool = True
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Get multiple vectors by ID.
        
        Args:
            ids: List of vector IDs
            include_vector: Whether to include vector arrays
            
        Returns:
            List of results (None for missing IDs)
        """
        return [self.get(id, include_vector) for id in ids]
    
    def update(
        self,
        id: str,
        vector: Optional[Union[NDArray, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_update_mode: str = "replace",  # "replace", "merge"
    ) -> str:
        """
        Update an existing vector.
        
        Args:
            id: Vector ID to update
            vector: New vector (optional)
            metadata: New metadata (optional)
            metadata_update_mode: "replace" or "merge"
            
        Returns:
            The vector ID
            
        Raises:
            VectorNotFoundError: If vector doesn't exist
        """
        with self._lock:
            if id not in self._vectors:
                raise VectorNotFoundError(f"Vector with ID '{id}' not found")
            
            record = self._vectors[id]
            old_metadata = record.metadata.copy()
            
            # Update vector
            if vector is not None:
                vector = validate_vector(vector, expected_dimension=self.dimension)
                if self.config.normalize_vectors:
                    vector = normalize_vector(vector)
                record.vector = vector
                self._matrix_dirty = True
            
            # Update metadata
            if metadata is not None:
                metadata = validate_metadata(metadata)
                
                if metadata_update_mode == "merge":
                    new_metadata = {**record.metadata, **metadata}
                else:
                    new_metadata = metadata
                
                record.metadata = new_metadata
                
                # Update metadata index
                if self._metadata_index:
                    self._metadata_index.update(id, old_metadata, new_metadata)
            
            # Update timestamp
            record.timestamp = time.time()
            self._updated_at = time.time()
            
            return id
    
    def delete(self, id: str) -> bool:
        """
        Delete a vector by ID.
        
        Args:
            id: Vector ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if id not in self._vectors:
                return False
            
            record = self._vectors[id]
            
            # Remove from metadata index
            if self._metadata_index and record.metadata:
                self._metadata_index.remove(id, record.metadata)
            
            # Remove from storage
            del self._vectors[id]
            
            # Mark matrix as dirty
            self._matrix_dirty = True
            self._updated_at = time.time()
            
            return True
    
    def delete_many(self, ids: List[str]) -> Dict[str, int]:
        """
        Delete multiple vectors.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            Dictionary with 'deleted' and 'not_found' counts
        """
        deleted = 0
        not_found = 0
        
        for id in ids:
            if self.delete(id):
                deleted += 1
            else:
                not_found += 1
        
        return {"deleted": deleted, "not_found": not_found}
    
    def clear(self) -> int:
        """
        Remove all vectors from the collection.
        
        Returns:
            Number of vectors removed
        """
        with self._lock:
            count = len(self._vectors)
            self._vectors.clear()
            self._vector_matrix = None
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._matrix_dirty = True
            
            if self._metadata_index:
                self._metadata_index.clear()
            
            self._updated_at = time.time()
            return count
    
    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================
    
    def search(
        self,
        query: Union[NDArray, List[float]],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
        include_metadata: bool = True,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector
            k: Number of results to return
            filter: Metadata filter (optional)
            include_vector: Include vectors in results
            include_metadata: Include metadata in results
            
        Returns:
            List of SearchResult objects, sorted by distance
        """
        if len(self._vectors) == 0:
            return []
        
        # Validate and prepare query
        query = validate_vector(query, expected_dimension=self.dimension)
        if self.config.normalize_vectors:
            query = normalize_vector(query)
        
        # Get candidate IDs (filter if needed)
        if filter and self._metadata_index:
            candidate_ids = self._metadata_index.query(filter)
            if not candidate_ids:
                return []
        else:
            candidate_ids = None  # Search all
        
        # Build/update vector matrix
        self._ensure_matrix()
        
        # Compute distances
        if candidate_ids is not None:
            # Filter to candidate indices
            indices = [self._id_to_index[id] for id in candidate_ids if id in self._id_to_index]
            if not indices:
                return []
            
            candidate_vectors = self._vector_matrix[indices]
            distances = query_distances(query, candidate_vectors, self.metric)
            
            # Map back to IDs
            id_distances = list(zip(
                [self._index_to_id[i] for i in indices],
                distances
            ))
        else:
            # Search all vectors
            distances = query_distances(query, self._vector_matrix, self.metric)
            id_distances = list(zip(
                [self._index_to_id[i] for i in range(len(distances))],
                distances
            ))
        
        # Sort by distance and take top k
        id_distances.sort(key=lambda x: x[1])
        top_k = id_distances[:k]
        
        # Build results
        results = []
        max_distance = top_k[-1][1] if top_k else 1.0
        
        for id, distance in top_k:
            record = self._vectors[id]
            
            # Compute score (normalized, higher = more similar)
            if max_distance > 0:
                score = 1.0 - (distance / (max_distance + 1e-8))
            else:
                score = 1.0
            
            result = SearchResult(
                id=id,
                distance=float(distance),
                score=float(score),
                vector=record.vector.copy() if include_vector else None,
                metadata=record.metadata.copy() if include_metadata else {},
            )
            results.append(result)
        
        return results
    
    def search_batch(
        self,
        queries: Union[NDArray, List[List[float]]],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
    ) -> List[List[SearchResult]]:
        """
        Search with multiple queries.
        
        Args:
            queries: Array of query vectors (n, dimension)
            k: Number of results per query
            filter: Metadata filter (applied to all queries)
            include_vector: Include vectors in results
            
        Returns:
            List of result lists, one per query
        """
        queries = np.atleast_2d(queries)
        return [
            self.search(q, k=k, filter=filter, include_vector=include_vector)
            for q in queries
        ]
    
    def _ensure_matrix(self) -> None:
        """Ensure vector matrix is up to date."""
        if not self._matrix_dirty and self._vector_matrix is not None:
            return
        
        if len(self._vectors) == 0:
            self._vector_matrix = np.zeros((0, self.dimension), dtype=np.float32)
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._matrix_dirty = False
            return
        
        # Build matrix and mappings
        ids = list(self._vectors.keys())
        vectors = [self._vectors[id].vector for id in ids]
        
        self._vector_matrix = np.array(vectors, dtype=np.float32)
        self._id_to_index = {id: i for i, id in enumerate(ids)}
        self._index_to_id = {i: id for i, id in enumerate(ids)}
        self._matrix_dirty = False
    
    # =========================================================================
    # QUERY BY METADATA
    # =========================================================================
    
    def query_by_metadata(
        self,
        filter: Dict[str, Any],
        limit: Optional[int] = None,
        include_vector: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Query vectors by metadata without similarity search.
        
        Args:
            filter: Metadata filter
            limit: Maximum number of results
            include_vector: Include vectors in results
            
        Returns:
            List of matching vectors
        """
        if not self._metadata_index:
            # Fallback to linear scan
            return self._query_metadata_linear(filter, limit, include_vector)
        
        matching_ids = self._metadata_index.query(filter)
        
        results = []
        for id in matching_ids:
            if limit and len(results) >= limit:
                break
            
            result = self.get(id, include_vector=include_vector)
            if result:
                results.append(result)
        
        return results
    
    def _query_metadata_linear(
        self,
        filter: Dict[str, Any],
        limit: Optional[int],
        include_vector: bool,
    ) -> List[Dict[str, Any]]:
        """Linear scan for metadata queries (fallback)."""
        results = []
        
        for id, record in self._vectors.items():
            if limit and len(results) >= limit:
                break
            
            if self._matches_filter(record.metadata, filter):
                results.append(self.get(id, include_vector=include_vector))
        
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                # Handle operators
                for op, op_value in value.items():
                    if op == "$in" and metadata[key] not in op_value:
                        return False
                    elif op == "$nin" and metadata[key] in op_value:
                        return False
                    elif op == "$gt" and not metadata[key] > op_value:
                        return False
                    elif op == "$gte" and not metadata[key] >= op_value:
                        return False
                    elif op == "$lt" and not metadata[key] < op_value:
                        return False
                    elif op == "$lte" and not metadata[key] <= op_value:
                        return False
                    elif op == "$ne" and metadata[key] == op_value:
                        return False
            else:
                # Simple equality
                if metadata[key] != value:
                    return False
        
        return True
    
    # =========================================================================
    # ITERATION
    # =========================================================================
    
    def __iter__(self) -> Iterator[VectorRecord]:
        """Iterate over all vector records."""
        return iter(self._vectors.values())
    
    def iter_ids(self) -> Iterator[str]:
        """Iterate over all vector IDs."""
        return iter(self._vectors.keys())
    
    def list_ids(self, limit: Optional[int] = None, offset: int = 0) -> List[str]:
        """
        List vector IDs with pagination.
        
        Args:
            limit: Maximum number of IDs to return
            offset: Number of IDs to skip
            
        Returns:
            List of vector IDs
        """
        ids = list(self._vectors.keys())
        
        if offset:
            ids = ids[offset:]
        
        if limit:
            ids = ids[:limit]
        
        return ids
    
    # =========================================================================
    # STATISTICS & INFO
    # =========================================================================
    
    def stats(self) -> CollectionStats:
        """Get collection statistics."""
        # Estimate memory usage
        memory = 0
        if self._vectors:
            # Vector data
            memory += len(self._vectors) * self.dimension * 4  # float32
            # Metadata (rough estimate)
            memory += sum(
                len(str(r.metadata)) for r in self._vectors.values()
            )
            # IDs
            memory += sum(len(id) for id in self._vectors.keys())
        
        return CollectionStats(
            name=self.name,
            dimension=self.dimension,
            metric=self.metric,
            vector_count=len(self._vectors),
            index_type=self.config.index_type.value,
            memory_usage_bytes=memory,
            created_at=self._created_at,
            updated_at=self._updated_at,
        )
    
    def describe(self) -> Dict[str, Any]:
        """Get collection description."""
        stats = self.stats()
        return {
            **stats.to_dict(),
            "config": {
                "normalize_vectors": self.config.normalize_vectors,
                "max_vectors": self.config.max_vectors,
                "enable_metadata_index": self.config.enable_metadata_index,
            },
            "indexed_fields": (
                self._metadata_index.get_indexed_fields() 
                if self._metadata_index else []
            ),
        }
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize collection to dictionary.
        
        Returns:
            Dictionary representation of collection
        """
        return {
            "name": self.name,
            "config": {
                "dimension": self.config.dimension,
                "metric": self.config.metric,
                "index_type": self.config.index_type.value,
                "normalize_vectors": self.config.normalize_vectors,
                "max_vectors": self.config.max_vectors,
                "enable_metadata_index": self.config.enable_metadata_index,
            },
            "vectors": [
                record.to_dict() for record in self._vectors.values()
            ],
            "created_at": self._created_at,
            "updated_at": self._updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Collection":
        """
        Deserialize collection from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Collection instance
        """
        config = data["config"]
        
        collection = cls(
            name=data["name"],
            dimension=config["dimension"],
            metric=config["metric"],
            normalize=config.get("normalize_vectors", False),
            max_vectors=config.get("max_vectors"),
            enable_metadata_index=config.get("enable_metadata_index", True),
        )
        
        # Add vectors
        for vec_data in data.get("vectors", []):
            record = VectorRecord.from_dict(vec_data)
            collection._vectors[record.id] = record
            if collection._metadata_index and record.metadata:
                collection._metadata_index.add(record.id, record.metadata)
        
        collection._matrix_dirty = True
        collection._created_at = data.get("created_at", time.time())
        collection._updated_at = data.get("updated_at", time.time())
        
        return collection
    
    def __repr__(self) -> str:
        return (
            f"Collection(name='{self.name}', "
            f"dimension={self.dimension}, "
            f"count={len(self)}, "
            f"metric='{self.metric}')"
        )


# =============================================================================
# THREADING HELPERS
# =============================================================================

class RWLock:
    """
    Simple Read-Write Lock implementation.
    
    Allows multiple readers or a single writer.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._read_ready = threading.Condition(self._lock)
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False
    
    def acquire_read(self):
        """Acquire read lock."""
        with self._lock:
            while self._writer_active or self._writers_waiting > 0:
                self._read_ready.wait()
            self._readers += 1
    
    def release_read(self):
        """Release read lock."""
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
    
    def acquire_write(self):
        """Acquire write lock."""
        with self._lock:
            self._writers_waiting += 1
            while self._readers > 0 or self._writer_active:
                self._read_ready.wait()
            self._writers_waiting -= 1
            self._writer_active = True
    
    def release_write(self):
        """Release write lock."""
        with self._lock:
            self._writer_active = False
            self._read_ready.notify_all()
    
    def __enter__(self):
        """Default to write lock for context manager."""
        self.acquire_write()
        return self
    
    def __exit__(self, *args):
        self.release_write()


# Add RWLock to threading module if not available
if not hasattr(threading, 'RWLock'):
    threading.RWLock = RWLock