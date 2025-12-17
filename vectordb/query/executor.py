"""
Query execution for VectorDB.

Executes query plans against collections.

Features:
- Plan-based execution
- Streaming results
- Execution statistics
- Timeout handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator, Callable
import time
import numpy as np
from numpy.typing import NDArray

from .parser import ParsedQuery, QueryType
from .planner import QueryPlan, PlanNode, PlanType
from .filters import Filter, evaluate_filter, create_filter_function


@dataclass
class QueryResult:
    """
    Result from a query execution.
    """
    
    # Result items
    items: List[Dict[str, Any]] = field(default_factory=list)
    
    # Total count (may be more than len(items) for paginated queries)
    total: int = 0
    
    # Execution stats
    stats: "ExecutionStats" = None
    
    def __iter__(self):
        return iter(self.items)
    
    def __len__(self):
        return len(self.items)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": self.items,
            "total": self.total,
            "stats": self.stats.to_dict() if self.stats else None,
        }


@dataclass
class ExecutionStats:
    """
    Statistics from query execution.
    """
    
    # Timing
    total_time_ms: float = 0.0
    search_time_ms: float = 0.0
    filter_time_ms: float = 0.0
    fetch_time_ms: float = 0.0
    
    # Counts
    vectors_scanned: int = 0
    vectors_matched: int = 0
    vectors_returned: int = 0
    
    # Index usage
    index_type: Optional[str] = None
    used_metadata_index: bool = False
    
    # Plan info
    plan_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time_ms": self.total_time_ms,
            "search_time_ms": self.search_time_ms,
            "filter_time_ms": self.filter_time_ms,
            "fetch_time_ms": self.fetch_time_ms,
            "vectors_scanned": self.vectors_scanned,
            "vectors_matched": self.vectors_matched,
            "vectors_returned": self.vectors_returned,
            "index_type": self.index_type,
            "used_metadata_index": self.used_metadata_index,
        }


@dataclass
class QueryContext:
    """
    Context for query execution.
    
    Provides access to collection data and tracks execution.
    """
    
    # Collection interface
    search_fn: Callable  # (query, k, filter_fn) -> results
    get_fn: Callable     # (id) -> (vector, metadata)
    iter_fn: Callable    # () -> Iterator[(id, vector, metadata)]
    count_fn: Callable   # () -> int
    
    # Collection info
    dimension: int
    size: int
    index_type: str
    
    # Execution tracking
    stats: ExecutionStats = field(default_factory=ExecutionStats)
    
    # Timeout
    timeout_seconds: Optional[float] = None
    start_time: float = field(default_factory=time.time)
    
    def check_timeout(self) -> None:
        """Check if timeout has been exceeded."""
        if self.timeout_seconds is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                raise TimeoutError(
                    f"Query timeout exceeded ({elapsed:.2f}s > {self.timeout_seconds}s)"
                )


class QueryExecutor:
    """
    Executes queries against a VectorDB collection.
    
    Example:
        >>> executor = QueryExecutor(collection)
        >>> 
        >>> # Execute a parsed query
        >>> result = executor.execute(parsed_query)
        >>> 
        >>> # Or use convenience methods
        >>> result = executor.search(query_vector, k=10, filter=filter)
        >>> result = executor.get_by_ids(["id1", "id2"])
    """
    
    def __init__(self, collection):
        """
        Initialize executor with a collection.
        
        Args:
            collection: VectorDB Collection instance
        """
        self.collection = collection
        self._setup_context()
    
    def _setup_context(self) -> QueryContext:
        """Set up execution context from collection."""
        
        def search_fn(query, k, filter_fn=None, **kwargs):
            return self.collection.search(
                query=query,
                k=k,
                filter=None,  # We handle filtering separately
                **kwargs,
            )
        
        def get_fn(id):
            return self.collection.get(id)
        
        def iter_fn():
            return self.collection.iter_vectors() if hasattr(self.collection, 'iter_vectors') else iter([])
        
        def count_fn():
            return len(self.collection)
        
        return QueryContext(
            search_fn=search_fn,
            get_fn=get_fn,
            iter_fn=iter_fn,
            count_fn=count_fn,
            dimension=self.collection.dimension,
            size=len(self.collection),
            index_type=getattr(self.collection.config, 'index_type', 'flat'),
        )
    
    def execute(
        self,
        query: ParsedQuery,
        timeout: Optional[float] = None,
    ) -> QueryResult:
        """
        Execute a parsed query.
        
        Args:
            query: Parsed query object
            timeout: Optional timeout in seconds
            
        Returns:
            QueryResult object
        """
        context = self._setup_context()
        context.timeout_seconds = timeout
        context.start_time = time.time()
        context.stats.index_type = context.index_type
        
        start_time = time.time()
        
        try:
            # Execute based on query type
            if query.query_type == QueryType.GET:
                items = self._execute_get(query, context)
            elif query.query_type == QueryType.COUNT:
                items = self._execute_count(query, context)
            elif query.query_type == QueryType.FILTER:
                items = self._execute_filter(query, context)
            else:  # SEARCH or HYBRID
                items = self._execute_search(query, context)
            
            # Update stats
            context.stats.total_time_ms = (time.time() - start_time) * 1000
            context.stats.vectors_returned = len(items)
            
            return QueryResult(
                items=items,
                total=context.stats.vectors_matched or len(items),
                stats=context.stats,
            )
        
        except TimeoutError:
            context.stats.total_time_ms = (time.time() - start_time) * 1000
            raise
    
    def _execute_search(
        self,
        query: ParsedQuery,
        context: QueryContext,
    ) -> List[Dict[str, Any]]:
        """Execute a vector search query."""
        
        if query.vector is None:
            raise ValueError("Search query requires a vector")
        
        query_vector = np.array(query.vector, dtype=np.float32)
        
        # Build filter function
        filter_fn = None
        if query.filter is not None:
            filter_fn = create_filter_function(query.filter)
        
        # Determine k (over-fetch if filtering)
        k = query.k
        if filter_fn is not None:
            # Over-fetch to account for filtered results
            k = min(k * 3, context.size)
        
        # Execute search
        search_start = time.time()
        
        search_kwargs = {}
        if query.ef_search is not None:
            search_kwargs["ef"] = query.ef_search
        if query.n_probe is not None:
            search_kwargs["n_probe"] = query.n_probe
        
        results = self.collection.search(
            query=query_vector,
            k=k,
            filter=query.filter.to_dict() if query.filter and hasattr(query.filter, 'to_dict') else None,
            include_vector=query.include_vector,
            include_metadata=query.include_metadata,
        )
        
        context.stats.search_time_ms = (time.time() - search_start) * 1000
        context.stats.vectors_scanned = context.size  # Approximate
        
        # Convert to items
        items = []
        for r in results:
            item = {
                "id": r.id,
                "distance": r.distance,
                "score": r.score,
            }
            if query.include_metadata and r.metadata:
                item["metadata"] = r.metadata
            if query.include_vector and r.vector is not None:
                item["vector"] = r.vector.tolist() if hasattr(r.vector, 'tolist') else r.vector
            items.append(item)
        
        context.stats.vectors_matched = len(items)
        
        # Apply limit
        if len(items) > query.k:
            items = items[:query.k]
        
        return items
    
    def _execute_filter(
        self,
        query: ParsedQuery,
        context: QueryContext,
    ) -> List[Dict[str, Any]]:
        """Execute a filter-only query."""
        
        filter_start = time.time()
        
        items = []
        count = 0
        
        # Use query_by_metadata if available
        if hasattr(self.collection, 'query_by_metadata'):
            filter_dict = {}
            if query.filter:
                # Convert filter to simple dict if possible
                if hasattr(query.filter, 'to_dict'):
                    filter_dict = query.filter.to_dict()
            
            results = self.collection.query_by_metadata(
                filter=filter_dict,
                limit=query.limit or query.k or 100,
                include_vector=query.include_vector,
            )
            
            for r in results:
                items.append(r)
                count += 1
        else:
            # Fallback: iterate and filter
            for record in self.collection:
                context.check_timeout()
                count += 1
                
                if query.filter and not evaluate_filter(query.filter, record.metadata):
                    continue
                
                item = {
                    "id": record.id,
                    "metadata": record.metadata,
                }
                if query.include_vector:
                    item["vector"] = record.vector.tolist()
                
                items.append(item)
                
                if query.limit and len(items) >= query.limit:
                    break
        
        context.stats.filter_time_ms = (time.time() - filter_start) * 1000
        context.stats.vectors_scanned = count
        context.stats.vectors_matched = len(items)
        
        # Apply sorting
        if query.sort_by:
            reverse = query.sort_order.lower() == "desc"
            items.sort(
                key=lambda x: x.get("metadata", {}).get(query.sort_by, 0),
                reverse=reverse,
            )
        
        # Apply offset/limit
        if query.offset:
            items = items[query.offset:]
        if query.limit:
            items = items[:query.limit]
        
        return items
    
    def _execute_get(
        self,
        query: ParsedQuery,
        context: QueryContext,
    ) -> List[Dict[str, Any]]:
        """Execute a get-by-ID query."""
        
        if not query.ids:
            return []
        
        fetch_start = time.time()
        
        items = []
        for id in query.ids:
            result = self.collection.get(id, include_vector=query.include_vector)
            if result:
                item = {
                    "id": id,
                    "metadata": result.get("metadata", {}),
                }
                if query.include_vector and "vector" in result:
                    vec = result["vector"]
                    item["vector"] = vec.tolist() if hasattr(vec, 'tolist') else vec
                items.append(item)
        
        context.stats.fetch_time_ms = (time.time() - fetch_start) * 1000
        context.stats.vectors_matched = len(items)
        
        return items
    
    def _execute_count(
        self,
        query: ParsedQuery,
        context: QueryContext,
    ) -> List[Dict[str, Any]]:
        """Execute a count query."""
        
        if query.filter is None:
            count = len(self.collection)
        else:
            # Count with filter
            count = 0
            for record in self.collection:
                if evaluate_filter(query.filter, record.metadata):
                    count += 1
        
        context.stats.vectors_matched = count
        
        return [{"count": count}]
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def search(
        self,
        query: NDArray,
        k: int = 10,
        filter: Optional[Filter] = None,
        include_vector: bool = False,
        include_metadata: bool = True,
        **kwargs,
    ) -> QueryResult:
        """
        Convenience method for vector search.
        
        Args:
            query: Query vector
            k: Number of results
            filter: Optional filter
            include_vector: Include vectors in results
            include_metadata: Include metadata in results
            **kwargs: Additional options (ef_search, n_probe)
            
        Returns:
            QueryResult object
        """
        parsed = ParsedQuery(
            query_type=QueryType.SEARCH,
            vector=query.tolist() if hasattr(query, 'tolist') else list(query),
            k=k,
            filter=filter,
            include_vector=include_vector,
            include_metadata=include_metadata,
            ef_search=kwargs.get("ef_search") or kwargs.get("ef"),
            n_probe=kwargs.get("n_probe"),
        )
        
        return self.execute(parsed)
    
    def get_by_ids(
        self,
        ids: List[str],
        include_vector: bool = True,
        include_metadata: bool = True,
    ) -> QueryResult:
        """
        Get vectors by IDs.
        
        Args:
            ids: List of vector IDs
            include_vector: Include vectors
            include_metadata: Include metadata
            
        Returns:
            QueryResult object
        """
        parsed = ParsedQuery(
            query_type=QueryType.GET,
            ids=ids,
            include_vector=include_vector,
            include_metadata=include_metadata,
        )
        
        return self.execute(parsed)
    
    def filter(
        self,
        filter: Filter,
        limit: int = 100,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
        include_vector: bool = False,
    ) -> QueryResult:
        """
        Filter vectors by metadata.
        
        Args:
            filter: Filter to apply
            limit: Maximum results
            offset: Skip first N results
            sort_by: Field to sort by
            sort_order: "asc" or "desc"
            include_vector: Include vectors
            
        Returns:
            QueryResult object
        """
        parsed = ParsedQuery(
            query_type=QueryType.FILTER,
            filter=filter,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
            include_vector=include_vector,
            include_metadata=True,
        )
        
        return self.execute(parsed)
    
    def count(self, filter: Optional[Filter] = None) -> int:
        """
        Count vectors matching filter.
        
        Args:
            filter: Optional filter
            
        Returns:
            Count of matching vectors
        """
        parsed = ParsedQuery(
            query_type=QueryType.COUNT,
            filter=filter,
        )
        
        result = self.execute(parsed)
        return result.items[0]["count"] if result.items else 0