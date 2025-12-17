"""
Query processing module for VectorDB.

This module provides:
- Query parsing and validation
- Query planning and optimization
- Query execution
- Advanced metadata filtering

Example:
    >>> from vectordb.query import QueryExecutor, FilterBuilder
    >>> 
    >>> # Build a filter
    >>> filter = (
    ...     FilterBuilder()
    ...     .field("category").equals("electronics")
    ...     .field("price").less_than(100)
    ...     .field("tags").contains("sale")
    ...     .build()
    ... )
    >>> 
    >>> # Execute query
    >>> executor = QueryExecutor(collection)
    >>> results = executor.search(query_vector, k=10, filter=filter)
"""

from .filters import (
    Filter,
    FilterBuilder,
    FilterCondition,
    FilterOperator,
    AndFilter,
    OrFilter,
    NotFilter,
    FieldFilter,
    evaluate_filter,
)

from .parser import (
    QueryParser,
    ParsedQuery,
    QueryType,
    parse_query,
    parse_filter,
)

from .planner import (
    QueryPlanner,
    QueryPlan,
    PlanNode,
    PlanType,
    optimize_plan,
)

from .executor import (
    QueryExecutor,
    QueryResult,
    QueryContext,
    ExecutionStats,
)

__all__ = [
    # Filters
    "Filter",
    "FilterBuilder",
    "FilterCondition",
    "FilterOperator",
    "AndFilter",
    "OrFilter",
    "NotFilter",
    "FieldFilter",
    "evaluate_filter",
    # Parser
    "QueryParser",
    "ParsedQuery",
    "QueryType",
    "parse_query",
    "parse_filter",
    # Planner
    "QueryPlanner",
    "QueryPlan",
    "PlanNode",
    "PlanType",
    "optimize_plan",
    # Executor
    "QueryExecutor",
    "QueryResult",
    "QueryContext",
    "ExecutionStats",
]