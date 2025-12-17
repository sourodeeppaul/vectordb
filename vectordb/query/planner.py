"""
Query planning and optimization for VectorDB.

Analyzes queries and creates optimized execution plans.

Features:
- Index selection
- Filter pushdown
- Cost estimation
- Parallel execution planning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import time

from .parser import ParsedQuery, QueryType
from .filters import Filter, FieldFilter, AndFilter, OrFilter, NotFilter


class PlanType(str, Enum):
    """Types of plan nodes."""
    
    # Scan operations
    FULL_SCAN = "full_scan"           # Scan all vectors
    INDEX_SCAN = "index_scan"         # Use vector index
    METADATA_SCAN = "metadata_scan"   # Scan by metadata only
    
    # Filter operations
    FILTER = "filter"                 # Apply filter
    PRE_FILTER = "pre_filter"         # Filter before search
    POST_FILTER = "post_filter"       # Filter after search
    
    # Search operations
    KNN_SEARCH = "knn_search"         # K-nearest neighbor search
    RANGE_SEARCH = "range_search"     # Range-based search
    
    # Set operations
    INTERSECT = "intersect"           # Intersection of results
    UNION = "union"                   # Union of results
    
    # Other
    FETCH = "fetch"                   # Fetch full vectors/metadata
    SORT = "sort"                     # Sort results
    LIMIT = "limit"                   # Limit results
    PROJECT = "project"               # Select fields


@dataclass
class PlanNode:
    """
    A node in the query execution plan.
    
    Forms a tree structure representing the execution plan.
    """
    
    type: PlanType
    
    # Child nodes
    children: List["PlanNode"] = field(default_factory=list)
    
    # Node-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Cost estimates
    estimated_cost: float = 0.0
    estimated_rows: int = 0
    
    # Execution stats (filled after execution)
    actual_rows: int = 0
    execution_time_ms: float = 0.0
    
    def add_child(self, child: "PlanNode") -> None:
        """Add a child node."""
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "params": self.params,
            "estimated_cost": self.estimated_cost,
            "estimated_rows": self.estimated_rows,
            "children": [c.to_dict() for c in self.children],
        }
    
    def explain(self, indent: int = 0) -> str:
        """Generate explain output."""
        prefix = "  " * indent
        lines = [f"{prefix}{self.type.value}"]
        
        if self.params:
            for key, value in self.params.items():
                lines.append(f"{prefix}  {key}: {value}")
        
        lines.append(f"{prefix}  cost: {self.estimated_cost:.2f}, rows: {self.estimated_rows}")
        
        for child in self.children:
            lines.append(child.explain(indent + 1))
        
        return "\n".join(lines)


@dataclass
class QueryPlan:
    """
    Complete query execution plan.
    """
    
    # Root node of the plan tree
    root: PlanNode
    
    # Original query
    query: ParsedQuery
    
    # Plan metadata
    index_type: Optional[str] = None
    uses_metadata_index: bool = False
    estimated_total_cost: float = 0.0
    
    # Planning stats
    planning_time_ms: float = 0.0
    
    def explain(self) -> str:
        """Generate explain output."""
        lines = [
            "Query Plan",
            "=" * 40,
            f"Query Type: {self.query.query_type.value}",
            f"Index: {self.index_type or 'none'}",
            f"Uses Metadata Index: {self.uses_metadata_index}",
            f"Estimated Cost: {self.estimated_total_cost:.2f}",
            f"Planning Time: {self.planning_time_ms:.2f}ms",
            "",
            "Plan Tree:",
            "-" * 40,
            self.root.explain(),
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "root": self.root.to_dict(),
            "index_type": self.index_type,
            "uses_metadata_index": self.uses_metadata_index,
            "estimated_total_cost": self.estimated_total_cost,
            "planning_time_ms": self.planning_time_ms,
        }


class CostModel:
    """
    Cost estimation model for query planning.
    
    Estimates the cost of various operations based on
    collection statistics.
    """
    
    # Base costs (arbitrary units)
    VECTOR_DISTANCE_COST = 1.0
    METADATA_COMPARE_COST = 0.1
    FETCH_COST = 0.5
    SORT_COST_PER_ITEM = 0.01
    
    # Index-specific costs
    FLAT_SCAN_COST_MULTIPLIER = 1.0
    HNSW_SEARCH_COST_MULTIPLIER = 0.1
    IVF_SEARCH_COST_MULTIPLIER = 0.2
    
    def __init__(
        self,
        collection_size: int,
        dimension: int,
        index_type: str = "flat",
    ):
        self.collection_size = collection_size
        self.dimension = dimension
        self.index_type = index_type
    
    def estimate_scan_cost(self, n_vectors: int) -> float:
        """Estimate cost of scanning n vectors."""
        base = n_vectors * self.VECTOR_DISTANCE_COST
        
        if self.index_type == "flat":
            return base * self.FLAT_SCAN_COST_MULTIPLIER
        elif self.index_type == "hnsw":
            # HNSW is O(log n)
            import math
            return math.log(n_vectors + 1) * 100 * self.HNSW_SEARCH_COST_MULTIPLIER
        elif self.index_type == "ivf":
            # IVF is O(n/k * n_probe)
            return (n_vectors / 100) * 10 * self.IVF_SEARCH_COST_MULTIPLIER
        
        return base
    
    def estimate_filter_cost(self, n_items: int, filter_complexity: int) -> float:
        """Estimate cost of filtering n items."""
        return n_items * filter_complexity * self.METADATA_COMPARE_COST
    
    def estimate_filter_selectivity(self, filter: Optional[Filter]) -> float:
        """
        Estimate what fraction of items will pass the filter.
        
        Returns a value between 0 and 1.
        """
        if filter is None:
            return 1.0
        
        # Simple heuristics
        if isinstance(filter, FieldFilter):
            if filter.operator in (filter.operator.EQ, filter.operator.IN):
                return 0.1  # Assume 10% selectivity for equality
            elif filter.operator in (filter.operator.GT, filter.operator.LT,
                                     filter.operator.GTE, filter.operator.LTE):
                return 0.3  # 30% for range
            else:
                return 0.5  # 50% for others
        
        elif isinstance(filter, AndFilter):
            # Multiply selectivities
            selectivity = 1.0
            for f in filter.filters:
                selectivity *= self.estimate_filter_selectivity(f)
            return selectivity
        
        elif isinstance(filter, OrFilter):
            # Approximate: 1 - (1-s1)(1-s2)...
            selectivity = 1.0
            for f in filter.filters:
                s = self.estimate_filter_selectivity(f)
                selectivity *= (1 - s)
            return 1 - selectivity
        
        elif isinstance(filter, NotFilter):
            return 1 - self.estimate_filter_selectivity(filter.filter)
        
        return 0.5
    
    def estimate_fetch_cost(self, n_items: int, include_vector: bool) -> float:
        """Estimate cost of fetching n items."""
        base = n_items * self.FETCH_COST
        if include_vector:
            base += n_items * self.dimension * 0.01  # Vector fetch cost
        return base
    
    def filter_complexity(self, filter: Optional[Filter]) -> int:
        """Estimate filter complexity (number of conditions)."""
        if filter is None:
            return 0
        
        if isinstance(filter, FieldFilter):
            return 1
        
        elif isinstance(filter, (AndFilter, OrFilter)):
            return sum(self.filter_complexity(f) for f in filter.filters)
        
        elif isinstance(filter, NotFilter):
            return self.filter_complexity(filter.filter)
        
        return 1


class QueryPlanner:
    """
    Query planner for VectorDB.
    
    Creates optimized execution plans for queries.
    
    Example:
        >>> planner = QueryPlanner(collection)
        >>> plan = planner.plan(parsed_query)
        >>> print(plan.explain())
    """
    
    def __init__(
        self,
        collection_size: int,
        dimension: int,
        index_type: str = "flat",
        has_metadata_index: bool = True,
    ):
        self.collection_size = collection_size
        self.dimension = dimension
        self.index_type = index_type
        self.has_metadata_index = has_metadata_index
        
        self.cost_model = CostModel(collection_size, dimension, index_type)
    
    def plan(self, query: ParsedQuery) -> QueryPlan:
        """
        Create an execution plan for a query.
        
        Args:
            query: Parsed query
            
        Returns:
            QueryPlan object
        """
        start_time = time.time()
        
        # Build plan based on query type
        if query.query_type == QueryType.GET:
            root = self._plan_get(query)
        elif query.query_type == QueryType.COUNT:
            root = self._plan_count(query)
        elif query.query_type == QueryType.FILTER:
            root = self._plan_filter_only(query)
        else:  # SEARCH or HYBRID
            root = self._plan_search(query)
        
        # Calculate total cost
        total_cost = self._calculate_total_cost(root)
        
        planning_time = (time.time() - start_time) * 1000
        
        return QueryPlan(
            root=root,
            query=query,
            index_type=self.index_type,
            uses_metadata_index=self._uses_metadata_index(query),
            estimated_total_cost=total_cost,
            planning_time_ms=planning_time,
        )
    
    def _plan_search(self, query: ParsedQuery) -> PlanNode:
        """Plan a vector search query."""
        
        # Estimate selectivity
        selectivity = self.cost_model.estimate_filter_selectivity(query.filter)
        
        # Decide on filter strategy
        if query.filter is None:
            # No filter - pure vector search
            return self._create_search_node(query)
        
        elif selectivity < 0.1 and self.has_metadata_index:
            # Highly selective filter - pre-filter
            # 1. Apply filter first to get candidates
            # 2. Search only among candidates
            filter_node = PlanNode(
                type=PlanType.PRE_FILTER,
                params={"filter": query.filter.to_dict() if query.filter else None},
                estimated_rows=int(self.collection_size * selectivity),
            )
            
            search_node = self._create_search_node(query)
            search_node.params["candidates_only"] = True
            search_node.estimated_rows = min(query.k, filter_node.estimated_rows)
            
            filter_node.add_child(search_node)
            return filter_node
        
        else:
            # Post-filter approach
            # 1. Get more results from search
            # 2. Filter after
            search_node = self._create_search_node(query)
            
            # Request more results to account for filtering
            over_fetch = min(int(query.k / selectivity * 1.5), self.collection_size)
            search_node.params["k"] = over_fetch
            search_node.estimated_rows = over_fetch
            
            filter_node = PlanNode(
                type=PlanType.POST_FILTER,
                params={"filter": query.filter.to_dict() if query.filter else None},
                estimated_rows=query.k,
            )
            filter_node.add_child(search_node)
            
            return filter_node
    
    def _create_search_node(self, query: ParsedQuery) -> PlanNode:
        """Create a search plan node."""
        params = {
            "k": query.k,
        }
        
        if query.ef_search is not None:
            params["ef_search"] = query.ef_search
        
        if query.n_probe is not None:
            params["n_probe"] = query.n_probe
        
        cost = self.cost_model.estimate_scan_cost(self.collection_size)
        
        return PlanNode(
            type=PlanType.KNN_SEARCH,
            params=params,
            estimated_cost=cost,
            estimated_rows=query.k,
        )
    
    def _plan_filter_only(self, query: ParsedQuery) -> PlanNode:
        """Plan a filter-only query (no vector search)."""
        
        selectivity = self.cost_model.estimate_filter_selectivity(query.filter)
        estimated_rows = int(self.collection_size * selectivity)
        
        # Decide on scan strategy
        if self.has_metadata_index and selectivity < 0.3:
            # Use metadata index
            scan_type = PlanType.METADATA_SCAN
        else:
            # Full scan
            scan_type = PlanType.FULL_SCAN
        
        scan_node = PlanNode(
            type=scan_type,
            params={"filter": query.filter.to_dict() if query.filter else None},
            estimated_rows=estimated_rows,
        )
        
        # Add sort if needed
        if query.sort_by:
            sort_node = PlanNode(
                type=PlanType.SORT,
                params={
                    "field": query.sort_by,
                    "order": query.sort_order,
                },
                estimated_rows=estimated_rows,
            )
            sort_node.add_child(scan_node)
            scan_node = sort_node
        
        # Add limit if needed
        if query.limit or query.k:
            limit = query.limit or query.k
            limit_node = PlanNode(
                type=PlanType.LIMIT,
                params={
                    "limit": limit,
                    "offset": query.offset,
                },
                estimated_rows=min(limit, estimated_rows),
            )
            limit_node.add_child(scan_node)
            return limit_node
        
        return scan_node
    
    def _plan_get(self, query: ParsedQuery) -> PlanNode:
        """Plan a get-by-ID query."""
        return PlanNode(
            type=PlanType.FETCH,
            params={
                "ids": query.ids,
                "include_vector": query.include_vector,
                "include_metadata": query.include_metadata,
            },
            estimated_rows=len(query.ids) if query.ids else 0,
        )
    
    def _plan_count(self, query: ParsedQuery) -> PlanNode:
        """Plan a count query."""
        selectivity = self.cost_model.estimate_filter_selectivity(query.filter)
        
        return PlanNode(
            type=PlanType.METADATA_SCAN if self.has_metadata_index else PlanType.FULL_SCAN,
            params={
                "filter": query.filter.to_dict() if query.filter else None,
                "count_only": True,
            },
            estimated_rows=int(self.collection_size * selectivity),
        )
    
    def _calculate_total_cost(self, node: PlanNode) -> float:
        """Calculate total cost of a plan tree."""
        cost = node.estimated_cost
        
        for child in node.children:
            cost += self._calculate_total_cost(child)
        
        return cost
    
    def _uses_metadata_index(self, query: ParsedQuery) -> bool:
        """Check if query can use metadata index."""
        if not self.has_metadata_index:
            return False
        
        if query.filter is None:
            return False
        
        selectivity = self.cost_model.estimate_filter_selectivity(query.filter)
        return selectivity < 0.5


def optimize_plan(plan: QueryPlan) -> QueryPlan:
    """
    Optimize a query plan.
    
    Applies optimization rules to improve performance.
    
    Args:
        plan: Query plan to optimize
        
    Returns:
        Optimized plan
    """
    # Currently just returns the plan as-is
    # Future optimizations:
    # - Predicate pushdown
    # - Join reordering
    # - Parallel execution
    # - Caching hints
    
    return plan