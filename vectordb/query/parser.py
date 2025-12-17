"""
Query parsing for VectorDB.

Parses query strings and dictionaries into structured query objects.

Supports:
- Simple key-value filters
- Complex nested conditions
- Query string syntax
- JSON/dict query format
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import re
import json

from .filters import (
    Filter,
    FilterBuilder,
    FieldFilter,
    FilterOperator,
    AndFilter,
    OrFilter,
    NotFilter,
    filter_from_dict,
)


class QueryType(str, Enum):
    """Types of queries."""
    SEARCH = "search"           # Vector similarity search
    FILTER = "filter"           # Metadata-only filter
    HYBRID = "hybrid"           # Combined search + filter
    GET = "get"                 # Get by ID(s)
    COUNT = "count"             # Count matching vectors
    AGGREGATE = "aggregate"     # Aggregation query


@dataclass
class ParsedQuery:
    """
    Parsed query representation.
    
    Contains all components of a parsed query.
    """
    
    query_type: QueryType = QueryType.SEARCH
    
    # Vector search
    vector: Optional[List[float]] = None
    k: int = 10
    
    # Filtering
    filter: Optional[Filter] = None
    
    # Options
    include_vector: bool = False
    include_metadata: bool = True
    
    # Pagination
    offset: int = 0
    limit: Optional[int] = None
    
    # Sorting (for filter-only queries)
    sort_by: Optional[str] = None
    sort_order: str = "asc"  # "asc" or "desc"
    
    # Index hints
    ef_search: Optional[int] = None     # HNSW
    n_probe: Optional[int] = None       # IVF
    
    # ID-based queries
    ids: Optional[List[str]] = None
    
    # Raw query for debugging
    raw_query: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "query_type": self.query_type.value,
            "k": self.k,
            "include_vector": self.include_vector,
            "include_metadata": self.include_metadata,
            "offset": self.offset,
        }
        
        if self.vector is not None:
            result["vector"] = self.vector
        
        if self.filter is not None:
            result["filter"] = self.filter.to_dict()
        
        if self.limit is not None:
            result["limit"] = self.limit
        
        if self.sort_by is not None:
            result["sort_by"] = self.sort_by
            result["sort_order"] = self.sort_order
        
        if self.ef_search is not None:
            result["ef_search"] = self.ef_search
        
        if self.n_probe is not None:
            result["n_probe"] = self.n_probe
        
        if self.ids is not None:
            result["ids"] = self.ids
        
        return result


class QueryParser:
    """
    Parser for VectorDB queries.
    
    Supports multiple input formats:
    - Dictionary format
    - Query string format
    - Simple key-value format
    
    Example:
        >>> parser = QueryParser()
        >>> 
        >>> # Dictionary format
        >>> query = parser.parse({
        ...     "vector": [0.1, 0.2, 0.3],
        ...     "k": 10,
        ...     "filter": {"category": "electronics"}
        ... })
        >>> 
        >>> # Simple format
        >>> query = parser.parse({"category": "electronics"}, query_type="filter")
    """
    
    # Query string operators
    OPERATORS = {
        ":": FilterOperator.EQ,
        ":=": FilterOperator.EQ,
        ":!": FilterOperator.NE,
        ":!=": FilterOperator.NE,
        ":>": FilterOperator.GT,
        ":>=": FilterOperator.GTE,
        ":<": FilterOperator.LT,
        ":<=": FilterOperator.LTE,
        ":~": FilterOperator.CONTAINS,
        ":^": FilterOperator.STARTSWITH,
        ":$": FilterOperator.ENDSWITH,
        ":*": FilterOperator.REGEX,
    }
    
    def __init__(self):
        pass
    
    def parse(
        self,
        query: Union[Dict[str, Any], str],
        query_type: Optional[QueryType] = None,
    ) -> ParsedQuery:
        """
        Parse a query from various formats.
        
        Args:
            query: Query dictionary or string
            query_type: Override query type
            
        Returns:
            ParsedQuery object
        """
        if isinstance(query, str):
            return self._parse_string(query, query_type)
        else:
            return self._parse_dict(query, query_type)
    
    def _parse_dict(
        self,
        query: Dict[str, Any],
        query_type: Optional[QueryType] = None,
    ) -> ParsedQuery:
        """Parse dictionary format query."""
        
        # Determine query type
        if query_type is None:
            if "vector" in query:
                query_type = QueryType.SEARCH
            elif "ids" in query:
                query_type = QueryType.GET
            elif "count" in query:
                query_type = QueryType.COUNT
            else:
                query_type = QueryType.FILTER
        
        # Parse components
        parsed = ParsedQuery(
            query_type=query_type,
            raw_query=query,
        )
        
        # Vector
        if "vector" in query:
            parsed.vector = query["vector"]
        
        # K (number of results)
        if "k" in query:
            parsed.k = int(query["k"])
        elif "top_k" in query:
            parsed.k = int(query["top_k"])
        elif "limit" in query:
            parsed.k = int(query["limit"])
        
        # Filter
        if "filter" in query:
            parsed.filter = self._parse_filter(query["filter"])
        elif "where" in query:
            parsed.filter = self._parse_filter(query["where"])
        
        # Options
        if "include_vector" in query:
            parsed.include_vector = bool(query["include_vector"])
        if "include_vectors" in query:
            parsed.include_vector = bool(query["include_vectors"])
        if "include_metadata" in query:
            parsed.include_metadata = bool(query["include_metadata"])
        
        # Pagination
        if "offset" in query:
            parsed.offset = int(query["offset"])
        if "skip" in query:
            parsed.offset = int(query["skip"])
        if "limit" in query:
            parsed.limit = int(query["limit"])
        
        # Sorting
        if "sort_by" in query:
            parsed.sort_by = query["sort_by"]
        if "order_by" in query:
            parsed.sort_by = query["order_by"]
        if "sort_order" in query:
            parsed.sort_order = query["sort_order"]
        if "order" in query:
            parsed.sort_order = query["order"]
        
        # Index hints
        if "ef" in query:
            parsed.ef_search = int(query["ef"])
        if "ef_search" in query:
            parsed.ef_search = int(query["ef_search"])
        if "n_probe" in query:
            parsed.n_probe = int(query["n_probe"])
        if "nprobe" in query:
            parsed.n_probe = int(query["nprobe"])
        
        # IDs
        if "ids" in query:
            parsed.ids = query["ids"]
        if "id" in query:
            parsed.ids = [query["id"]]
        
        return parsed
    
    def _parse_filter(
        self,
        filter_spec: Union[Dict[str, Any], List, Filter],
    ) -> Optional[Filter]:
        """Parse a filter specification."""
        
        if filter_spec is None:
            return None
        
        if isinstance(filter_spec, Filter):
            return filter_spec
        
        if isinstance(filter_spec, list):
            # List of conditions (AND)
            filters = [self._parse_filter(f) for f in filter_spec if f]
            filters = [f for f in filters if f is not None]
            if not filters:
                return None
            if len(filters) == 1:
                return filters[0]
            return AndFilter(filters)
        
        if isinstance(filter_spec, dict):
            # Check for logical operators
            if "$and" in filter_spec:
                filters = [self._parse_filter(f) for f in filter_spec["$and"]]
                filters = [f for f in filters if f is not None]
                if not filters:
                    return None
                return AndFilter(filters)
            
            if "$or" in filter_spec:
                filters = [self._parse_filter(f) for f in filter_spec["$or"]]
                filters = [f for f in filters if f is not None]
                if not filters:
                    return None
                return OrFilter(filters)
            
            if "$not" in filter_spec:
                inner = self._parse_filter(filter_spec["$not"])
                if inner:
                    return NotFilter(inner)
                return None
            
            # Check for "type" key (serialized filter)
            if "type" in filter_spec:
                return filter_from_dict(filter_spec)
            
            # Parse as field conditions
            conditions = []
            
            for field, value in filter_spec.items():
                if field.startswith("$"):
                    continue  # Skip operators we don't recognize
                
                if isinstance(value, dict):
                    # Complex condition with operators
                    for op, op_value in value.items():
                        operator = self._parse_operator(op)
                        if operator:
                            conditions.append(FieldFilter(field, operator, op_value))
                else:
                    # Simple equality
                    conditions.append(FieldFilter(field, FilterOperator.EQ, value))
            
            if not conditions:
                return None
            if len(conditions) == 1:
                return conditions[0]
            return AndFilter(conditions)
        
        return None
    
    def _parse_operator(self, op: str) -> Optional[FilterOperator]:
        """Parse an operator string."""
        op = op.lower().lstrip("$")
        
        op_map = {
            "eq": FilterOperator.EQ,
            "equals": FilterOperator.EQ,
            "ne": FilterOperator.NE,
            "neq": FilterOperator.NE,
            "not_equals": FilterOperator.NE,
            "gt": FilterOperator.GT,
            "greater_than": FilterOperator.GT,
            "gte": FilterOperator.GTE,
            "ge": FilterOperator.GTE,
            "greater_than_or_equal": FilterOperator.GTE,
            "lt": FilterOperator.LT,
            "less_than": FilterOperator.LT,
            "lte": FilterOperator.LTE,
            "le": FilterOperator.LTE,
            "less_than_or_equal": FilterOperator.LTE,
            "in": FilterOperator.IN,
            "nin": FilterOperator.NIN,
            "not_in": FilterOperator.NIN,
            "contains": FilterOperator.CONTAINS,
            "icontains": FilterOperator.ICONTAINS,
            "startswith": FilterOperator.STARTSWITH,
            "starts_with": FilterOperator.STARTSWITH,
            "endswith": FilterOperator.ENDSWITH,
            "ends_with": FilterOperator.ENDSWITH,
            "regex": FilterOperator.REGEX,
            "match": FilterOperator.REGEX,
            "exists": FilterOperator.EXISTS,
            "between": FilterOperator.BETWEEN,
            "contains_any": FilterOperator.CONTAINS_ANY,
            "contains_all": FilterOperator.CONTAINS_ALL,
        }
        
        return op_map.get(op)
    
    def _parse_string(
        self,
        query_string: str,
        query_type: Optional[QueryType] = None,
    ) -> ParsedQuery:
        """
        Parse a query string.
        
        Format: "field:value field2:>10 field3:~pattern"
        """
        query_string = query_string.strip()
        
        if not query_string:
            return ParsedQuery(query_type=query_type or QueryType.FILTER)
        
        # Try parsing as JSON first
        if query_string.startswith("{"):
            try:
                query_dict = json.loads(query_string)
                return self._parse_dict(query_dict, query_type)
            except json.JSONDecodeError:
                pass
        
        # Parse as query string
        conditions = []
        
        # Pattern: field[:op]value or "field[:op]value with spaces"
        pattern = r'(\w+(?:\.\w+)*)(:[<>=!~^$*]?=?)("[^"]*"|\'[^\']*\'|\S+)'
        
        for match in re.finditer(pattern, query_string):
            field = match.group(1)
            operator_str = match.group(2)
            value_str = match.group(3)
            
            # Remove quotes from value
            if value_str.startswith('"') and value_str.endswith('"'):
                value_str = value_str[1:-1]
            elif value_str.startswith("'") and value_str.endswith("'"):
                value_str = value_str[1:-1]
            
            # Parse operator
            operator = self.OPERATORS.get(operator_str, FilterOperator.EQ)
            
            # Parse value
            value = self._parse_value(value_str)
            
            conditions.append(FieldFilter(field, operator, value))
        
        filter = None
        if conditions:
            if len(conditions) == 1:
                filter = conditions[0]
            else:
                filter = AndFilter(conditions)
        
        return ParsedQuery(
            query_type=query_type or QueryType.FILTER,
            filter=filter,
            raw_query={"query_string": query_string},
        )
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string into appropriate type."""
        # Boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False
        
        # Null
        if value_str.lower() in ("null", "none", "nil"):
            return None
        
        # Number
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # List (comma-separated)
        if "," in value_str:
            return [self._parse_value(v.strip()) for v in value_str.split(",")]
        
        # String
        return value_str


# Convenience functions
def parse_query(
    query: Union[Dict[str, Any], str],
    query_type: Optional[QueryType] = None,
) -> ParsedQuery:
    """
    Parse a query.
    
    Args:
        query: Query dictionary or string
        query_type: Override query type
        
    Returns:
        ParsedQuery object
    """
    parser = QueryParser()
    return parser.parse(query, query_type)


def parse_filter(
    filter_spec: Union[Dict[str, Any], str, None],
) -> Optional[Filter]:
    """
    Parse a filter specification.
    
    Args:
        filter_spec: Filter dictionary or string
        
    Returns:
        Filter object or None
    """
    if filter_spec is None:
        return None
    
    parser = QueryParser()
    
    if isinstance(filter_spec, str):
        parsed = parser._parse_string(filter_spec)
        return parsed.filter
    else:
        return parser._parse_filter(filter_spec)