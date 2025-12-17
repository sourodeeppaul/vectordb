"""
Advanced metadata filtering for VectorDB queries.

Supports:
- Comparison operators (eq, ne, gt, gte, lt, lte)
- String operators (contains, startswith, endswith, regex)
- Array operators (in, nin, contains_any, contains_all)
- Logical operators (and, or, not)
- Nested field access (field.subfield)

Example:
    >>> # Simple filter
    >>> filter = FieldFilter("category", FilterOperator.EQ, "electronics")
    >>> 
    >>> # Using builder
    >>> filter = (
    ...     FilterBuilder()
    ...     .field("price").gte(10).lte(100)
    ...     .field("category").in_(["electronics", "computers"])
    ...     .build()
    ... )
    >>> 
    >>> # Complex filter with OR
    >>> filter = OrFilter([
    ...     FieldFilter("category", FilterOperator.EQ, "electronics"),
    ...     FieldFilter("price", FilterOperator.LT, 50),
    ... ])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import re
import operator


class FilterOperator(str, Enum):
    """Filter comparison operators."""
    
    # Equality
    EQ = "eq"           # equals
    NE = "ne"           # not equals
    
    # Numeric comparison
    GT = "gt"           # greater than
    GTE = "gte"         # greater than or equal
    LT = "lt"           # less than
    LTE = "lte"         # less than or equal
    
    # Range
    BETWEEN = "between" # between two values
    
    # String operations
    CONTAINS = "contains"       # string contains
    STARTSWITH = "startswith"   # string starts with
    ENDSWITH = "endswith"       # string ends with
    REGEX = "regex"             # regex match
    ICONTAINS = "icontains"     # case-insensitive contains
    
    # Array operations
    IN = "in"                   # value in list
    NIN = "nin"                 # value not in list
    CONTAINS_ANY = "contains_any"   # array contains any of values
    CONTAINS_ALL = "contains_all"   # array contains all of values
    ARRAY_LENGTH = "array_length"   # array length comparison
    
    # Existence
    EXISTS = "exists"           # field exists
    IS_NULL = "is_null"         # field is null
    IS_NOT_NULL = "is_not_null" # field is not null
    
    # Type checking
    TYPE = "type"               # check field type


class Filter(ABC):
    """Abstract base class for all filters."""
    
    @abstractmethod
    def evaluate(self, metadata: Dict[str, Any]) -> bool:
        """
        Evaluate the filter against metadata.
        
        Args:
            metadata: The metadata dictionary to check
            
        Returns:
            True if metadata matches the filter
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary representation."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Filter":
        """Create filter from dictionary representation."""
        pass
    
    def __and__(self, other: "Filter") -> "AndFilter":
        """Combine filters with AND."""
        return AndFilter([self, other])
    
    def __or__(self, other: "Filter") -> "OrFilter":
        """Combine filters with OR."""
        return OrFilter([self, other])
    
    def __invert__(self) -> "NotFilter":
        """Negate filter with NOT."""
        return NotFilter(self)


@dataclass
class FilterCondition:
    """A single filter condition."""
    field: str
    operator: FilterOperator
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterCondition":
        return cls(
            field=data["field"],
            operator=FilterOperator(data["operator"]),
            value=data["value"],
        )


class FieldFilter(Filter):
    """
    Filter on a single field.
    
    Supports nested field access using dot notation:
        FieldFilter("user.profile.age", FilterOperator.GTE, 18)
    """
    
    def __init__(
        self,
        field: str,
        operator: FilterOperator,
        value: Any,
    ):
        self.field = field
        self.operator = operator
        self.value = value
    
    def _get_field_value(self, metadata: Dict[str, Any]) -> Any:
        """Get field value, supporting nested access."""
        parts = self.field.split(".")
        current = metadata
        
        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    return None
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        
        return current
    
    def evaluate(self, metadata: Dict[str, Any]) -> bool:
        """Evaluate the filter."""
        field_value = self._get_field_value(metadata)
        
        try:
            return self._compare(field_value, self.operator, self.value)
        except (TypeError, ValueError):
            return False
    
    def _compare(
        self,
        field_value: Any,
        op: FilterOperator,
        compare_value: Any,
    ) -> bool:
        """Compare field value using operator."""
        
        # Equality
        if op == FilterOperator.EQ:
            return field_value == compare_value
        
        if op == FilterOperator.NE:
            return field_value != compare_value
        
        # Numeric comparison
        if op == FilterOperator.GT:
            return field_value is not None and field_value > compare_value
        
        if op == FilterOperator.GTE:
            return field_value is not None and field_value >= compare_value
        
        if op == FilterOperator.LT:
            return field_value is not None and field_value < compare_value
        
        if op == FilterOperator.LTE:
            return field_value is not None and field_value <= compare_value
        
        # Range
        if op == FilterOperator.BETWEEN:
            if field_value is None or not isinstance(compare_value, (list, tuple)):
                return False
            low, high = compare_value[0], compare_value[1]
            return low <= field_value <= high
        
        # String operations
        if op == FilterOperator.CONTAINS:
            return (
                isinstance(field_value, str) and
                isinstance(compare_value, str) and
                compare_value in field_value
            )
        
        if op == FilterOperator.ICONTAINS:
            return (
                isinstance(field_value, str) and
                isinstance(compare_value, str) and
                compare_value.lower() in field_value.lower()
            )
        
        if op == FilterOperator.STARTSWITH:
            return (
                isinstance(field_value, str) and
                isinstance(compare_value, str) and
                field_value.startswith(compare_value)
            )
        
        if op == FilterOperator.ENDSWITH:
            return (
                isinstance(field_value, str) and
                isinstance(compare_value, str) and
                field_value.endswith(compare_value)
            )
        
        if op == FilterOperator.REGEX:
            return (
                isinstance(field_value, str) and
                isinstance(compare_value, str) and
                bool(re.search(compare_value, field_value))
            )
        
        # Array operations
        if op == FilterOperator.IN:
            return field_value in compare_value
        
        if op == FilterOperator.NIN:
            return field_value not in compare_value
        
        if op == FilterOperator.CONTAINS_ANY:
            if not isinstance(field_value, (list, tuple, set)):
                return False
            return any(v in field_value for v in compare_value)
        
        if op == FilterOperator.CONTAINS_ALL:
            if not isinstance(field_value, (list, tuple, set)):
                return False
            return all(v in field_value for v in compare_value)
        
        if op == FilterOperator.ARRAY_LENGTH:
            if not isinstance(field_value, (list, tuple)):
                return False
            # compare_value should be {"op": "eq/gt/lt/...", "value": n}
            if isinstance(compare_value, int):
                return len(field_value) == compare_value
            elif isinstance(compare_value, dict):
                length = len(field_value)
                sub_op = compare_value.get("op", "eq")
                sub_val = compare_value.get("value", 0)
                if sub_op == "eq":
                    return length == sub_val
                elif sub_op == "gt":
                    return length > sub_val
                elif sub_op == "gte":
                    return length >= sub_val
                elif sub_op == "lt":
                    return length < sub_val
                elif sub_op == "lte":
                    return length <= sub_val
            return False
        
        # Existence
        if op == FilterOperator.EXISTS:
            exists = field_value is not None
            return exists == compare_value
        
        if op == FilterOperator.IS_NULL:
            return field_value is None
        
        if op == FilterOperator.IS_NOT_NULL:
            return field_value is not None
        
        # Type checking
        if op == FilterOperator.TYPE:
            type_map = {
                "string": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "null": type(None),
            }
            expected_type = type_map.get(compare_value)
            if expected_type:
                return isinstance(field_value, expected_type)
            return False
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "field",
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldFilter":
        return cls(
            field=data["field"],
            operator=FilterOperator(data["operator"]),
            value=data["value"],
        )
    
    def __repr__(self) -> str:
        return f"FieldFilter({self.field} {self.operator.value} {self.value!r})"


class AndFilter(Filter):
    """Logical AND of multiple filters."""
    
    def __init__(self, filters: List[Filter]):
        self.filters = filters
    
    def evaluate(self, metadata: Dict[str, Any]) -> bool:
        return all(f.evaluate(metadata) for f in self.filters)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "and",
            "filters": [f.to_dict() for f in self.filters],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AndFilter":
        filters = [filter_from_dict(f) for f in data["filters"]]
        return cls(filters)
    
    def __repr__(self) -> str:
        return f"AndFilter({self.filters})"


class OrFilter(Filter):
    """Logical OR of multiple filters."""
    
    def __init__(self, filters: List[Filter]):
        self.filters = filters
    
    def evaluate(self, metadata: Dict[str, Any]) -> bool:
        return any(f.evaluate(metadata) for f in self.filters)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "or",
            "filters": [f.to_dict() for f in self.filters],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrFilter":
        filters = [filter_from_dict(f) for f in data["filters"]]
        return cls(filters)
    
    def __repr__(self) -> str:
        return f"OrFilter({self.filters})"


class NotFilter(Filter):
    """Logical NOT of a filter."""
    
    def __init__(self, filter: Filter):
        self.filter = filter
    
    def evaluate(self, metadata: Dict[str, Any]) -> bool:
        return not self.filter.evaluate(metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "not",
            "filter": self.filter.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotFilter":
        filter = filter_from_dict(data["filter"])
        return cls(filter)
    
    def __repr__(self) -> str:
        return f"NotFilter({self.filter})"


def filter_from_dict(data: Dict[str, Any]) -> Filter:
    """Create a filter from dictionary representation."""
    filter_type = data.get("type", "field")
    
    if filter_type == "field":
        return FieldFilter.from_dict(data)
    elif filter_type == "and":
        return AndFilter.from_dict(data)
    elif filter_type == "or":
        return OrFilter.from_dict(data)
    elif filter_type == "not":
        return NotFilter.from_dict(data)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


class FieldFilterBuilder:
    """Builder for field filters with fluent API."""
    
    def __init__(self, parent: "FilterBuilder", field: str):
        self._parent = parent
        self._field = field
    
    def equals(self, value: Any) -> "FilterBuilder":
        """Field equals value."""
        self._parent._add_condition(self._field, FilterOperator.EQ, value)
        return self._parent
    
    def eq(self, value: Any) -> "FilterBuilder":
        """Alias for equals."""
        return self.equals(value)
    
    def not_equals(self, value: Any) -> "FilterBuilder":
        """Field not equals value."""
        self._parent._add_condition(self._field, FilterOperator.NE, value)
        return self._parent
    
    def ne(self, value: Any) -> "FilterBuilder":
        """Alias for not_equals."""
        return self.not_equals(value)
    
    def greater_than(self, value: Any) -> "FilterBuilder":
        """Field greater than value."""
        self._parent._add_condition(self._field, FilterOperator.GT, value)
        return self._parent
    
    def gt(self, value: Any) -> "FilterBuilder":
        """Alias for greater_than."""
        return self.greater_than(value)
    
    def greater_than_or_equal(self, value: Any) -> "FilterBuilder":
        """Field greater than or equal to value."""
        self._parent._add_condition(self._field, FilterOperator.GTE, value)
        return self._parent
    
    def gte(self, value: Any) -> "FilterBuilder":
        """Alias for greater_than_or_equal."""
        return self.greater_than_or_equal(value)
    
    def less_than(self, value: Any) -> "FilterBuilder":
        """Field less than value."""
        self._parent._add_condition(self._field, FilterOperator.LT, value)
        return self._parent
    
    def lt(self, value: Any) -> "FilterBuilder":
        """Alias for less_than."""
        return self.less_than(value)
    
    def less_than_or_equal(self, value: Any) -> "FilterBuilder":
        """Field less than or equal to value."""
        self._parent._add_condition(self._field, FilterOperator.LTE, value)
        return self._parent
    
    def lte(self, value: Any) -> "FilterBuilder":
        """Alias for less_than_or_equal."""
        return self.less_than_or_equal(value)
    
    def between(self, low: Any, high: Any) -> "FilterBuilder":
        """Field between low and high (inclusive)."""
        self._parent._add_condition(self._field, FilterOperator.BETWEEN, [low, high])
        return self._parent
    
    def in_(self, values: List[Any]) -> "FilterBuilder":
        """Field value in list."""
        self._parent._add_condition(self._field, FilterOperator.IN, values)
        return self._parent
    
    def not_in(self, values: List[Any]) -> "FilterBuilder":
        """Field value not in list."""
        self._parent._add_condition(self._field, FilterOperator.NIN, values)
        return self._parent
    
    def contains(self, value: str) -> "FilterBuilder":
        """String field contains value."""
        self._parent._add_condition(self._field, FilterOperator.CONTAINS, value)
        return self._parent
    
    def icontains(self, value: str) -> "FilterBuilder":
        """Case-insensitive contains."""
        self._parent._add_condition(self._field, FilterOperator.ICONTAINS, value)
        return self._parent
    
    def startswith(self, value: str) -> "FilterBuilder":
        """String starts with value."""
        self._parent._add_condition(self._field, FilterOperator.STARTSWITH, value)
        return self._parent
    
    def endswith(self, value: str) -> "FilterBuilder":
        """String ends with value."""
        self._parent._add_condition(self._field, FilterOperator.ENDSWITH, value)
        return self._parent
    
    def regex(self, pattern: str) -> "FilterBuilder":
        """String matches regex pattern."""
        self._parent._add_condition(self._field, FilterOperator.REGEX, pattern)
        return self._parent
    
    def contains_any(self, values: List[Any]) -> "FilterBuilder":
        """Array field contains any of values."""
        self._parent._add_condition(self._field, FilterOperator.CONTAINS_ANY, values)
        return self._parent
    
    def contains_all(self, values: List[Any]) -> "FilterBuilder":
        """Array field contains all of values."""
        self._parent._add_condition(self._field, FilterOperator.CONTAINS_ALL, values)
        return self._parent
    
    def exists(self, exists: bool = True) -> "FilterBuilder":
        """Field exists (or not)."""
        self._parent._add_condition(self._field, FilterOperator.EXISTS, exists)
        return self._parent
    
    def is_null(self) -> "FilterBuilder":
        """Field is null."""
        self._parent._add_condition(self._field, FilterOperator.IS_NULL, True)
        return self._parent
    
    def is_not_null(self) -> "FilterBuilder":
        """Field is not null."""
        self._parent._add_condition(self._field, FilterOperator.IS_NOT_NULL, True)
        return self._parent
    
    def is_type(self, type_name: str) -> "FilterBuilder":
        """Field is of type (string, int, float, bool, list, dict, null)."""
        self._parent._add_condition(self._field, FilterOperator.TYPE, type_name)
        return self._parent


class FilterBuilder:
    """
    Fluent builder for creating filters.
    
    Example:
        >>> filter = (
        ...     FilterBuilder()
        ...     .field("category").equals("electronics")
        ...     .field("price").gte(10).lte(100)
        ...     .field("tags").contains_any(["sale", "new"])
        ...     .build()
        ... )
    """
    
    def __init__(self):
        self._conditions: List[FieldFilter] = []
        self._logic = "and"  # "and" or "or"
    
    def field(self, name: str) -> FieldFilterBuilder:
        """Start building a condition for a field."""
        return FieldFilterBuilder(self, name)
    
    def _add_condition(
        self,
        field: str,
        operator: FilterOperator,
        value: Any,
    ) -> None:
        """Add a condition (internal)."""
        self._conditions.append(FieldFilter(field, operator, value))
    
    def or_(self) -> "FilterBuilder":
        """Switch to OR logic for subsequent conditions."""
        self._logic = "or"
        return self
    
    def and_(self) -> "FilterBuilder":
        """Switch to AND logic (default)."""
        self._logic = "and"
        return self
    
    def build(self) -> Optional[Filter]:
        """Build the final filter."""
        if not self._conditions:
            return None
        
        if len(self._conditions) == 1:
            return self._conditions[0]
        
        if self._logic == "and":
            return AndFilter(self._conditions)
        else:
            return OrFilter(self._conditions)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Optional[Filter]:
        """Create filter from simple dictionary format."""
        if not data:
            return None
        
        conditions = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                # Complex condition with operator
                for op, op_value in value.items():
                    operator = FilterOperator(op.lstrip("$"))
                    conditions.append(FieldFilter(key, operator, op_value))
            else:
                # Simple equality
                conditions.append(FieldFilter(key, FilterOperator.EQ, value))
        
        if len(conditions) == 1:
            return conditions[0]
        
        return AndFilter(conditions)


def evaluate_filter(
    filter: Optional[Filter],
    metadata: Dict[str, Any],
) -> bool:
    """
    Evaluate a filter against metadata.
    
    Args:
        filter: The filter to evaluate (None = always True)
        metadata: The metadata dictionary
        
    Returns:
        True if metadata matches the filter
    """
    if filter is None:
        return True
    return filter.evaluate(metadata)


def create_filter_function(
    filter: Optional[Filter],
) -> Callable[[str, Dict[str, Any]], bool]:
    """
    Create a filter function for use with index search.
    
    Args:
        filter: The filter to apply
        
    Returns:
        A callable (id, metadata) -> bool
    """
    if filter is None:
        return lambda id, meta: True
    
    def filter_fn(id: str, metadata: Dict[str, Any]) -> bool:
        return filter.evaluate(metadata)
    
    return filter_fn