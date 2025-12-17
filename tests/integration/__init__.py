"""
Integration tests for VectorDB.

These tests verify that all components work together correctly,
including database operations, persistence, and complex queries.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


def get_temp_dir():
    """Create a temporary directory for test data."""
    return tempfile.mkdtemp(prefix="vectordb_integration_")


def cleanup_temp_dir(path: str):
    """Clean up temporary directory."""
    shutil.rmtree(path, ignore_errors=True)


# Integration test markers
integration = pytest.mark.integration
slow = pytest.mark.slow
requires_persistence = pytest.mark.requires_persistence