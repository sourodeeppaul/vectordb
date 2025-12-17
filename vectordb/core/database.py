"""
VectorDB - Main database class managing multiple collections.
"""

from __future__ import annotations

import os
import json
import time
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pickle

from .collection import Collection, CollectionConfig
from .exceptions import (
    CollectionNotFoundError,
    CollectionExistsError,
    StorageError,
    ValidationError,
)
from ..utils.validation import validate_id
from ..utils.logging import get_logger


logger = get_logger(__name__)


class VectorDB:
    """
    Main database class for VectorDB.
    
    Manages multiple collections, provides persistence,
    and serves as the main entry point for the database.
    
    Example:
        >>> db = VectorDB(data_dir="./my_data")
        >>> 
        >>> # Create a collection
        >>> collection = db.create_collection(
        ...     "documents",
        ...     dimension=384,
        ...     metric="cosine"
        ... )
        >>> 
        >>> # Add vectors
        >>> collection.add("doc1", embedding, {"title": "Hello"})
        >>> 
        >>> # Search
        >>> results = collection.search(query_embedding, k=10)
        >>> 
        >>> # Save to disk
        >>> db.save()
        >>> 
        >>> # Load from disk
        >>> db = VectorDB.load("./my_data")
    """
    
    VERSION = "0.1.0"
    MANIFEST_FILE = "manifest.json"
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        auto_save: bool = False,
        auto_save_interval: int = 60,
    ):
        """
        Initialize VectorDB.
        
        Args:
            data_dir: Directory for persistent storage (None = in-memory only)
            auto_save: Whether to auto-save periodically
            auto_save_interval: Seconds between auto-saves
        """
        self._collections: Dict[str, Collection] = {}
        self._data_dir = Path(data_dir) if data_dir else None
        self._lock = threading.RLock()
        
        self._created_at = time.time()
        self._updated_at = time.time()
        
        # Auto-save
        self._auto_save = auto_save
        self._auto_save_interval = auto_save_interval
        self._auto_save_thread: Optional[threading.Thread] = None
        self._running = True
        
        # Create data directory if needed
        if self._data_dir:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"VectorDB initialized with data_dir: {self._data_dir}")
        else:
            logger.info("VectorDB initialized in memory-only mode")
        
        # Start auto-save thread if enabled
        if auto_save and self._data_dir:
            self._start_auto_save()
    
    # =========================================================================
    # COLLECTION MANAGEMENT
    # =========================================================================
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "euclidean",
        normalize: bool = False,
        max_vectors: Optional[int] = None,
        exist_ok: bool = False,
        **kwargs,
    ) -> Collection:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            metric: Distance metric
            normalize: Whether to normalize vectors
            max_vectors: Maximum vectors (None = unlimited)
            exist_ok: If True, return existing collection instead of error
            **kwargs: Additional collection config
            
        Returns:
            The created or existing Collection
            
        Raises:
            CollectionExistsError: If collection exists and exist_ok=False
        """
        name = validate_id(name)
        
        with self._lock:
            if name in self._collections:
                if exist_ok:
                    return self._collections[name]
                raise CollectionExistsError(f"Collection '{name}' already exists")
            
            collection = Collection(
                name=name,
                dimension=dimension,
                metric=metric,
                normalize=normalize,
                max_vectors=max_vectors,
                **kwargs,
            )
            
            self._collections[name] = collection
            self._updated_at = time.time()
            
            logger.info(
                f"Created collection '{name}' "
                f"(dim={dimension}, metric={metric})"
            )
            
            return collection
    
    def get_collection(self, name: str) -> Collection:
        """
        Get a collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            The Collection
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        with self._lock:
            if name not in self._collections:
                raise CollectionNotFoundError(f"Collection '{name}' not found")
            return self._collections[name]
    
    def get_or_create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "euclidean",
        **kwargs,
    ) -> Collection:
        """
        Get existing collection or create new one.
        
        Args:
            name: Collection name
            dimension: Vector dimension (for creation)
            metric: Distance metric (for creation)
            **kwargs: Additional config (for creation)
            
        Returns:
            The Collection
        """
        return self.create_collection(
            name=name,
            dimension=dimension,
            metric=metric,
            exist_ok=True,
            **kwargs,
        )
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if name not in self._collections:
                return False
            
            del self._collections[name]
            self._updated_at = time.time()
            
            # Delete from disk if persistent
            if self._data_dir:
                collection_path = self._data_dir / f"{name}.collection"
                if collection_path.exists():
                    collection_path.unlink()
            
            logger.info(f"Deleted collection '{name}'")
            return True
    
    def list_collections(self) -> List[str]:
        """
        List all collection names.
        
        Returns:
            List of collection names
        """
        with self._lock:
            return list(self._collections.keys())
    
    def has_collection(self, name: str) -> bool:
        """
        Check if collection exists.
        
        Args:
            name: Collection name
            
        Returns:
            True if exists
        """
        return name in self._collections
    
    def collection_stats(self, name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Collection statistics dictionary
        """
        collection = self.get_collection(name)
        return collection.stats().to_dict()
    
    # =========================================================================
    # CONVENIENCE METHODS (shortcuts to collection operations)
    # =========================================================================
    
    def add(
        self,
        collection_name: str,
        id: str,
        vector: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a vector to a collection.
        
        Shortcut for: db.get_collection(name).add(...)
        """
        return self.get_collection(collection_name).add(id, vector, metadata)
    
    def search(
        self,
        collection_name: str,
        query: Any,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Search a collection.
        
        Shortcut for: db.get_collection(name).search(...)
        """
        return self.get_collection(collection_name).search(
            query, k=k, filter=filter, **kwargs
        )
    
    def delete(self, collection_name: str, id: str) -> bool:
        """
        Delete a vector from a collection.
        
        Shortcut for: db.get_collection(name).delete(...)
        """
        return self.get_collection(collection_name).delete(id)
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save database to disk.
        
        Args:
            path: Directory path (uses data_dir if not specified)
            
        Raises:
            StorageError: If save fails
        """
        save_dir = Path(path) if path else self._data_dir
        
        if not save_dir:
            raise StorageError("No save path specified and no data_dir configured")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with self._lock:
                # Save manifest
                manifest = {
                    "version": self.VERSION,
                    "created_at": self._created_at,
                    "updated_at": self._updated_at,
                    "collections": list(self._collections.keys()),
                }
                
                manifest_path = save_dir / self.MANIFEST_FILE
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                # Save each collection
                for name, collection in self._collections.items():
                    collection_path = save_dir / f"{name}.collection"
                    collection_data = collection.to_dict()
                    
                    with open(collection_path, 'wb') as f:
                        pickle.dump(collection_data, f)
                
                logger.info(
                    f"Saved database to {save_dir} "
                    f"({len(self._collections)} collections)"
                )
        
        except Exception as e:
            raise StorageError(f"Failed to save database: {e}")
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "VectorDB":
        """
        Load database from disk.
        
        Args:
            path: Directory path containing saved database
            **kwargs: Additional arguments for VectorDB constructor
            
        Returns:
            Loaded VectorDB instance
            
        Raises:
            StorageError: If load fails
        """
        load_dir = Path(path)
        
        if not load_dir.exists():
            raise StorageError(f"Database path does not exist: {path}")
        
        manifest_path = load_dir / cls.MANIFEST_FILE
        
        if not manifest_path.exists():
            raise StorageError(f"Manifest file not found in {path}")
        
        try:
            # Load manifest
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Create database
            db = cls(data_dir=path, **kwargs)
            db._created_at = manifest.get("created_at", time.time())
            db._updated_at = manifest.get("updated_at", time.time())
            
            # Load collections
            for name in manifest.get("collections", []):
                collection_path = load_dir / f"{name}.collection"
                
                if collection_path.exists():
                    with open(collection_path, 'rb') as f:
                        collection_data = pickle.load(f)
                    
                    collection = Collection.from_dict(collection_data)
                    db._collections[name] = collection
                    
                    logger.debug(f"Loaded collection '{name}'")
            
            logger.info(
                f"Loaded database from {path} "
                f"({len(db._collections)} collections)"
            )
            
            return db
        
        except Exception as e:
            raise StorageError(f"Failed to load database: {e}")
    
    def _start_auto_save(self) -> None:
        """Start auto-save background thread."""
        def auto_save_loop():
            while self._running:
                time.sleep(self._auto_save_interval)
                if self._running:
                    try:
                        self.save()
                        logger.debug("Auto-save completed")
                    except Exception as e:
                        logger.error(f"Auto-save failed: {e}")
        
        self._auto_save_thread = threading.Thread(
            target=auto_save_loop,
            daemon=True,
        )
        self._auto_save_thread.start()
        logger.debug("Auto-save thread started")
    
    # =========================================================================
    # DATABASE INFO
    # =========================================================================
    
    def info(self) -> Dict[str, Any]:
        """
        Get database information.
        
        Returns:
            Dictionary with database info
        """
        with self._lock:
            total_vectors = sum(
                len(c) for c in self._collections.values()
            )
            
            total_memory = sum(
                c.stats().memory_usage_bytes 
                for c in self._collections.values()
            )
            
            return {
                "version": self.VERSION,
                "data_dir": str(self._data_dir) if self._data_dir else None,
                "collection_count": len(self._collections),
                "total_vectors": total_vectors,
                "total_memory_mb": round(total_memory / (1024 * 1024), 2),
                "created_at": self._created_at,
                "updated_at": self._updated_at,
                "auto_save": self._auto_save,
                "collections": {
                    name: collection.stats().to_dict()
                    for name, collection in self._collections.items()
                },
            }
    
    def __repr__(self) -> str:
        return (
            f"VectorDB(collections={len(self._collections)}, "
            f"data_dir='{self._data_dir}')"
        )
    
    # =========================================================================
    # CONTEXT MANAGER & CLEANUP
    # =========================================================================
    
    def close(self) -> None:
        """
        Close the database and cleanup resources.
        
        Saves data if persistent and auto-save is enabled.
        """
        self._running = False
        
        if self._auto_save_thread:
            self._auto_save_thread.join(timeout=5)
        
        if self._data_dir and self._auto_save:
            try:
                self.save()
            except Exception as e:
                logger.error(f"Failed to save on close: {e}")
        
        logger.info("VectorDB closed")
    
    def __enter__(self) -> "VectorDB":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
    
    # =========================================================================
    # COLLECTION ACCESS SHORTCUTS
    # =========================================================================
    
    def __getitem__(self, name: str) -> Collection:
        """
        Get collection by name using indexing.
        
        Example:
            >>> collection = db["my_collection"]
        """
        return self.get_collection(name)
    
    def __contains__(self, name: str) -> bool:
        """
        Check if collection exists.
        
        Example:
            >>> if "my_collection" in db:
            ...     pass
        """
        return self.has_collection(name)
    
    def __iter__(self):
        """Iterate over collection names."""
        return iter(self._collections.keys())
    
    def __len__(self) -> int:
        """Number of collections."""
        return len(self._collections)