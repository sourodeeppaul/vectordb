"""
Example usage of VectorDB storage backends.
"""

import numpy as np
import time
import tempfile
import shutil
from pathlib import Path

from vectordb.storage import MemoryStorage, DiskStorage


def main():
    print("=" * 60)
    print("Storage Backend Examples")
    print("=" * 60)
    
    dimension = 128
    n_vectors = 10000
    
    # =========================================================================
    # Memory Storage
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. Memory Storage")
    print("=" * 60)
    
    memory_storage = MemoryStorage(dimension=dimension)
    
    # Add vectors
    print("\nAdding vectors...")
    start = time.time()
    
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    ids = [f"vec{i}" for i in range(n_vectors)]
    metadata = [{"index": i, "category": f"cat_{i % 10}"} for i in range(n_vectors)]
    
    memory_storage.put_batch(ids, vectors, metadata)
    
    add_time = time.time() - start
    print(f"Added {n_vectors} vectors in {add_time:.3f}s")
    
    # Random access
    print("\nRandom access benchmark...")
    n_reads = 1000
    
    start = time.time()
    for _ in range(n_reads):
        i = np.random.randint(n_vectors)
        _ = memory_storage.get(f"vec{i}")
    read_time = time.time() - start
    
    print(f"{n_reads} random reads in {read_time:.3f}s ({n_reads/read_time:.0f} reads/s)")
    
    # Stats
    stats = memory_storage.stats()
    print(f"\nMemory Storage Stats:")
    print(f"  Vectors: {stats.vector_count}")
    print(f"  Memory: {stats.memory_bytes / 1024 / 1024:.2f} MB")
    
    # =========================================================================
    # Disk Storage
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. Disk Storage")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        disk_storage = DiskStorage(temp_dir, dimension=dimension)
        
        # Add vectors
        print("\nAdding vectors...")
        start = time.time()
        
        batch_size = 1000
        for batch_start in range(0, n_vectors, batch_size):
            batch_end = min(batch_start + batch_size, n_vectors)
            batch_ids = ids[batch_start:batch_end]
            batch_vectors = vectors[batch_start:batch_end]
            batch_metadata = metadata[batch_start:batch_end]
            
            disk_storage.put_batch(batch_ids, batch_vectors, batch_metadata)
        
        disk_storage.flush()
        add_time = time.time() - start
        print(f"Added {n_vectors} vectors in {add_time:.3f}s")
        
        # Random access
        print("\nRandom access benchmark...")
        
        start = time.time()
        for _ in range(n_reads):
            i = np.random.randint(n_vectors)
            _ = disk_storage.get(f"vec{i}")
        read_time = time.time() - start
        
        print(f"{n_reads} random reads in {read_time:.3f}s ({n_reads/read_time:.0f} reads/s)")
        
        # Stats
        stats = disk_storage.stats()
        print(f"\nDisk Storage Stats:")
        print(f"  Vectors: {stats.vector_count}")
        print(f"  Disk: {stats.disk_bytes / 1024 / 1024:.2f} MB")
        print(f"  Memory: {stats.memory_bytes / 1024 / 1024:.2f} MB")
        print(f"  Cache hit rate: {stats.cache_hits / (stats.cache_hits + stats.cache_misses + 1):.1%}")
        
        # Persistence test
        print("\n--- Persistence Test ---")
        disk_storage.close()
        
        # Reopen and verify
        disk_storage2 = DiskStorage(temp_dir, dimension=dimension)
        
        # Check a few vectors
        for i in [0, n_vectors // 2, n_vectors - 1]:
            result = disk_storage2.get(f"vec{i}")
            if result:
                vec, meta = result
                assert np.allclose(vec, vectors[i])
                assert meta["index"] == i
        
        print("Persistence verified!")
        
        # Compaction test
        print("\n--- Compaction Test ---")
        
        # Delete half the vectors
        for i in range(0, n_vectors, 2):
            disk_storage2.delete(f"vec{i}")
        
        print(f"Deleted {n_vectors // 2} vectors")
        
        stats_before = disk_storage2.stats()
        print(f"Before compaction: {stats_before.disk_bytes / 1024 / 1024:.2f} MB on disk")
        
        disk_storage2.compact()
        
        stats_after = disk_storage2.stats()
        print(f"After compaction: {stats_after.disk_bytes / 1024 / 1024:.2f} MB on disk")
        print(f"Space saved: {(stats_before.disk_bytes - stats_after.disk_bytes) / 1024 / 1024:.2f} MB")
        
        disk_storage2.close()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # =========================================================================
    # Comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. Storage Comparison")
    print("=" * 60)
    
    print("\n| Operation      | Memory Storage | Disk Storage |")
    print("|----------------|----------------|--------------|")
    print(f"| Add {n_vectors:,} vecs | Fast (in-mem)  | + Disk I/O   |")
    print(f"| Random Read    | O(1)           | O(1) + cache |")
    print(f"| Persistence    | No             | Yes          |")
    print(f"| Memory Usage   | ~Full dataset  | Cache only   |")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()