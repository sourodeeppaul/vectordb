"""
Image Similarity Search Example for VectorDB.

This example demonstrates how to build an image similarity search system
using VectorDB with image embeddings from a pre-trained model.

Features demonstrated:
- Image embedding extraction
- Storing image vectors with metadata
- Similar image search
- Batch processing of image directories

Requirements:
    pip install torch torchvision pillow
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
import hashlib

# Check for required packages
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Note: torch/torchvision not installed. Using mock embeddings.")
    print("Install with: pip install torch torchvision pillow")

from vectordb import VectorDatabase


class ImageSimilaritySearch:
    """
    An image similarity search engine built on VectorDB.
    
    Uses a pre-trained CNN to extract image embeddings and
    enables similarity-based image retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        db_path: Optional[str] = None,
        collection_name: str = "images",
        device: Optional[str] = None
    ):
        """
        Initialize the image similarity search engine.
        
        Args:
            model_name: Pre-trained model to use (resnet50, resnet18, vgg16)
            db_path: Path for persistent storage (None for in-memory)
            collection_name: Name of the image collection
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.collection_name = collection_name
        
        # Initialize model
        if HAS_TORCH:
            self._init_model(model_name, device)
        else:
            self.model = None
            self.dimension = 2048  # Mock dimension
            self.device = "cpu"
        
        # Initialize VectorDB
        self.db = VectorDatabase(storage_path=db_path)
        
        # Create or get collection
        if self.db.has_collection(collection_name):
            self.collection = self.db.get_collection(collection_name)
            print(f"Loaded existing collection with {self.collection.count()} images")
        else:
            self.collection = self.db.create_collection(
                name=collection_name,
                dimension=self.dimension,
                metric="cosine",
                index_type="hnsw",
                index_params={"M": 16, "ef_construction": 100}
            )
            print(f"Created new collection: {collection_name}")
    
    def _init_model(self, model_name: str, device: Optional[str]):
        """Initialize the image embedding model."""
        # Select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading {model_name} on {self.device}...")
        
        # Load pre-trained model
        if model_name == "resnet50":
            base_model = models.resnet50(pretrained=True)
            self.dimension = 2048
        elif model_name == "resnet18":
            base_model = models.resnet18(pretrained=True)
            self.dimension = 512
        elif model_name == "vgg16":
            base_model = models.vgg16(pretrained=True)
            self.dimension = 4096
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove classification layer to get embeddings
        if "resnet" in model_name:
            self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        else:
            # For VGG, use features + avgpool
            self.model = torch.nn.Sequential(
                base_model.features,
                base_model.avgpool,
                torch.nn.Flatten()
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_image(self, image_path: Union[str, Path]) -> "Image.Image":
        """Load and preprocess an image."""
        image = Image.open(image_path).convert("RGB")
        return image
    
    def _embed_images(self, images: List["Image.Image"]) -> np.ndarray:
        """Generate embeddings for a list of images."""
        if not HAS_TORCH or self.model is None:
            # Mock embeddings
            return np.random.randn(len(images), self.dimension).astype(np.float32)
        
        # Preprocess images
        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(batch)
            embeddings = embeddings.squeeze()
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings.cpu().numpy()
    
    def _get_image_hash(self, image_path: Union[str, Path]) -> str:
        """Generate a hash ID for an image file."""
        path_str = str(Path(image_path).absolute())
        return hashlib.md5(path_str.encode()).hexdigest()[:16]
    
    def add_image(
        self,
        image_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        image_id: Optional[str] = None
    ) -> str:
        """
        Add a single image to the index.
        
        Args:
            image_path: Path to the image file
            metadata: Optional metadata dictionary
            image_id: Optional custom ID
            
        Returns:
            Image ID
        """
        image_path = Path(image_path)
        
        # Load and embed image
        image = self._load_image(image_path)
        embedding = self._embed_images([image])[0]
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "path": str(image_path.absolute()),
            "filename": image_path.name,
            "extension": image_path.suffix.lower(),
            "size_bytes": image_path.stat().st_size,
            "width": image.width,
            "height": image.height,
        })
        
        # Generate ID if not provided
        if image_id is None:
            image_id = self._get_image_hash(image_path)
        
        # Add to collection
        self.collection.add(
            embedding.reshape(1, -1),
            ids=[image_id],
            metadata=[metadata]
        )
        
        return image_id
    
    def add_directory(
        self,
        directory: Union[str, Path],
        extensions: List[str] = None,
        recursive: bool = True,
        batch_size: int = 32
    ) -> List[str]:
        """
        Add all images from a directory.
        
        Args:
            directory: Path to directory
            extensions: List of file extensions to include
            recursive: Whether to search subdirectories
            batch_size: Batch size for processing
            
        Returns:
            List of added image IDs
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        
        directory = Path(directory)
        
        # Find all image files
        image_paths = []
        pattern = "**/*" if recursive else "*"
        for ext in extensions:
            image_paths.extend(directory.glob(f"{pattern}{ext}"))
            image_paths.extend(directory.glob(f"{pattern}{ext.upper()}"))
        
        image_paths = list(set(image_paths))
        print(f"Found {len(image_paths)} images in {directory}")
        
        if not image_paths:
            return []
        
        all_ids = []
        start_time = time.time()
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load images
            images = []
            valid_paths = []
            for path in batch_paths:
                try:
                    images.append(self._load_image(path))
                    valid_paths.append(path)
                except Exception as e:
                    print(f"  Warning: Could not load {path}: {e}")
            
            if not images:
                continue
            
            # Generate embeddings
            embeddings = self._embed_images(images)
            
            # Prepare metadata and IDs
            metadata_list = []
            ids = []
            for path, img in zip(valid_paths, images):
                ids.append(self._get_image_hash(path))
                metadata_list.append({
                    "path": str(path.absolute()),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "width": img.width,
                    "height": img.height,
                })
            
            # Add to collection
            self.collection.add(embeddings, ids=ids, metadata=metadata_list)
            all_ids.extend(ids)
            
            if (i + batch_size) % 100 == 0 or i + batch_size >= len(image_paths):
                print(f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
        
        elapsed = time.time() - start_time
        print(f"Added {len(all_ids)} images in {elapsed:.2f}s")
        
        return all_ids
    
    def search(
        self,
        query_image: Union[str, Path, "Image.Image"],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images.
        
        Args:
            query_image: Query image (path or PIL Image)
            k: Number of results
            filter: Optional metadata filter
            
        Returns:
            List of similar images with metadata and scores
        """
        # Load query image if path
        if isinstance(query_image, (str, Path)):
            query_image = self._load_image(query_image)
        
        # Generate embedding
        query_embedding = self._embed_images([query_image])[0]
        
        # Search
        results = self.collection.search(
            query_embedding,
            k=k,
            filter=filter
        )
        
        # Format results
        formatted = []
        for r in results:
            formatted.append({
                "id": r["id"],
                "path": r["metadata"].get("path", ""),
                "filename": r["metadata"].get("filename", ""),
                "similarity": 1 - r["distance"],  # Convert distance to similarity
                "metadata": r["metadata"]
            })
        
        return formatted
    
    def search_by_id(
        self,
        image_id: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find images similar to an indexed image.
        
        Args:
            image_id: ID of the query image
            k: Number of results
            filter: Optional metadata filter
            
        Returns:
            List of similar images
        """
        # Get the image vector
        result = self.collection.get(image_id)
        query_vector = result["vector"]
        
        # Search (k+1 to exclude the query image itself)
        results = self.collection.search(
            query_vector,
            k=k + 1,
            filter=filter
        )
        
        # Filter out query image and format
        formatted = []
        for r in results:
            if r["id"] != image_id:
                formatted.append({
                    "id": r["id"],
                    "path": r["metadata"].get("path", ""),
                    "filename": r["metadata"].get("filename", ""),
                    "similarity": 1 - r["distance"],
                    "metadata": r["metadata"]
                })
        
        return formatted[:k]
    
    def get_duplicates(
        self,
        similarity_threshold: float = 0.98
    ) -> List[List[Dict[str, Any]]]:
        """
        Find duplicate or near-duplicate images.
        
        Args:
            similarity_threshold: Minimum similarity to consider duplicate
            
        Returns:
            List of duplicate groups
        """
        # Get all images
        all_ids = []  # Would need to implement iteration over collection
        
        # For each image, find highly similar ones
        # This is a simplified approach - production would use clustering
        
        print("Duplicate detection requires iterating over all images.")
        print("This is a placeholder for the full implementation.")
        
        return []
    
    def count(self) -> int:
        """Get total number of indexed images."""
        return self.collection.count()
    
    def close(self):
        """Close the database connection."""
        self.db.close()


def create_mock_images(output_dir: Path, n_images: int = 20):
    """Create mock images for demonstration."""
    if not HAS_TORCH:
        print("Cannot create mock images without PIL")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (255, 128, 0)
    ]
    
    for i in range(n_images):
        # Create simple colored image with some variation
        color = colors[i % len(colors)]
        img = Image.new("RGB", (224, 224), color)
        
        # Add some variation
        pixels = img.load()
        for x in range(0, 224, 20):
            for y in range(0, 224, 20):
                variation = ((i * 17 + x + y) % 50)
                new_color = tuple(min(255, c + variation) for c in color)
                for dx in range(min(10, 224-x)):
                    for dy in range(min(10, 224-y)):
                        pixels[x+dx, y+dy] = new_color
        
        img.save(output_dir / f"image_{i:03d}.jpg")
    
    print(f"Created {n_images} mock images in {output_dir}")


def main():
    """Run the image similarity search example."""
    print("=" * 60)
    print("Image Similarity Search Example")
    print("=" * 60)
    
    # Create temp directory with mock images
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    image_dir = temp_dir / "images"
    
    if HAS_TORCH:
        create_mock_images(image_dir, n_images=20)
    else:
        # Create empty directory and placeholder
        image_dir.mkdir(parents=True, exist_ok=True)
        print("\nNote: Running in mock mode without actual images")
    
    # Initialize search engine
    print("\nInitializing image similarity search...")
    engine = ImageSimilaritySearch(
        model_name="resnet18",  # Smaller model for demo
        db_path=None  # In-memory
    )
    
    # Index images
    if HAS_TORCH and list(image_dir.glob("*.jpg")):
        print("\nIndexing images...")
        ids = engine.add_directory(image_dir)
        print(f"Total images indexed: {engine.count()}")
        
        # Demo searches
        print("\n" + "=" * 60)
        print("Search Examples")
        print("=" * 60)
        
        # Search by image file
        query_image = list(image_dir.glob("*.jpg"))[0]
        print(f"\n1. Search similar to: {query_image.name}")
        print("-" * 40)
        results = engine.search(query_image, k=5)
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['filename']}")
        
        # Search by ID
        if ids:
            print(f"\n2. Find images similar to ID: {ids[0]}")
            print("-" * 40)
            results = engine.search_by_id(ids[0], k=5)
            for r in results:
                print(f"  [{r['similarity']:.3f}] {r['filename']}")
        
        # Search with filter (if we had more metadata)
        print("\n3. Search with metadata filter")
        print("-" * 40)
        results = engine.search(
            query_image,
            k=5,
            filter={"extension": ".jpg"}
        )
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['filename']} ({r['metadata'].get('width')}x{r['metadata'].get('height')})")
    else:
        print("\nNo images available for demo.")
        print("To run the full demo, install torch/torchvision/pillow")
    
    # Clean up
    engine.close()
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    main()