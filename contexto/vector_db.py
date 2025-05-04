"""Vector database implementation using Qdrant."""

import os
from typing import List, Optional, Tuple, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class VectorDB:
    """Vector database for storing and retrieving word embeddings."""

    def __init__(self, collection_name: str = "words", path: str = "./data/vector_index",
                 use_docker: bool = False, url: str = None, batch_size: int = None):
        """Initialize the vector database.

        Args:
            collection_name: Name of the Qdrant collection
            path: Path to store the Qdrant database
            use_docker: Whether to use Docker Qdrant instance instead of local mode
            url: URL of the Qdrant server if using Docker (e.g., "http://localhost:6333")
            batch_size: Custom batch size for embedding generation (overrides auto-detection)
        """
        self.collection_name = collection_name
        self.path = path
        self.model = None
        self.use_docker = use_docker

        # Set custom batch size if provided
        if batch_size is not None:
            self.custom_batch_size = batch_size
            print(f"Using custom batch size: {batch_size}")

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Initialize Qdrant client
        if use_docker and url:
            print(f"Using Qdrant server at {url}")
            self.client = QdrantClient(url=url)
        else:
            print(f"Using local Qdrant instance at {path}")
            self.client = QdrantClient(path=path)

        # Track if we've shown the large collection warning
        self._large_collection_warning_shown = False

    def load_model(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Load the embedding model.

        Args:
            model_name: Name of the sentence transformer model
        """
        # Check if GPU is available
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"Using GPU for sentence transformer model")
            else:
                print("No GPU detected, using CPU for sentence transformer model")
        except ImportError:
            print("PyTorch not found, using CPU for sentence transformer model")
            device = "cpu"

        # Load the model on the appropriate device
        self.model = SentenceTransformer(model_name, device=device)

    def get_embedding(self, word: str) -> np.ndarray:
        """Get the embedding vector for a word.

        Args:
            word: Word to embed

        Returns:
            Embedding vector
        """
        if self.model is None:
            self.load_model()

        return self.model.encode(word)

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if the collection exists, False otherwise
        """
        if collection_name is None:
            collection_name = self.collection_name

        collections = self.client.get_collections().collections
        return any(collection.name == collection_name for collection in collections)

    def build_index(self, word_list_path: str) -> int:
        """Build the vector index from a word list.

        Args:
            word_list_path: Path to text file with words (one per line)

        Returns:
            Number of words indexed
        """
        # Load the model if not already loaded
        if self.model is None:
            print("Loading sentence transformer model...")
            self.load_model()
            print(f"Model loaded: {self.model.get_sentence_embedding_dimension()} dimensions")

        # Read words from file
        print(f"Reading words from {word_list_path}...")
        with open(word_list_path, "r") as f:
            words = [line.strip() for line in f if line.strip()]
        print(f"Read {len(words)} words")

        # Check if we should warn about large collections
        if len(words) > 20000 and not self.use_docker and not self._large_collection_warning_shown:
            print("\n" + "!" * 80)
            print("WARNING: You are attempting to index a large collection with more than 20,000 words.")
            print("Qdrant's local mode is not optimized for collections of this size.")
            print("For better performance, consider:")
            print("1. Using Qdrant in Docker: docker run -p 6333:6333 qdrant/qdrant")
            print("2. Then initialize VectorDB with: VectorDB(use_docker=True, url='http://localhost:6333')")
            print("!" * 80 + "\n")

            # Only show this warning once
            self._large_collection_warning_shown = True

            # Ask user if they want to continue
            if not os.environ.get("QDRANT_FORCE_LOCAL", ""):
                response = input("Do you want to continue with local mode anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Aborting. Please restart with Docker mode.")
                    return 0

        # Create collection if it doesn't exist
        if not self.collection_exists():
            print(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.get_embedding("test").shape[0],
                    distance=models.Distance.COSINE
                )
            )
            print("Collection created successfully")
        else:
            print(f"Collection '{self.collection_name}' already exists")

        # Prepare points for batch upload
        print("Generating embeddings for words...")
        points = []

        # Check if we can use batch processing (more efficient on GPU)
        try:
            import torch
            has_gpu = torch.cuda.is_available()
            if has_gpu:
                # Get GPU memory info
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
                print(f"GPU memory: {gpu_memory:.2f} GB")

                # Adjust batch size based on available GPU memory
                if gpu_memory > 16:  # High-end GPU (like L4)
                    suggested_batch_size = 128
                elif gpu_memory > 8:  # Mid-range GPU
                    suggested_batch_size = 64
                else:  # Low-end GPU
                    suggested_batch_size = 32

                print(f"Suggested batch size for your GPU: {suggested_batch_size}")
            else:
                suggested_batch_size = 32
        except ImportError:
            has_gpu = False
            suggested_batch_size = 32

        # Batch size for embedding generation - can be overridden by custom_batch_size
        embedding_batch_size = getattr(self, 'custom_batch_size', suggested_batch_size)

        # Process words in batches for better GPU utilization
        for i in tqdm(range(0, len(words), embedding_batch_size), desc="Embedding batches", unit="batch"):
            batch_words = words[i:i+embedding_batch_size]

            # Generate embeddings for the batch
            batch_embeddings = self.model.encode(batch_words, show_progress_bar=False)

            # Create points for each word in the batch
            for j, (word, embedding) in enumerate(zip(batch_words, batch_embeddings)):
                points.append(
                    models.PointStruct(
                        id=i+j,  # Ensure unique ID for each word
                        vector=embedding.tolist(),
                        payload={"word": word}
                    )
                )

            # Log progress for large datasets
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i+len(batch_words)} words out of {len(words)} ({(i+len(batch_words))/len(words)*100:.1f}%)")

        # Upload points in batches
        print("Uploading embeddings to Qdrant...")
        upload_batch_size = 100
        total_batches = (len(points) + upload_batch_size - 1) // upload_batch_size

        # Process upload in batches
        for i in tqdm(range(0, len(points), upload_batch_size), desc="Uploading batches", unit="batch", total=total_batches):
            batch = points[i:i+upload_batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

            # Track total points for progress reporting
            if not hasattr(self, '_total_points_count'):
                self._total_points_count = len(points)

            # Log progress for large datasets
            if i % 1000 == 0 and i > 0:
                print(f"Uploaded {i+len(batch)} points out of {self._total_points_count} ({(i+len(batch))/self._total_points_count*100:.1f}%)")

            # Free memory for very large datasets by clearing processed points
            if len(points) > 10000 and i % 5000 == 0 and i > 0:
                # Clear already processed points to free memory
                points_to_keep = points[i+upload_batch_size:]
                del points
                points = points_to_keep
                print(f"Memory optimization: Cleared processed points. {len(points)} points remaining to process.")

        print(f"Successfully indexed {len(words)} words")
        return len(words)

    def search(self, query: Union[str, np.ndarray], limit: int = 10) -> List[Tuple[str, float]]:
        """Search for semantically similar words.

        Args:
            query: Query word or phrase, or a pre-computed embedding vector
            limit: Maximum number of results

        Returns:
            List of (word, score) tuples
        """
        # Get query embedding if it's a string
        if isinstance(query, str):
            query_embedding = self.get_embedding(query)
        else:
            # Assume it's already an embedding vector
            query_embedding = query

        # Search for similar vectors using query_points
        try:
            # Convert to list if it's a numpy array, otherwise use as is
            query_vector = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            )

            # Extract words and scores
            results = []

            # Debug information
            print(f"Debug - search_result type: {type(search_result)}")

            # Handle different response formats
            if hasattr(search_result, 'points'):
                # Handle QueryResponse object
                for hit in search_result.points:
                    if hit.payload and "word" in hit.payload:
                        word = hit.payload.get("word")
                        score = 1.0 - hit.score  # Convert cosine distance to similarity
                        results.append((word, score))
            elif isinstance(search_result, list):
                # Handle list of ScoredPoint objects
                for hit in search_result:
                    if hasattr(hit, 'payload') and hit.payload and "word" in hit.payload:
                        word = hit.payload.get("word")
                        score = 1.0 - hit.score  # Convert cosine distance to similarity
                        results.append((word, score))
                    elif isinstance(hit, dict) and "payload" in hit and "word" in hit["payload"]:
                        word = hit["payload"]["word"]
                        score = 1.0 - hit["score"]
                        results.append((word, score))

            # If we couldn't extract results, print debug info
            if not results:
                print(f"Warning: Could not extract results from search_result: {search_result}")

                # Try a fallback approach for unknown response formats
                if isinstance(search_result, list) and search_result:
                    print(f"First result item: {search_result[0]}")
                    print(f"First result item type: {type(search_result[0])}")
                    if hasattr(search_result[0], '__dict__'):
                        print(f"First result item attributes: {search_result[0].__dict__}")

            return results

        except Exception as e:
            print(f"Error during search: {e}")
            print(f"Query embedding shape: {query_embedding.shape}")
            print(f"Query embedding type: {type(query_embedding)}")
            # Return empty results on error
            return []

    def get_distance(self, word1: str, word2: str) -> float:
        """Calculate the distance between two words.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Distance between the words
        """
        # Get embeddings
        embedding1 = self.get_embedding(word1)
        embedding2 = self.get_embedding(word2)

        # Calculate cosine distance
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        distance = 1.0 - similarity

        return distance

    def get_all_words(self) -> List[str]:
        """Get all words in the vector database.

        Returns:
            List of all words
        """
        try:
            # Scroll through all points in the collection
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Get up to 1000 points at a time
                with_payload=True,
                with_vectors=False
            )

            # Extract words from payload
            words = []
            if hasattr(scroll_result, 'points'):
                # Handle QueryResponse object
                for point in scroll_result.points:
                    if point.payload and "word" in point.payload:
                        words.append(point.payload["word"])
            elif isinstance(scroll_result, tuple) and len(scroll_result) >= 1:
                # Handle tuple response (points, next_page_offset)
                for point in scroll_result[0]:
                    if hasattr(point, 'payload') and point.payload and "word" in point.payload:
                        words.append(point.payload["word"])
                    elif isinstance(point, dict) and "payload" in point and "word" in point["payload"]:
                        words.append(point["payload"]["word"])

            # If we couldn't extract words, use fallback
            if not words:
                print("Warning: Could not extract words from vector database. Using fallback words.")
                words = [
                    "apple", "banana", "cherry", "date", "elderberry",
                    "fig", "grape", "honeydew", "kiwi", "lemon",
                    "mango", "nectarine", "orange", "papaya", "quince",
                    "raspberry", "strawberry", "tangerine", "watermelon"
                ]

            return words

        except Exception as e:
            print(f"Error getting all words: {e}")
            # Return some fallback words
            return [
                "apple", "banana", "cherry", "date", "elderberry",
                "fig", "grape", "honeydew", "kiwi", "lemon",
                "mango", "nectarine", "orange", "papaya", "quince",
                "raspberry", "strawberry", "tangerine", "watermelon"
            ]
