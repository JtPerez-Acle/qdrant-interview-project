"""Vector database implementation using Qdrant for the curated word list approach."""

import os
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorDB:
    """Vector database for storing and retrieving word embeddings using a curated word list."""

    def __init__(self, collection_name: str = "words_curated", path: str = "./data/vector_index",
                 use_docker: bool = False, url: str = None, batch_size: int = 64,
                 max_words: int = 20000):
        """Initialize the vector database.

        Args:
            collection_name: Name of the Qdrant collection (default: words_curated)
            path: Path to store the Qdrant database
            use_docker: Whether to use Docker Qdrant instance instead of local mode
            url: URL of the Qdrant server if using Docker (e.g., "http://localhost:6333")
            batch_size: Batch size for embedding generation
            max_words: Maximum number of words to use from the collection
        """
        self.collection_name = collection_name
        self.path = path
        self.model = None
        self.use_docker = use_docker
        self.batch_size = batch_size
        self.max_words = max_words
        self.client = None

        # Create directory if it doesn't exist
        os.makedirs(self.path, exist_ok=True)

        # Initialize the client
        self.initialize_client()

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.close()

    def close(self):
        """Close the Qdrant client and clean up resources."""
        if hasattr(self, 'client') and self.client is not None:
            try:
                # Close the client if it has a close method
                if hasattr(self.client, 'close'):
                    self.client.close()
                self.client = None
                logger.info("Qdrant client closed successfully")
            except Exception as e:
                logger.error(f"Error closing Qdrant client: {e}")

    def initialize_client(self, force_new=False):
        """Initialize the Qdrant client.

        Args:
            force_new: If True, remove the existing storage directory and create a new one
        """
        # If force_new is True, remove the entire storage directory
        if force_new and os.path.exists(self.path):
            logger.warning(f"Removing existing storage directory at {self.path}")
            try:
                import shutil
                shutil.rmtree(self.path)
                logger.info("Storage directory removed successfully")
            except Exception as e:
                logger.error(f"Error removing storage directory: {e}")

        # Create directory if it doesn't exist
        os.makedirs(self.path, exist_ok=True)

        # Check for lock files and remove them if they exist
        lock_file = os.path.join(self.path, "storage.lock")
        if os.path.exists(lock_file):
            logger.warning(f"Found lock file at {lock_file}. Removing it to avoid resource conflicts.")
            try:
                os.remove(lock_file)
                logger.info("Lock file removed successfully.")
            except Exception as e:
                logger.error(f"Error removing lock file: {e}")

        # Try to find and kill any processes using the storage directory
        try:
            self._kill_qdrant_processes()
        except Exception as e:
            logger.error(f"Error killing Qdrant processes: {e}")

        # Initialize Qdrant client
        try:
            if self.use_docker and hasattr(self, 'url') and self.url:
                logger.info(f"Using Qdrant server at {self.url}")
                self.client = QdrantClient(url=self.url)
            else:
                logger.info(f"Using local Qdrant instance at {self.path}")
                self.client = QdrantClient(path=self.path)
            logger.info("Qdrant client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            self.client = None

    def _kill_qdrant_processes(self):
        """Try to kill any processes that might be using the Qdrant storage directory."""
        try:
            import psutil
            import signal

            # Get the absolute path to the storage directory
            abs_path = os.path.abspath(self.path)

            # Find all Python processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Check if it's a Python process
                    if proc.info['name'] == 'python' or proc.info['name'] == 'python3':
                        # Check if it's using our storage directory
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if 'qdrant' in cmdline.lower() and abs_path in cmdline:
                            # It's a Qdrant process using our storage directory
                            logger.warning(f"Found Qdrant process (PID: {proc.info['pid']}) using our storage directory. Killing it.")
                            os.kill(proc.info['pid'], signal.SIGTERM)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except ImportError:
            logger.warning("psutil not installed. Cannot check for running Qdrant processes.")
            pass

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
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info("Using GPU for sentence transformer model")
            else:
                logger.info("No GPU detected, using CPU for sentence transformer model")
        except ImportError:
            logger.info("PyTorch not found, using CPU for sentence transformer model")
            device = "cpu"

        # Load the model on the appropriate device
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model loaded with {self.model.get_sentence_embedding_dimension()} dimensions")

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

        # Check if client is initialized
        if self.client is None:
            logger.error("Qdrant client is not initialized")
            return False

        try:
            collections = self.client.get_collections().collections
            return any(collection.name == collection_name for collection in collections)
        except Exception as e:
            logger.error(f"Error checking if collection exists: {e}")
            return False

    def build_index(self, word_list_path: str) -> int:
        """Build the vector index from a curated word list.

        Args:
            word_list_path: Path to text file with words (one per line)

        Returns:
            Number of words indexed
        """
        # Load the model if not already loaded
        if self.model is None:
            logger.info("Loading sentence transformer model...")
            self.load_model()

        # Read words from file
        logger.info(f"Reading words from {word_list_path}...")
        with open(word_list_path, "r") as f:
            words = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(words)} words")

        # Check if client is initialized
        if self.client is None:
            logger.error("Qdrant client is not initialized. Trying to reinitialize...")
            self.initialize_client()

        # Check again if client is initialized
        if self.client is None:
            logger.error("Failed to initialize Qdrant client. Cannot build index.")
            return 0

        # Create collection if it doesn't exist
        if not self.collection_exists():
            logger.info(f"Creating collection '{self.collection_name}'...")
            # Get embedding dimension
            test_embedding = self.get_embedding("test")
            # Handle both numpy arrays and lists
            embedding_size = test_embedding.shape[0] if hasattr(test_embedding, 'shape') else len(test_embedding)

            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            except Exception as e:
                logger.error(f"Error creating collection: {e}")
                return 0
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

        # Prepare points for batch upload
        logger.info("Generating embeddings for words...")
        points = []

        # Process words in batches for better GPU utilization
        for i in tqdm(range(0, len(words), self.batch_size), desc="Embedding batches", unit="batch"):
            batch_words = words[i:i+self.batch_size]

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

        # Upload points in batches
        logger.info("Uploading embeddings to Qdrant...")
        upload_batch_size = 100
        total_batches = (len(points) + upload_batch_size - 1) // upload_batch_size

        # Process upload in batches
        for i in tqdm(range(0, len(points), upload_batch_size), desc="Uploading batches", unit="batch", total=total_batches):
            batch = points[i:i+upload_batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            except Exception as e:
                logger.error(f"Error uploading batch {i//upload_batch_size + 1}/{total_batches}: {e}")
                # Continue with next batch

        logger.info(f"Successfully indexed {len(words)} words")
        return len(words)

    def search(self, query: Union[str, np.ndarray], limit: int = 10) -> List[Tuple[str, float]]:
        """Search for semantically similar words.

        Args:
            query: Query word or phrase, or a pre-computed embedding vector
            limit: Maximum number of results

        Returns:
            List of (word, score) tuples
        """
        # Check if client is initialized
        if self.client is None:
            logger.error("Qdrant client is not initialized")
            return []

        # Get query embedding if it's a string
        if isinstance(query, str):
            query_embedding = self.get_embedding(query)
        else:
            # Assume it's already an embedding vector
            query_embedding = query

        # Search for similar vectors
        try:
            # Convert to list if it's a numpy array, otherwise use as is
            query_vector = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding

            # Check if collection exists
            if not self.collection_exists():
                logger.error(f"Collection {self.collection_name} does not exist")
                return []

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            )

            # Extract words and scores
            results = []

            # Handle response format
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

            # If we couldn't extract results, log a warning
            if not results:
                logger.warning(f"Could not extract results from search_result")

            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
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
            words = []
            offset = None

            # We may need to make multiple scroll requests for large collections
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,  # Get up to 1000 points at a time
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )

                # Extract points and next offset
                if isinstance(scroll_result, tuple) and len(scroll_result) >= 2:
                    points, offset = scroll_result[0], scroll_result[1]
                else:
                    points = scroll_result.points if hasattr(scroll_result, 'points') else []
                    offset = None

                # No more points to process
                if not points:
                    break

                # Extract words from payload
                for point in points:
                    if hasattr(point, 'payload') and point.payload and "word" in point.payload:
                        words.append(point.payload["word"])
                    elif isinstance(point, dict) and "payload" in point and "word" in point["payload"]:
                        words.append(point["payload"]["word"])

                # If there's no offset, we've processed all points
                if offset is None:
                    break

            # If we couldn't extract words, use fallback
            if not words:
                logger.warning("Could not extract words from vector database. Using fallback words.")
                words = [
                    "apple", "banana", "cherry", "date", "elderberry",
                    "fig", "grape", "honeydew", "kiwi", "lemon"
                ]

            logger.info(f"Retrieved {len(words)} words from the vector database")
            return words

        except Exception as e:
            logger.error(f"Error getting all words: {e}")
            # Return some fallback words
            return [
                "apple", "banana", "cherry", "date", "elderberry",
                "fig", "grape", "honeydew", "kiwi", "lemon"
            ]
