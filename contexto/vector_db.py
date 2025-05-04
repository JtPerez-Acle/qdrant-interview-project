"""Vector database implementation using Qdrant."""

import os
from typing import List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class VectorDB:
    """Vector database for storing and retrieving word embeddings."""

    def __init__(self, collection_name: str = "words", path: str = "./data/vector_index"):
        """Initialize the vector database.

        Args:
            collection_name: Name of the Qdrant collection
            path: Path to store the Qdrant database
        """
        self.collection_name = collection_name
        self.path = path
        self.model = None

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Initialize Qdrant client
        self.client = QdrantClient(path=path)

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
        except ImportError:
            has_gpu = False

        # Batch size for embedding generation
        embedding_batch_size = 64 if has_gpu else 32

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

            # Log progress for large datasets
            if i % 1000 == 0 and i > 0:
                print(f"Uploaded {i+len(batch)} points out of {len(points)} ({(i+len(batch))/len(points)*100:.1f}%)")

            # Free memory for very large datasets by clearing processed points
            if len(points) > 10000 and i % 5000 == 0 and i > 0:
                # Clear already processed points to free memory
                points_to_keep = points[i+upload_batch_size:]
                del points
                points = points_to_keep

        print(f"Successfully indexed {len(words)} words")
        return len(words)

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search for semantically similar words.

        Args:
            query: Query word or phrase
            limit: Maximum number of results

        Returns:
            List of (word, score) tuples
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Search for similar vectors using query_points (non-deprecated method)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit
        )

        # Extract words and scores
        results = []
        # Check the type of search_result to handle it correctly
        if hasattr(search_result, 'points'):
            # Handle QueryResponse object
            for hit in search_result.points:
                word = hit.payload.get("word")
                score = 1.0 - hit.score  # Convert cosine distance to similarity
                results.append((word, score))
        else:
            # Handle list of tuples or other return types
            print(f"Debug - search_result type: {type(search_result)}")
            if search_result and isinstance(search_result[0], tuple):
                for hit in search_result:
                    if len(hit) >= 2 and isinstance(hit[1], dict) and "word" in hit[1]:
                        word = hit[1]["word"]
                        score = 1.0 - hit[0]  # Assuming hit[0] is the score
                        results.append((word, score))

        return results

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
