"""Vector database implementation using Qdrant."""

import os
from typing import List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


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
        self.model = SentenceTransformer(model_name)

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
            self.load_model()

        # Read words from file
        with open(word_list_path, "r") as f:
            words = [line.strip() for line in f if line.strip()]

        # Create collection if it doesn't exist
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.get_embedding("test").shape[0],
                    distance=models.Distance.COSINE
                )
            )

        # Prepare points for batch upload
        points = []
        for i, word in enumerate(words):
            embedding = self.get_embedding(word)
            points.append(
                models.PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload={"word": word}
                )
            )

        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

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
