"""Tests for the VectorDB class."""

import os
import pytest
import numpy as np

# We'll implement this class later
from contexto.vector_db import VectorDB


class TestVectorDB:
    """Test suite for the VectorDB class."""

    def test_init(self, vector_index_dir):
        """Test that the VectorDB class initializes correctly."""
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        assert db.collection_name == "test_collection"
        assert db.path == vector_index_dir
        assert db.model is None  # Model should be loaded on demand

    def test_load_model(self, vector_index_dir):
        """Test that the model loads correctly."""
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        db.load_model()
        assert db.model is not None
        # Check that the model has the expected methods
        assert hasattr(db.model, "encode")

    def test_get_embedding(self, vector_index_dir):
        """Test that embeddings are generated correctly."""
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        db.load_model()
        embedding = db.get_embedding("test")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 768  # Expected dimension for the model

    def test_build_index(self, vector_index_dir, small_word_list):
        """Test that the index builds correctly."""
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        count = db.build_index(small_word_list)
        assert count > 0
        assert os.path.exists(vector_index_dir)
        # Check that the collection exists
        assert db.collection_exists("test_collection")

    def test_search(self, vector_index_dir, small_word_list):
        """Test that search returns expected results."""
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        db.build_index(small_word_list)
        results = db.search("paper", limit=5)
        assert len(results) <= 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # The first result should be "paper" itself or something very close
        assert results[0][0] in ["paper", "document", "manuscript"]

    def test_get_distance(self, vector_index_dir):
        """Test that distance calculation works correctly."""
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        db.load_model()
        distance = db.get_distance("paper", "document")
        assert 0 <= distance <= 2  # Cosine distance is between 0 and 2
        # Distance to self should be 0
        assert db.get_distance("paper", "paper") < 0.01

    def test_collection_exists(self, vector_index_dir, small_word_list):
        """Test that collection existence check works."""
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        assert not db.collection_exists("test_collection")
        db.build_index(small_word_list)
        assert db.collection_exists("test_collection")
