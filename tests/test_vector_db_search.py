"""Tests for the VectorDB search functionality."""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from collections import namedtuple

from contexto.vector_db import VectorDB


class TestVectorDBSearch:
    """Test the VectorDB search functionality."""

    def test_search_with_string_query(self, vector_index_dir):
        """Test search with a string query."""
        # Create a mock client
        mock_client = MagicMock()
        
        # Create a mock response
        MockPoint = namedtuple('MockPoint', ['id', 'payload', 'score'])
        mock_points = [
            MockPoint(id=1, payload={"word": "apple"}, score=0.1),
            MockPoint(id=2, payload={"word": "banana"}, score=0.2),
        ]
        mock_client.query_points.return_value = mock_points
        
        # Create VectorDB with mock client
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        db.client = mock_client
        
        # Mock the get_embedding method
        with patch.object(db, 'get_embedding', return_value=np.array([0.1, 0.2, 0.3])):
            # Call search
            results = db.search("test", limit=2)
            
            # Check results
            assert len(results) == 2
            assert results[0][0] == "apple"
            assert results[0][1] == pytest.approx(0.9)  # 1.0 - 0.1
            assert results[1][0] == "banana"
            assert results[1][1] == pytest.approx(0.8)  # 1.0 - 0.2
            
            # Check that query_points was called correctly
            mock_client.query_points.assert_called_once()
            args, kwargs = mock_client.query_points.call_args
            assert kwargs["collection_name"] == "test_collection"
            assert kwargs["limit"] == 2
            assert kwargs["query"] == [0.1, 0.2, 0.3]

    def test_search_with_vector_query(self, vector_index_dir):
        """Test search with a vector query."""
        # Create a mock client
        mock_client = MagicMock()
        
        # Create a mock response
        MockPoint = namedtuple('MockPoint', ['id', 'payload', 'score'])
        mock_points = [
            MockPoint(id=1, payload={"word": "apple"}, score=0.1),
            MockPoint(id=2, payload={"word": "banana"}, score=0.2),
        ]
        mock_client.query_points.return_value = mock_points
        
        # Create VectorDB with mock client
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        db.client = mock_client
        
        # Call search with a vector
        query_vector = np.array([0.1, 0.2, 0.3])
        results = db.search(query_vector, limit=2)
        
        # Check results
        assert len(results) == 2
        assert results[0][0] == "apple"
        assert results[0][1] == pytest.approx(0.9)  # 1.0 - 0.1
        assert results[1][0] == "banana"
        assert results[1][1] == pytest.approx(0.8)  # 1.0 - 0.2
        
        # Check that query_points was called correctly
        mock_client.query_points.assert_called_once()
        args, kwargs = mock_client.query_points.call_args
        assert kwargs["collection_name"] == "test_collection"
        assert kwargs["limit"] == 2
        assert kwargs["query"] == [0.1, 0.2, 0.3]

    def test_search_with_response_points_attribute(self, vector_index_dir):
        """Test search with a response that has a 'points' attribute."""
        # Create a mock client
        mock_client = MagicMock()
        
        # Create a mock response with 'points' attribute
        MockResponse = namedtuple('MockResponse', ['points'])
        MockPoint = namedtuple('MockPoint', ['id', 'payload', 'score'])
        mock_points = [
            MockPoint(id=1, payload={"word": "apple"}, score=0.1),
            MockPoint(id=2, payload={"word": "banana"}, score=0.2),
        ]
        mock_response = MockResponse(points=mock_points)
        mock_client.query_points.return_value = mock_response
        
        # Create VectorDB with mock client
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        db.client = mock_client
        
        # Mock the get_embedding method
        with patch.object(db, 'get_embedding', return_value=np.array([0.1, 0.2, 0.3])):
            # Call search
            results = db.search("test", limit=2)
            
            # Check results
            assert len(results) == 2
            assert results[0][0] == "apple"
            assert results[0][1] == pytest.approx(0.9)  # 1.0 - 0.1
            assert results[1][0] == "banana"
            assert results[1][1] == pytest.approx(0.8)  # 1.0 - 0.2

    def test_search_with_dict_response(self, vector_index_dir):
        """Test search with a response that is a list of dictionaries."""
        # Create a mock client
        mock_client = MagicMock()
        
        # Create a mock response as a list of dictionaries
        mock_response = [
            {"id": 1, "payload": {"word": "apple"}, "score": 0.1},
            {"id": 2, "payload": {"word": "banana"}, "score": 0.2},
        ]
        mock_client.query_points.return_value = mock_response
        
        # Create VectorDB with mock client
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        db.client = mock_client
        
        # Mock the get_embedding method
        with patch.object(db, 'get_embedding', return_value=np.array([0.1, 0.2, 0.3])):
            # Call search
            results = db.search("test", limit=2)
            
            # Check results
            assert len(results) == 2
            assert results[0][0] == "apple"
            assert results[0][1] == pytest.approx(0.9)  # 1.0 - 0.1
            assert results[1][0] == "banana"
            assert results[1][1] == pytest.approx(0.8)  # 1.0 - 0.2

    def test_search_with_empty_response(self, vector_index_dir):
        """Test search with an empty response."""
        # Create a mock client
        mock_client = MagicMock()
        
        # Create an empty mock response
        mock_client.query_points.return_value = []
        
        # Create VectorDB with mock client
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        db.client = mock_client
        
        # Mock the get_embedding method
        with patch.object(db, 'get_embedding', return_value=np.array([0.1, 0.2, 0.3])):
            # Call search
            results = db.search("test", limit=2)
            
            # Check results
            assert len(results) == 0

    def test_search_with_exception(self, vector_index_dir):
        """Test search when an exception occurs."""
        # Create a mock client
        mock_client = MagicMock()
        
        # Make query_points raise an exception
        mock_client.query_points.side_effect = Exception("Test exception")
        
        # Create VectorDB with mock client
        db = VectorDB(collection_name="test_collection", path=vector_index_dir)
        db.client = mock_client
        
        # Mock the get_embedding method
        with patch.object(db, 'get_embedding', return_value=np.array([0.1, 0.2, 0.3])):
            # Call search
            results = db.search("test", limit=2)
            
            # Check results
            assert len(results) == 0
