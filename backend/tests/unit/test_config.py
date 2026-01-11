"""Unit tests for configuration validation"""
import pytest
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from config import config


class TestConfigValidation:
    """Test configuration values and validation"""

    def test_max_results_is_positive(self):
        """Test that MAX_RESULTS is greater than 0"""
        # This test will FAIL with the current bug
        assert config.MAX_RESULTS > 0, f"MAX_RESULTS must be positive, got {config.MAX_RESULTS}"

    def test_max_results_is_reasonable(self):
        """Test that MAX_RESULTS is in a reasonable range"""
        # Should be between 1 and 100
        assert 1 <= config.MAX_RESULTS <= 100, f"MAX_RESULTS should be between 1 and 100, got {config.MAX_RESULTS}"

    def test_chunk_size_is_positive(self):
        """Test that CHUNK_SIZE is greater than 0"""
        assert config.CHUNK_SIZE > 0, f"CHUNK_SIZE must be positive, got {config.CHUNK_SIZE}"

    def test_chunk_overlap_is_non_negative(self):
        """Test that CHUNK_OVERLAP is non-negative"""
        assert config.CHUNK_OVERLAP >= 0, f"CHUNK_OVERLAP must be non-negative, got {config.CHUNK_OVERLAP}"

    def test_chunk_overlap_less_than_chunk_size(self):
        """Test that CHUNK_OVERLAP is less than CHUNK_SIZE"""
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE, \
            f"CHUNK_OVERLAP ({config.CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({config.CHUNK_SIZE})"

    def test_max_history_is_non_negative(self):
        """Test that MAX_HISTORY is non-negative"""
        assert config.MAX_HISTORY >= 0, f"MAX_HISTORY must be non-negative, got {config.MAX_HISTORY}"

    def test_anthropic_api_key_exists(self):
        """Test that ANTHROPIC_API_KEY is set"""
        # Should be set in environment or config
        assert config.ANTHROPIC_API_KEY is not None, "ANTHROPIC_API_KEY must be set"
        # Can be empty if .env not loaded, but should exist as attribute
        assert hasattr(config, 'ANTHROPIC_API_KEY'), "Config must have ANTHROPIC_API_KEY attribute"

    def test_anthropic_model_is_set(self):
        """Test that a valid Claude model is specified"""
        assert config.ANTHROPIC_MODEL is not None, "ANTHROPIC_MODEL must be set"
        assert len(config.ANTHROPIC_MODEL) > 0, "ANTHROPIC_MODEL cannot be empty"
        assert "claude" in config.ANTHROPIC_MODEL.lower(), "ANTHROPIC_MODEL should be a Claude model"

    def test_embedding_model_is_set(self):
        """Test that embedding model is specified"""
        assert config.EMBEDDING_MODEL is not None, "EMBEDDING_MODEL must be set"
        assert len(config.EMBEDDING_MODEL) > 0, "EMBEDDING_MODEL cannot be empty"

    def test_chroma_path_is_set(self):
        """Test that ChromaDB path is specified"""
        assert config.CHROMA_PATH is not None, "CHROMA_PATH must be set"
        assert len(config.CHROMA_PATH) > 0, "CHROMA_PATH cannot be empty"
