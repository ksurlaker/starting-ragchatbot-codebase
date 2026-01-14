"""Integration tests for RAG System end-to-end functionality"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models import Source
from rag_system import RAGSystem


class TestRAGSystemIntegration:
    """Integration tests for full RAG query flow"""

    @pytest.fixture
    def rag_system(self, mock_config):
        """Create RAGSystem with mocked dependencies"""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.SessionManager"),
        ):

            # Create RAG system instance
            rag = RAGSystem(mock_config)

            # Replace with our mocks
            rag.vector_store = MagicMock()
            rag.ai_generator = MagicMock()
            rag.tool_manager = MagicMock()
            rag.session_manager = MagicMock()

            return rag

    def test_query_without_tool_use(self, rag_system, sample_sources):
        """Test full query flow when Claude answers without searching"""
        # Arrange
        rag_system.session_manager.get_conversation_history.return_value = None
        rag_system.ai_generator.generate_response.return_value = "2+2 equals 4"
        rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        answer, sources = rag_system.query("What is 2+2?", session_id="test_session")

        # Assert
        assert answer == "2+2 equals 4"
        assert sources == []
        rag_system.ai_generator.generate_response.assert_called_once()
        rag_system.session_manager.add_exchange.assert_called_once()

    def test_query_with_tool_use(self, rag_system, sample_sources):
        """Test full query flow when Claude searches then answers"""
        # Arrange
        rag_system.session_manager.get_conversation_history.return_value = None
        rag_system.ai_generator.generate_response.return_value = (
            "MCP stands for Model Context Protocol"
        )
        rag_system.tool_manager.get_last_sources.return_value = sample_sources

        # Act
        answer, sources = rag_system.query("What is MCP?", session_id="test_session")

        # Assert
        assert answer == "MCP stands for Model Context Protocol"
        assert len(sources) == 2
        assert all(isinstance(source, Source) for source in sources)
        rag_system.ai_generator.generate_response.assert_called_once()
        rag_system.tool_manager.get_last_sources.assert_called_once()

    def test_source_tracking(self, rag_system, sample_sources):
        """Test that sources are captured and returned correctly"""
        # Arrange
        rag_system.session_manager.get_conversation_history.return_value = None
        rag_system.ai_generator.generate_response.return_value = "Answer with sources"
        rag_system.tool_manager.get_last_sources.return_value = sample_sources

        # Act
        answer, sources = rag_system.query("Test query", session_id="test_session")

        # Assert
        assert len(sources) == 2
        assert sources[0].course_title == "Introduction to MCP"
        assert sources[0].lesson_number == 1
        assert sources[0].url == "https://example.com/lesson/1"
        assert sources[1].lesson_number == 2

    def test_source_reset(self, rag_system, sample_sources):
        """Test that sources are reset after retrieval"""
        # Arrange
        rag_system.session_manager.get_conversation_history.return_value = None
        rag_system.ai_generator.generate_response.return_value = "Answer"
        rag_system.tool_manager.get_last_sources.return_value = sample_sources

        # Act
        answer, sources = rag_system.query("Test query", session_id="test_session")

        # Assert
        # Verify reset was called after get_last_sources
        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_empty_search_results(self, rag_system):
        """Test handling when search returns no results"""
        # Arrange
        rag_system.session_manager.get_conversation_history.return_value = None
        rag_system.ai_generator.generate_response.return_value = "No information found"
        rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        answer, sources = rag_system.query(
            "Very specific query", session_id="test_session"
        )

        # Assert
        assert answer == "No information found"
        assert sources == []
        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_chromadb_error_handling(self, rag_system):
        """Test that errors propagate correctly from vector store"""
        # Arrange
        rag_system.session_manager.get_conversation_history.return_value = None
        # Simulate AIGenerator receiving error from tool execution
        rag_system.ai_generator.generate_response.return_value = (
            "I encountered an error searching the database"
        )
        rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        answer, sources = rag_system.query("Test query", session_id="test_session")

        # Assert
        assert "error" in answer.lower()
        assert sources == []

    def test_session_management_integration(self, rag_system):
        """Test that conversation history persists across queries"""
        # Arrange
        session_id = "test_session_123"
        rag_system.session_manager.get_conversation_history.return_value = (
            "User: Hello\nAssistant: Hi there!"
        )
        rag_system.ai_generator.generate_response.return_value = "How can I help you?"
        rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        answer, sources = rag_system.query("Can you help me?", session_id=session_id)

        # Assert
        # Verify history was retrieved
        rag_system.session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )

        # Verify AI generator received history
        call_kwargs = rag_system.ai_generator.generate_response.call_args[1]
        assert (
            call_kwargs["conversation_history"] == "User: Hello\nAssistant: Hi there!"
        )

        # Verify exchange was added
        rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id, "Can you help me?", "How can I help you?"
        )

    def test_tool_definitions_passed_to_ai(self, rag_system):
        """Test that tool definitions are passed to AIGenerator"""
        # Arrange
        rag_system.session_manager.get_conversation_history.return_value = None
        rag_system.ai_generator.generate_response.return_value = "Answer"
        rag_system.tool_manager.get_last_sources.return_value = []
        rag_system.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search"}
        ]

        # Act
        answer, sources = rag_system.query("Test query", session_id="test_session")

        # Assert
        call_kwargs = rag_system.ai_generator.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == [
            {"name": "search_course_content", "description": "Search"}
        ]
        assert "tool_manager" in call_kwargs

    def test_query_without_session_id(self, rag_system):
        """Test query without providing session ID"""
        # Arrange
        rag_system.session_manager.get_conversation_history.return_value = None
        rag_system.ai_generator.generate_response.return_value = "Answer"
        rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        answer, sources = rag_system.query("Test query", session_id=None)

        # Assert
        # Should still work, just without history
        assert answer == "Answer"
        rag_system.session_manager.get_conversation_history.assert_not_called()


class TestRAGSystemCourseManagement:
    """Test RAG system course document management"""

    @pytest.fixture
    def rag_system_with_real_vector_store(self, mock_config):
        """Create RAGSystem with mocked VectorStore but real interface"""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.SessionManager"),
        ):

            rag = RAGSystem(mock_config)
            rag.vector_store = MagicMock()
            return rag

    def test_get_course_analytics(self, rag_system_with_real_vector_store):
        """Test retrieving course analytics"""
        # Arrange
        rag_system_with_real_vector_store.vector_store.get_course_count.return_value = 3
        rag_system_with_real_vector_store.vector_store.get_existing_course_titles.return_value = [
            "Introduction to MCP",
            "Python Basics",
            "Web Development",
        ]

        # Act
        analytics = rag_system_with_real_vector_store.get_course_analytics()

        # Assert
        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Introduction to MCP" in analytics["course_titles"]
