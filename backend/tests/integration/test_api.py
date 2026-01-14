"""API endpoint tests for the RAG chatbot FastAPI application"""
import pytest
from unittest.mock import MagicMock
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models import Source


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_success_with_sources(self, sync_test_client, mock_rag_system):
        """Test successful query returns answer and sources"""
        # Act
        response = sync_test_client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "existing-session"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test answer from the RAG system."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["course_title"] == "Test Course"
        assert data["session_id"] == "existing-session"

    def test_query_creates_session_when_not_provided(self, sync_test_client, mock_rag_system):
        """Test that session is created when session_id is not provided"""
        # Act
        response = sync_test_client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-id-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_provided_session_id(self, sync_test_client, mock_rag_system):
        """Test that provided session_id is used"""
        # Act
        response = sync_test_client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "user-provided-session"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "user-provided-session"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_empty_sources(self, sync_test_client, mock_rag_system):
        """Test query response when no sources are found"""
        # Arrange
        mock_rag_system.query.return_value = ("Answer without sources", [])

        # Act
        response = sync_test_client.post(
            "/api/query",
            json={"query": "Generic question", "session_id": "test-session"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Answer without sources"
        assert data["sources"] == []

    def test_query_missing_query_field(self, sync_test_client):
        """Test that missing query field returns 422 validation error"""
        # Act
        response = sync_test_client.post(
            "/api/query",
            json={"session_id": "test-session"}
        )

        # Assert
        assert response.status_code == 422

    def test_query_empty_query_string(self, sync_test_client, mock_rag_system):
        """Test query with empty string"""
        # Act
        response = sync_test_client.post(
            "/api/query",
            json={"query": "", "session_id": "test-session"}
        )

        # Assert
        # Empty query is valid input - the RAG system handles it
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with("", "test-session")

    def test_query_rag_system_error(self, sync_test_client, mock_rag_system):
        """Test that RAG system errors return 500 status"""
        # Arrange
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        # Act
        response = sync_test_client.post(
            "/api/query",
            json={"query": "Test query", "session_id": "test-session"}
        )

        # Assert
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_query_multiple_sources(self, sync_test_client, mock_rag_system):
        """Test query returning multiple sources"""
        # Arrange
        mock_rag_system.query.return_value = (
            "Comprehensive answer from multiple lessons",
            [
                Source(
                    text="Course A - Lesson 1",
                    url="https://example.com/a/1",
                    course_title="Course A",
                    lesson_number=1
                ),
                Source(
                    text="Course A - Lesson 2",
                    url="https://example.com/a/2",
                    course_title="Course A",
                    lesson_number=2
                ),
                Source(
                    text="Course B - Lesson 1",
                    url="https://example.com/b/1",
                    course_title="Course B",
                    lesson_number=1
                )
            ]
        )

        # Act
        response = sync_test_client.post(
            "/api/query",
            json={"query": "Compare topics", "session_id": "test-session"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3
        assert data["sources"][0]["course_title"] == "Course A"
        assert data["sources"][2]["course_title"] == "Course B"


@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_get_courses_success(self, sync_test_client, mock_rag_system):
        """Test successful retrieval of course statistics"""
        # Act
        response = sync_test_client.get("/api/courses")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Introduction to Python" in data["course_titles"]
        assert "Web Development Basics" in data["course_titles"]

    def test_get_courses_empty(self, sync_test_client, mock_rag_system):
        """Test response when no courses are loaded"""
        # Arrange
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        # Act
        response = sync_test_client.get("/api/courses")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_error(self, sync_test_client, mock_rag_system):
        """Test that analytics errors return 500 status"""
        # Arrange
        mock_rag_system.get_course_analytics.side_effect = Exception("ChromaDB unavailable")

        # Act
        response = sync_test_client.get("/api/courses")

        # Assert
        assert response.status_code == 500
        assert "ChromaDB unavailable" in response.json()["detail"]

    def test_get_courses_many_courses(self, sync_test_client, mock_rag_system):
        """Test response with many courses"""
        # Arrange
        course_titles = [f"Course {i}" for i in range(50)]
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 50,
            "course_titles": course_titles
        }

        # Act
        response = sync_test_client.get("/api/courses")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 50
        assert len(data["course_titles"]) == 50


@pytest.mark.api
class TestRootEndpoint:
    """Tests for GET / root endpoint"""

    def test_root_returns_status(self, sync_test_client):
        """Test that root endpoint returns status information"""
        # Act
        response = sync_test_client.get("/")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


@pytest.mark.api
class TestAPIRequestValidation:
    """Tests for API request validation and edge cases"""

    def test_invalid_json_body(self, sync_test_client):
        """Test handling of invalid JSON in request body"""
        # Act
        response = sync_test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        # Assert
        assert response.status_code == 422

    def test_wrong_content_type(self, sync_test_client):
        """Test handling of wrong content type"""
        # Act
        response = sync_test_client.post(
            "/api/query",
            content="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        # Assert
        assert response.status_code == 422

    def test_extra_fields_ignored(self, sync_test_client, mock_rag_system):
        """Test that extra fields in request are ignored"""
        # Act
        response = sync_test_client.post(
            "/api/query",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "extra_field": "should be ignored",
                "another_field": 12345
            }
        )

        # Assert
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with("Test query", "test-session")

    def test_unicode_query(self, sync_test_client, mock_rag_system):
        """Test handling of unicode characters in query"""
        # Act
        response = sync_test_client.post(
            "/api/query",
            json={"query": "What is 日本語? How about émojis 🎉?", "session_id": "test"}
        )

        # Assert
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(
            "What is 日本語? How about émojis 🎉?",
            "test"
        )

    def test_long_query(self, sync_test_client, mock_rag_system):
        """Test handling of very long query strings"""
        # Arrange
        long_query = "word " * 1000  # 5000 character query

        # Act
        response = sync_test_client.post(
            "/api/query",
            json={"query": long_query, "session_id": "test"}
        )

        # Assert
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()


@pytest.mark.api
class TestAsyncAPIEndpoints:
    """Async tests for API endpoints using httpx AsyncClient"""

    @pytest.mark.asyncio
    async def test_async_query_endpoint(self, test_client, mock_rag_system):
        """Test query endpoint with async client"""
        async with test_client:
            response = await test_client.post(
                "/api/query",
                json={"query": "Async test query", "session_id": "async-session"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a test answer from the RAG system."

    @pytest.mark.asyncio
    async def test_async_courses_endpoint(self, test_client, mock_rag_system):
        """Test courses endpoint with async client"""
        async with test_client:
            response = await test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2

    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self, test_client, mock_rag_system):
        """Test handling of concurrent requests"""
        import asyncio

        async with test_client:
            # Make multiple concurrent requests
            tasks = [
                test_client.post(
                    "/api/query",
                    json={"query": f"Query {i}", "session_id": f"session-{i}"}
                )
                for i in range(5)
            ]
            responses = await asyncio.gather(*tasks)

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)
        assert mock_rag_system.query.call_count == 5
