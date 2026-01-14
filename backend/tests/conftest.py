"""Shared test fixtures for the RAG chatbot test suite"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, Lesson, Source
from vector_store import SearchResults


# =============================================================================
# API Test Fixtures
# =============================================================================

@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem for API testing"""
    mock_rag = MagicMock()
    mock_rag.query.return_value = (
        "This is a test answer from the RAG system.",
        [
            Source(
                text="Test Course - Lesson 1",
                url="https://example.com/lesson/1",
                course_title="Test Course",
                lesson_number=1
            )
        ]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to Python", "Web Development Basics"]
    }
    mock_rag.session_manager = MagicMock()
    mock_rag.session_manager.create_session.return_value = "test-session-id-123"
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app with API endpoints defined inline.

    This avoids import issues with the main app.py which mounts static files
    that don't exist in the test environment.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Optional

    app = FastAPI(title="Test RAG API")

    # Store mock RAG system on app state for access in endpoints
    app.state.rag_system = mock_rag_system

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = app.state.rag_system.session_manager.create_session()

            answer, sources = app.state.rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = app.state.rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "RAG API is running"}

    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for API testing using httpx"""
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=test_app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def sync_test_client(test_app):
    """Create a synchronous test client using Starlette TestClient"""
    from starlette.testclient import TestClient
    return TestClient(test_app)


@pytest.fixture
def mock_config():
    """Configuration with correct MAX_RESULTS value for testing"""
    config = MagicMock()
    config.ANTHROPIC_API_KEY = "test-api-key-123"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.MAX_RESULTS = 5
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    return config


@pytest.fixture
def sample_search_results():
    """Valid SearchResults with multiple documents"""
    return SearchResults(
        documents=[
            "MCP (Model Context Protocol) is a protocol for connecting AI models to data sources.",
            "The MCP server allows tools to access external data and services.",
        ],
        metadata=[
            {
                "course_title": "Introduction to MCP",
                "lesson_number": 1,
                "chunk_index": 0,
            },
            {
                "course_title": "Introduction to MCP",
                "lesson_number": 2,
                "chunk_index": 1,
            },
        ],
        distances=[0.3, 0.5],
    )


@pytest.fixture
def empty_search_results():
    """SearchResults with no documents"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """SearchResults with error message"""
    return SearchResults.empty("No course found matching 'NonExistent'")


@pytest.fixture
def mock_vector_store(sample_search_results):
    """Mocked VectorStore that returns valid search results"""
    mock_store = MagicMock()
    mock_store.search.return_value = sample_search_results
    mock_store._resolve_course_name.return_value = "Introduction to MCP"
    mock_store.get_lesson_link.return_value = "https://example.com/lesson/1"
    return mock_store


@pytest.fixture
def sample_course():
    """Sample Course object for testing"""
    return Course(
        title="Introduction to MCP",
        course_link="https://example.com/course/mcp",
        instructor="John Doe",
        lessons=[
            Lesson(
                lesson_number=1,
                title="What is MCP?",
                lesson_link="https://example.com/lesson/1",
            ),
            Lesson(
                lesson_number=2,
                title="MCP Architecture",
                lesson_link="https://example.com/lesson/2",
            ),
            Lesson(
                lesson_number=3,
                title="Building MCP Servers",
                lesson_link="https://example.com/lesson/3",
            ),
        ],
    )


@pytest.fixture
def sample_sources():
    """Sample Source objects for testing"""
    return [
        Source(
            text="Introduction to MCP - Lesson 1",
            url="https://example.com/lesson/1",
            course_title="Introduction to MCP",
            lesson_number=1,
        ),
        Source(
            text="Introduction to MCP - Lesson 2",
            url="https://example.com/lesson/2",
            course_title="Introduction to MCP",
            lesson_number=2,
        ),
    ]


@pytest.fixture
def mock_anthropic_text_response():
    """Mock Anthropic response without tool use"""
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"

    # Create mock content block
    mock_content = MagicMock()
    mock_content.text = "MCP stands for Model Context Protocol. It's a standardized way for AI models to connect to data sources."
    mock_content.type = "text"

    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_anthropic_tool_use_response():
    """Mock Anthropic response requesting tool use"""
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"

    # Create mock tool use content block
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "toolu_01234567890"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.input = {"query": "What is MCP?", "course_name": "MCP"}

    mock_response.content = [mock_tool_block]
    return mock_response


@pytest.fixture
def mock_anthropic_final_response():
    """Mock Anthropic final response after tool execution"""
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"

    # Create mock content block
    mock_content = MagicMock()
    mock_content.text = "Based on the course materials, MCP (Model Context Protocol) is a protocol that allows AI models to connect to external data sources and tools."
    mock_content.type = "text"

    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_anthropic_client(mock_anthropic_text_response):
    """Mock Anthropic client with messages.create method"""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_anthropic_text_response
    return mock_client


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection"""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Sample document content"]],
        "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
        "distances": [[0.3]],
    }
    mock_collection.get.return_value = {
        "ids": ["test_course_1"],
        "metadatas": [
            {
                "title": "Test Course",
                "instructor": "Test Instructor",
                "course_link": "https://example.com/course",
                "lessons_json": '[{"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "https://example.com/lesson/1"}]',
            }
        ],
    }
    return mock_collection


@pytest.fixture
def mock_tool_manager(sample_sources):
    """Mock ToolManager"""
    mock_manager = MagicMock()
    mock_manager.execute_tool.return_value = (
        "[Introduction to MCP - Lesson 1]\nMCP is a protocol for AI models."
    )
    mock_manager.get_last_sources.return_value = sample_sources
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"},
                },
                "required": ["query"],
            },
        }
    ]
    return mock_manager


@pytest.fixture
def mock_anthropic_second_tool_use_response():
    """Mock response for second round of tool use"""
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "toolu_round2_abc"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.input = {
        "query": "MCP Architecture details",
        "course_name": "MCP",
        "lesson_number": 2,
    }
    mock_response.content = [mock_tool_block]
    return mock_response


@pytest.fixture
def mock_anthropic_multi_round_sequence(
    mock_anthropic_tool_use_response,
    mock_anthropic_second_tool_use_response,
    mock_anthropic_final_response,
):
    """Sequence of responses for 2-round tool calling"""
    # Response 1: First tool use (get course outline)
    response1 = MagicMock()
    response1.stop_reason = "tool_use"
    tool_block1 = MagicMock()
    tool_block1.type = "tool_use"
    tool_block1.id = "toolu_round1"
    tool_block1.name = "get_course_outline"
    tool_block1.input = {"course_name": "MCP"}
    response1.content = [tool_block1]

    # Response 2: Second tool use (search content)
    response2 = MagicMock()
    response2.stop_reason = "tool_use"
    tool_block2 = MagicMock()
    tool_block2.type = "tool_use"
    tool_block2.id = "toolu_round2"
    tool_block2.name = "search_course_content"
    tool_block2.input = {"query": "MCP Architecture", "lesson_number": 2}
    response2.content = [tool_block2]

    # Response 3: Final text answer
    response3 = MagicMock()
    response3.stop_reason = "end_turn"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Based on the course outline and detailed search, MCP Architecture covers the core protocol design and implementation patterns."
    response3.content = [text_block]

    return [response1, response2, response3]


@pytest.fixture
def mock_anthropic_infinite_tool_use():
    """Mock response that always requests tools (for testing max rounds)"""
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "toolu_infinite"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.input = {"query": "test query"}

    mock_response.content = [mock_tool_block]
    return mock_response
