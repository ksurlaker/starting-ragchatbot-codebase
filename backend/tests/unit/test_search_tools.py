"""Unit tests for CourseSearchTool and ToolManager"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models import Source
from search_tools import CourseSearchTool, Tool, ToolManager


class TestCourseSearchTool:
    """Test CourseSearchTool execution and functionality"""

    def test_execute_basic_query_success(
        self, mock_vector_store, sample_search_results
    ):
        """Test basic search query without filters returns formatted results"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="What is MCP?")

        # Assert
        assert result is not None
        assert "MCP (Model Context Protocol)" in result or "MCP" in result
        assert "Introduction to MCP" in result
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name=None, lesson_number=None
        )

    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test search with course name filter"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="What is MCP?", course_name="MCP")

        # Assert
        assert result is not None
        assert "Introduction to MCP" in result
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name="MCP", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test search with lesson number filter"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="MCP architecture", lesson_number=1)

        # Assert
        assert result is not None
        assert "Lesson 1" in result or "MCP" in result
        mock_vector_store.search.assert_called_once_with(
            query="MCP architecture", course_name=None, lesson_number=1
        )

    def test_execute_course_not_found(self, mock_vector_store, error_search_results):
        """Test handling when course is not found"""
        # Arrange
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="test query", course_name="NonExistent")

        # Assert
        assert "No course found" in result or "NonExistent" in result

    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test handling when search returns no results"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="very specific query")

        # Assert
        assert "No relevant content found" in result

    def test_execute_formats_results_correctly(
        self, mock_vector_store, sample_search_results
    ):
        """Test that results are formatted with course and lesson information"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="What is MCP?")

        # Assert
        # Check that course title is included
        assert "Introduction to MCP" in result
        # Check that lesson numbers are included
        assert "Lesson" in result
        # Check that actual content is included
        assert "MCP" in result

    def test_execute_tracks_sources(self, mock_vector_store, sample_search_results):
        """Test that sources are properly tracked after search"""
        # Arrange
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"
        tool = CourseSearchTool(mock_vector_store)

        # Act
        _result = tool.execute(query="What is MCP?")

        # Assert
        assert len(tool.last_sources) == 2  # Two documents in sample results
        assert all(isinstance(source, Source) for source in tool.last_sources)
        assert tool.last_sources[0].course_title == "Introduction to MCP"
        assert tool.last_sources[0].lesson_number == 1
        assert tool.last_sources[1].lesson_number == 2


class TestToolManager:
    """Test ToolManager functionality"""

    def test_register_tool_success(self):
        """Test successful tool registration"""
        # Arrange
        manager = ToolManager()
        mock_tool = MagicMock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "A test tool",
        }

        # Act
        manager.register_tool(mock_tool)

        # Assert
        assert "test_tool" in manager.tools
        assert manager.tools["test_tool"] == mock_tool

    def test_register_tool_no_name(self):
        """Test that tool registration fails without name"""
        # Arrange
        manager = ToolManager()
        mock_tool = MagicMock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {}

        # Act & Assert
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self, mock_vector_store):
        """Test retrieving all tool definitions"""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Act
        definitions = manager.get_tool_definitions()

        # Assert
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
        assert "description" in definitions[0]
        assert "input_schema" in definitions[0]

    def test_execute_tool_found(self, mock_vector_store, sample_search_results):
        """Test executing a registered tool"""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Act
        result = manager.execute_tool("search_course_content", query="test query")

        # Assert
        assert result is not None
        assert isinstance(result, str)
        mock_vector_store.search.assert_called_once()

    def test_execute_tool_not_found(self):
        """Test executing a non-existent tool"""
        # Arrange
        manager = ToolManager()

        # Act
        result = manager.execute_tool("nonexistent_tool", query="test")

        # Assert
        assert "not found" in result.lower()

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test retrieving sources from last search"""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="test query")

        # Act
        sources = manager.get_last_sources()

        # Assert
        assert len(sources) > 0
        assert all(isinstance(source, Source) for source in sources)

    def test_get_last_sources_empty(self):
        """Test retrieving sources when none exist"""
        # Arrange
        manager = ToolManager()

        # Act
        sources = manager.get_last_sources()

        # Assert
        assert sources == []

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources from all tools"""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="test query")
        assert len(manager.get_last_sources()) > 0

        # Act
        manager.reset_sources()

        # Assert
        assert manager.get_last_sources() == []
