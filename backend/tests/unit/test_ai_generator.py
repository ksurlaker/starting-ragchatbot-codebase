"""Unit tests for AIGenerator tool calling and response generation"""

import os
import sys
from unittest.mock import MagicMock, patch

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ai_generator import AIGenerator


class TestAIGeneratorToolCalling:
    """Test AIGenerator tool calling functionality"""

    def test_generate_no_tool_use(self, mock_anthropic_text_response):
        """Test response generation when Claude doesn't use tools"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_text_response
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514"
            )

            # Act
            result = generator.generate_response(
                query="What is 2+2?", tools=None, tool_manager=None
            )

            # Assert
            assert (
                result
                == "MCP stands for Model Context Protocol. It's a standardized way for AI models to connect to data sources."
            )
            mock_client.messages.create.assert_called_once()

    def test_generate_with_tool_use(
        self,
        mock_anthropic_tool_use_response,
        mock_anthropic_final_response,
        mock_tool_manager,
    ):
        """Test response generation when Claude requests tool use"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            # First call returns tool_use, second call returns final response
            mock_client.messages.create.side_effect = [
                mock_anthropic_tool_use_response,
                mock_anthropic_final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514"
            )

            # Act
            result = generator.generate_response(
                query="What is MCP?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Assert
            assert (
                result
                == "Based on the course materials, MCP (Model Context Protocol) is a protocol that allows AI models to connect to external data sources and tools."
            )
            assert mock_client.messages.create.call_count == 2
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="What is MCP?", course_name="MCP"
            )

    def test_handle_tool_execution_single(
        self,
        mock_anthropic_tool_use_response,
        mock_anthropic_final_response,
        mock_tool_manager,
    ):
        """Test handling of single tool execution"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [
                mock_anthropic_tool_use_response,
                mock_anthropic_final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514"
            )
            tool_definitions = [
                {"name": "search_course_content", "description": "Search"}
            ]

            # Act
            result = generator.generate_response(
                query="What is MCP?",
                tools=tool_definitions,
                tool_manager=mock_tool_manager,
            )

            # Assert
            assert result is not None
            assert isinstance(result, str)
            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once()

    def test_handle_tool_execution_formats_messages(
        self,
        mock_anthropic_tool_use_response,
        mock_anthropic_final_response,
        mock_tool_manager,
    ):
        """Test that messages are properly formatted for second API call"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [
                mock_anthropic_tool_use_response,
                mock_anthropic_final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514"
            )

            # Act
            _result = generator.generate_response(
                query="What is MCP?",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

            # Assert
            assert mock_client.messages.create.call_count == 2

            # Check second call (final response generation)
            second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
            messages = second_call_kwargs["messages"]

            # Should have 3 messages: original user query, assistant tool use, user tool results
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"

            # Tool results should be in the last message
            assert "content" in messages[2]
            assert isinstance(messages[2]["content"], list)

    def test_tool_execution_error_propagation(
        self,
        mock_anthropic_tool_use_response,
        mock_anthropic_final_response,
        mock_tool_manager,
    ):
        """Test that tool execution errors are passed to Claude for explanation"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [
                mock_anthropic_tool_use_response,
                mock_anthropic_final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            # Configure tool manager to return error
            mock_tool_manager.execute_tool.return_value = (
                "Search error: ChromaDB connection failed"
            )

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514"
            )

            # Act
            _result = generator.generate_response(
                query="What is MCP?",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

            # Assert
            # Tool error should be included in second API call
            second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
            messages = second_call_kwargs["messages"]

            # Check that tool result contains the error
            tool_result_content = messages[2]["content"][0]
            assert tool_result_content["type"] == "tool_result"
            assert "Search error" in tool_result_content["content"]

    def test_conversation_history_included(self, mock_anthropic_text_response):
        """Test that conversation history is included in system prompt"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_text_response
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514"
            )
            conversation_history = (
                "User: What is MCP?\nAssistant: MCP stands for Model Context Protocol."
            )

            # Act
            _result = generator.generate_response(
                query="Tell me more", conversation_history=conversation_history
            )

            # Assert
            call_kwargs = mock_client.messages.create.call_args[1]
            system_prompt = call_kwargs["system"]

            # History should be in system prompt
            assert "What is MCP?" in system_prompt
            assert "Model Context Protocol" in system_prompt

    def test_generate_with_tools_but_no_manager_raises_error(
        self, mock_anthropic_tool_use_response
    ):
        """Test that providing tools without tool_manager handles gracefully"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_tool_use_response
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514"
            )

            # Act
            # When tool_use is requested but no tool_manager provided, should return the response as-is
            result = generator.generate_response(
                query="What is MCP?",
                tools=[{"name": "search_course_content"}],
                tool_manager=None,
            )

            # Assert
            # Without tool_manager, cannot execute tools, so returns initial response
            # This tests the safety of the code
            assert result is not None


class TestAIGeneratorConfiguration:
    """Test AIGenerator configuration and setup"""

    def test_initialization_with_valid_params(self):
        """Test AIGenerator initialization with valid parameters"""
        # Arrange & Act
        with patch("anthropic.Anthropic"):
            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514"
            )

            # Assert
            assert generator.model == "claude-sonnet-4-20250514"
            assert generator.base_params["model"] == "claude-sonnet-4-20250514"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800

    def test_system_prompt_includes_tool_guidance(self):
        """Test that system prompt includes tool usage guidance"""
        # Arrange & Act
        with patch("anthropic.Anthropic"):
            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514"
            )

            # Assert
            assert "search_course_content" in generator.SYSTEM_PROMPT
            assert "get_course_outline" in generator.SYSTEM_PROMPT
            assert "Tool Usage Guidelines" in generator.SYSTEM_PROMPT


class TestSequentialToolCalling:
    """Test multi-round tool calling with iterative loop"""

    def test_zero_rounds_no_tools_used(self, mock_anthropic_text_response):
        """Claude answers directly without tools"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_text_response
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514", max_tool_rounds=2
            )

            # Act
            result = generator.generate_response(
                query="What is 2+2?",
                tools=[{"name": "search_course_content"}],
                tool_manager=MagicMock(),
            )

            # Assert
            assert (
                result
                == "MCP stands for Model Context Protocol. It's a standardized way for AI models to connect to data sources."
            )
            assert mock_client.messages.create.call_count == 1
            # Verify tools were included in the call
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "tools" in call_kwargs

    def test_single_round_tool_use(
        self,
        mock_anthropic_tool_use_response,
        mock_anthropic_final_response,
        mock_tool_manager,
    ):
        """Simple query requiring one tool call"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [
                mock_anthropic_tool_use_response,
                mock_anthropic_final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514", max_tool_rounds=2
            )

            # Act
            result = generator.generate_response(
                query="What is MCP?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Assert
            assert (
                result
                == "Based on the course materials, MCP (Model Context Protocol) is a protocol that allows AI models to connect to external data sources and tools."
            )
            assert mock_client.messages.create.call_count == 2
            mock_tool_manager.execute_tool.assert_called_once()

            # Verify both calls included tools parameter
            for call in mock_client.messages.create.call_args_list:
                call_kwargs = call[1]
                assert "tools" in call_kwargs

    def test_two_rounds_sequential_tools(
        self, mock_anthropic_multi_round_sequence, mock_tool_manager
    ):
        """Complex query requiring two tool calls"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = (
                mock_anthropic_multi_round_sequence
            )
            mock_anthropic_class.return_value = mock_client

            # Configure tool manager to return different results for each call
            mock_tool_manager.execute_tool.side_effect = [
                "Course: MCP\nLesson 1: Introduction\nLesson 2: MCP Architecture\nLesson 3: Building Servers",
                "[Lesson 2 - MCP Architecture]\nThe architecture consists of client, server, and protocol layers.",
            ]

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514", max_tool_rounds=2
            )

            # Act
            result = generator.generate_response(
                query="What topics are covered in the MCP Architecture lesson?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Assert
            assert (
                result
                == "Based on the course outline and detailed search, MCP Architecture covers the core protocol design and implementation patterns."
            )
            assert mock_client.messages.create.call_count == 3  # 3 API calls total
            assert mock_tool_manager.execute_tool.call_count == 2  # 2 tool executions

            # Verify messages grow: 1 → 3 → 5
            call_1_messages = mock_client.messages.create.call_args_list[0][1][
                "messages"
            ]
            call_2_messages = mock_client.messages.create.call_args_list[1][1][
                "messages"
            ]
            call_3_messages = mock_client.messages.create.call_args_list[2][1][
                "messages"
            ]

            assert len(call_1_messages) == 1  # Initial user query
            assert (
                len(call_2_messages) == 3
            )  # user, assistant(tool_use), user(tool_results)
            assert (
                len(call_3_messages) == 5
            )  # previous 3 + assistant(tool_use) + user(tool_results)

    def test_max_rounds_exceeded(
        self, mock_anthropic_infinite_tool_use, mock_tool_manager
    ):
        """Loop terminates at MAX_TOOL_ROUNDS"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()

            # Create final response for when tools are removed
            final_response = MagicMock()
            final_response.stop_reason = "end_turn"
            final_text = MagicMock()
            final_text.text = "I've gathered the information."
            final_text.type = "text"
            final_response.content = [final_text]

            # Always return tool_use for first 2 calls, then final response
            mock_client.messages.create.side_effect = [
                mock_anthropic_infinite_tool_use,
                mock_anthropic_infinite_tool_use,
                final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514", max_tool_rounds=2
            )

            # Act
            result = generator.generate_response(
                query="Test query",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Assert
            assert result == "I've gathered the information."
            assert (
                mock_client.messages.create.call_count == 3
            )  # Exactly MAX_TOOL_ROUNDS + 1 calls
            assert (
                mock_tool_manager.execute_tool.call_count == 2
            )  # Exactly MAX_TOOL_ROUNDS executions

            # Verify final call does NOT include tools
            final_call_kwargs = mock_client.messages.create.call_args_list[2][1]
            assert "tools" not in final_call_kwargs

    def test_tool_error_passed_to_claude(
        self,
        mock_anthropic_tool_use_response,
        mock_anthropic_final_response,
        mock_tool_manager,
    ):
        """Tool errors included in next API call"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [
                mock_anthropic_tool_use_response,
                mock_anthropic_final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            # Configure tool to return error
            mock_tool_manager.execute_tool.return_value = (
                "error: ChromaDB connection failed"
            )

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514", max_tool_rounds=2
            )

            # Act
            _result = generator.generate_response(
                query="What is MCP?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Assert
            # Verify error is in tool_result content
            second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
            messages = second_call_kwargs["messages"]
            tool_result_content = messages[2]["content"][0]
            assert "error: ChromaDB connection failed" in tool_result_content["content"]

    def test_all_tools_fail_terminates_loop(
        self, mock_anthropic_tool_use_response, mock_tool_manager
    ):
        """Complete tool failure stops iteration"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_tool_use_response
            mock_anthropic_class.return_value = mock_client

            # Configure all tools to fail
            mock_tool_manager.execute_tool.return_value = "No relevant content found"

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514", max_tool_rounds=2
            )

            # Act
            _result = generator.generate_response(
                query="What is XYZ?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Assert
            # Loop should terminate early after first failed tool
            assert mock_client.messages.create.call_count == 1  # Only initial call
            assert mock_tool_manager.execute_tool.call_count == 1  # Only one execution

    def test_message_history_accumulation(
        self, mock_anthropic_multi_round_sequence, mock_tool_manager
    ):
        """Messages array builds correctly"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = (
                mock_anthropic_multi_round_sequence
            )
            mock_anthropic_class.return_value = mock_client

            mock_tool_manager.execute_tool.return_value = "Tool result"

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514", max_tool_rounds=2
            )

            # Act
            _result = generator.generate_response(
                query="Test query",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Assert message structure
            # Round 1: [user] → execute tool → [user, assistant, user]
            round_1_messages = mock_client.messages.create.call_args_list[0][1][
                "messages"
            ]
            assert len(round_1_messages) == 1
            assert round_1_messages[0]["role"] == "user"

            # Round 2: [user, assistant, user] → execute tool → [user, assistant, user, assistant, user]
            round_2_messages = mock_client.messages.create.call_args_list[1][1][
                "messages"
            ]
            assert len(round_2_messages) == 3
            assert round_2_messages[0]["role"] == "user"
            assert round_2_messages[1]["role"] == "assistant"
            assert round_2_messages[2]["role"] == "user"

            # Final call: 5 messages
            final_messages = mock_client.messages.create.call_args_list[2][1][
                "messages"
            ]
            assert len(final_messages) == 5

    def test_backward_compatibility_single_tool(
        self,
        mock_anthropic_tool_use_response,
        mock_anthropic_final_response,
        mock_tool_manager,
    ):
        """Existing single-tool queries still work"""
        # Arrange
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [
                mock_anthropic_tool_use_response,
                mock_anthropic_final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="claude-sonnet-4-20250514", max_tool_rounds=2
            )

            # Act
            result = generator.generate_response(
                query="What is MCP?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Assert - same results as before refactoring
            assert (
                result
                == "Based on the course materials, MCP (Model Context Protocol) is a protocol that allows AI models to connect to external data sources and tools."
            )
            assert mock_client.messages.create.call_count == 2
            mock_tool_manager.execute_tool.assert_called_once()
