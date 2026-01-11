import anthropic
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course content and retrieving course outlines.

Tool Usage Guidelines:

1. **Content Search Tool** (search_course_content):
   - Use **only** for questions about specific course content or detailed educational materials
   - **Up to 2 tool rounds per query** - You can search, analyze results, then search again if needed
   - Synthesize search results into accurate, fact-based responses
   - If search yields no results, state this clearly without offering alternatives

2. **Course Outline Tool** (get_course_outline):
   - Use when users ask about course structure, outline, lessons, or "what's covered"
   - Returns: course title, course link, and complete list of lessons with their numbers and titles
   - Use this tool for questions like "What lessons are in...", "Show me the outline of...", or "What's covered in..."

3. **Multi-Step Reasoning**:
   - You can use tools multiple times (up to 2 rounds) to gather comprehensive information
   - Example workflow:
     * Round 1: Use get_course_outline to find lesson titles
     * Round 2: Use search_course_content with specific lesson information
   - Another example:
     * Round 1: Search for a topic in one course
     * Round 2: Search for the same topic in another course for comparison
   - Always synthesize information from all searches into a cohesive final answer

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool, then present the complete course structure
- **Course-specific content questions**: Use search_course_content; if more context needed, search again
- **Multi-step queries requiring comparison or multiple sources**:
  - First search: Identify relevant courses/lessons or gather initial context
  - Second search: Dive deeper into specific content or compare with other sources
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or round numbers
 - Do not mention "based on the search results", "in round 1", or "after searching twice"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with iterative tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS of sequential tool execution, allowing Claude
        to chain tools for complex multi-step queries.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # 1. Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # 2. Initialize messages array
        messages = [{"role": "user", "content": query}]

        # 3. Prepare base API parameters
        base_api_params = {
            **self.base_params,
            "system": system_content
        }

        # 4. Add tools if available
        if tools:
            base_api_params["tools"] = tools
            base_api_params["tool_choice"] = {"type": "auto"}

        # 5. ITERATIVE LOOP for sequential tool calling
        current_response = None
        for round_num in range(self.max_tool_rounds):
            logger.debug(f"Tool round {round_num + 1}/{self.max_tool_rounds}")

            # 5a. Make API call with current messages (pass a copy to avoid mutation issues)
            try:
                api_params = {**base_api_params, "messages": list(messages)}
                logger.debug(f"Making Claude API call with {len(messages)} message(s)")
                current_response = self.client.messages.create(**api_params)
                logger.debug(f"Response stop_reason: {current_response.stop_reason}")
            except Exception as e:
                logger.error(f"Claude API error in round {round_num + 1}: {e}")
                if round_num == 0:
                    raise  # First call failed - propagate error
                else:
                    return "I encountered an error while processing your request."

            # 5b. TERMINATION CHECK: No tool use requested
            if current_response.stop_reason != "tool_use":
                logger.debug("Claude provided final answer without requesting tools")
                break

            # 5c. TERMINATION CHECK: No tool manager available
            if not tool_manager:
                logger.warning("Tools requested but no tool_manager available")
                break

            # 5d. Execute all requested tools
            tool_results = []
            all_tools_failed = True

            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    logger.debug(f"Executing tool: {content_block.name}")
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                    logger.debug(f"Tool result length: {len(tool_result) if tool_result else 0} characters")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })

                    # Check if tool returned "no results" (should terminate)
                    # Errors should be passed to Claude for explanation (don't terminate)
                    is_no_results = tool_result and ("no " in tool_result.lower()[:10] or "not found" in tool_result.lower())
                    if not is_no_results:
                        all_tools_failed = False

            # 5e. TERMINATION CHECK: All tools failed
            if all_tools_failed and tool_results:
                logger.error("All tools failed - terminating loop")
                break

            # 5f. Append messages for next iteration
            messages.append({
                "role": "assistant",
                "content": current_response.content
            })

            if tool_results:
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

            logger.debug(f"Round {round_num + 1} complete. Total messages: {len(messages)}")
        else:
            # 6. Handle max rounds exceeded (only if loop completed without break)
            if current_response and current_response.stop_reason == "tool_use":
                logger.warning(f"Max tool rounds ({self.max_tool_rounds}) reached. Making final call without tools.")
                # Make final call WITHOUT tools to force text response
                final_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": system_content
                }

                try:
                    current_response = self.client.messages.create(**final_params)
                except Exception as e:
                    logger.error(f"Final API call failed: {e}")
                    return "I gathered information but encountered an error generating the final response."

        # 7. Extract and return final text
        if not current_response or not current_response.content:
            logger.error("No response content received")
            return "I apologize, but I couldn't generate a response."

        # Find text content block
        for content_block in current_response.content:
            if hasattr(content_block, 'text'):
                return content_block.text

        # Fallback if no text block found
        logger.warning("No text content in final response")
        return "I couldn't generate a proper response."