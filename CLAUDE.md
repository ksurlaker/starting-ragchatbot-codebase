# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about course materials. The system uses ChromaDB for vector storage, Anthropic's Claude API for generation, and implements a tool-calling pattern where Claude autonomously decides when to search documents.

## Development Commands

**IMPORTANT: Always use `uv` for package management and running Python commands. Do not use `pip` directly.**

### Setup
```bash
# Install dependencies (requires uv package manager)
uv sync

# Create .env file with:
ANTHROPIC_API_KEY=your_key_here
```

### Running the Application
```bash
# Quick start (creates docs/ directory and starts server)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app runs at `http://localhost:8000` (frontend) and `http://localhost:8000/docs` (API docs).

### Code Quality
```bash
# Install dev dependencies (includes black and ruff)
uv sync --extra dev

# Run all quality checks (formatting + linting)
./quality.sh check

# Auto-fix formatting and lint issues
./quality.sh fix

# Run only formatter (black)
./quality.sh format

# Run only linter (ruff)
./quality.sh lint
```

## Architecture

### Request Flow
The system uses a multi-step RAG pipeline with tool calling:

1. **Frontend** (`frontend/script.js`) - Vanilla JS sends POST to `/api/query`
2. **FastAPI Endpoint** (`backend/app.py:56`) - Receives query + session_id
3. **RAG Orchestrator** (`backend/rag_system.py:102`) - Coordinates all components
4. **Session Manager** (`backend/session_manager.py`) - Maintains conversation history (max 2 exchanges by default)
5. **AI Generator** (`backend/ai_generator.py:43`) - Makes first Claude API call with tool definitions
6. **Tool Execution** (if Claude requests it):
   - **Tool Manager** (`backend/search_tools.py`) - Executes CourseSearchTool
   - **Vector Store** (`backend/vector_store.py:61`) - Queries ChromaDB (2 collections: `course_catalog` for metadata, `course_content` for chunks)
   - Returns top 5 relevant chunks with sources
7. **Final Generation** (`backend/ai_generator.py:127`) - Second Claude API call with retrieved context
8. **Response** - Returns answer + sources to frontend

### Key Components

**Two ChromaDB Collections:**
- `course_catalog`: Course metadata (for semantic course name resolution)
- `course_content`: Text chunks with metadata (course_title, lesson_number, chunk_index)

**Tool Calling Pattern:**
- Claude receives tool definitions on first API call
- If `stop_reason == "tool_use"`, the system executes `search_course_content` tool
- Tool results are added to conversation, then a second API call generates the final answer
- This allows Claude to decide whether to search or answer from conversation context

**Session Management:**
- In-memory conversation history (not persistent across server restarts)
- Each session stores user/assistant message pairs
- History is formatted as "Role: content" strings and included in system prompt

### Data Models

All models in `backend/models.py`:
- `Course`: title, course_link, instructor, lessons[]
- `Lesson`: lesson_number, title, lesson_link
- `CourseChunk`: content, course_title, lesson_number, chunk_index
- `SearchResults`: documents[], metadata[], distances[], error (in `vector_store.py`)

API models (in `app.py`):
- `QueryRequest`: query, session_id
- `QueryResponse`: answer, sources[], session_id

### Configuration

All settings in `backend/config.py`:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (sentence-transformers)
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results per query
- `MAX_HISTORY`: 2 conversation exchanges (4 messages total)
- `CHROMA_PATH`: "./chroma_db"

## Document Processing

Course materials are stored as `.txt` files in `docs/` directory with naming convention:
- `course1_script.txt`, `course2_script.txt`, etc.

The `DocumentProcessor` (`backend/document_processor.py`):
- Parses course metadata from structured text format
- Chunks content using overlapping windows (800 chars, 100 overlap)
- Stores chunks in ChromaDB with metadata for filtering

## Important Implementation Details

**Vector Search with Filtering:**
- Course names are resolved semantically (user says "Python" → matches "Introduction to Python Programming")
- Filters are built as ChromaDB `where` clauses combining course_title and lesson_number
- See `vector_store.py:102` for `_resolve_course_name()` and `vector_store.py:118` for `_build_filter()`

**Source Tracking:**
- Tool execution stores sources in `ToolManager.last_sources`
- Sources are reset after each query to prevent cross-contamination
- Frontend displays sources in collapsible sections

**Frontend Architecture:**
- No build step - vanilla HTML/CSS/JS served by FastAPI static files
- Markdown parsing using `marked.js` library
- Session ID persists in browser for conversation continuity

## File References Convention

When discussing code, use the pattern `file_path:line_number` for easy navigation. Example: "Sessions are created in `session_manager.py:18`"
