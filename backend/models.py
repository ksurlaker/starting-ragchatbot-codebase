from pydantic import BaseModel


class Lesson(BaseModel):
    """Represents a lesson within a course"""

    lesson_number: int  # Sequential lesson number (1, 2, 3, etc.)
    title: str  # Lesson title
    lesson_link: str | None = None  # URL link to the lesson


class Course(BaseModel):
    """Represents a complete course with its lessons"""

    title: str  # Full course title (used as unique identifier)
    course_link: str | None = None  # URL link to the course
    instructor: str | None = None  # Course instructor name (optional metadata)
    lessons: list[Lesson] = []  # List of lessons in this course


class CourseChunk(BaseModel):
    """Represents a text chunk from a course for vector storage"""

    content: str  # The actual text content
    course_title: str  # Which course this chunk belongs to
    lesson_number: int | None = None  # Which lesson this chunk is from
    chunk_index: int  # Position of this chunk in the document


class Source(BaseModel):
    """Represents a source citation with optional link"""

    text: str  # Display text like "Course Title - Lesson 1"
    url: str | None = None  # Lesson link (None if not available)
    course_title: str  # Course title for metadata
    lesson_number: int | None = None  # Lesson number for metadata
