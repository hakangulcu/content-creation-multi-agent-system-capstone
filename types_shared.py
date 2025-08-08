"""
Shared types and classes for the Content Creation Multi-Agent System.

This module contains common data structures to avoid circular imports.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """Content type enumeration."""
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    SOCIAL_MEDIA = "social_media"
    NEWSLETTER = "newsletter"
    MARKETING_COPY = "marketing_copy"


@dataclass
class ContentRequest:
    """Content creation request structure."""
    topic: str
    content_type: ContentType
    target_audience: str
    word_count: int
    tone: str = "professional"
    keywords: List[str] = None
    special_requirements: str = ""


@dataclass
class ResearchData:
    """Research data structure."""
    sources: List[str]
    key_facts: List[str]
    statistics: List[str]
    quotes: List[str]
    related_topics: List[str]


@dataclass
class ContentPlan:
    """Content plan structure."""
    title: str
    outline: List[str]
    key_points: List[str]
    target_keywords: List[str]
    estimated_length: int


@dataclass
class ContentDraft:
    """Content draft structure."""
    title: str
    content: str
    word_count: int
    reading_time: int


@dataclass
class ContentAnalysis:
    """Content analysis structure."""
    readability_score: float
    grade_level: float
    keyword_density: Dict[str, float]
    suggestions: List[str]