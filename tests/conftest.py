"""Pytest configuration and shared fixtures for testing suite."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
import tempfile
import os

from main import (
    ContentCreationWorkflow, 
    ContentRequest, 
    ContentType, 
    ResearchData, 
    ContentPlan, 
    ContentDraft, 
    ContentAnalysis
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_ollama_llm():
    """Mock Ollama LLM for testing without actual model calls."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock()
    mock_llm.astream = AsyncMock()
    return mock_llm


@pytest.fixture
def sample_content_request():
    """Sample content request for testing."""
    return ContentRequest(
        topic="Test Topic",
        content_type=ContentType.BLOG_POST,
        target_audience="Test audience",
        word_count=500,
        tone="professional",
        keywords=["test", "sample"],
        special_requirements="Test requirements"
    )


@pytest.fixture
def sample_research_data():
    """Sample research data for testing."""
    return ResearchData(
        sources=["https://example.com/1", "https://example.com/2"],
        key_facts=["Fact 1", "Fact 2", "Fact 3"],
        statistics=["50% increase", "2x improvement"],
        quotes=["Expert quote here", "Another expert opinion"],
        related_topics=["Related topic 1", "Related topic 2"]
    )


@pytest.fixture
def sample_content_plan():
    """Sample content plan for testing."""
    return ContentPlan(
        title="Test Article Title",
        outline=["Introduction", "Main Point 1", "Main Point 2", "Conclusion"],
        key_points=["Key point 1", "Key point 2", "Key point 3"],
        target_keywords=["test", "article", "sample"],
        estimated_length=500
    )


@pytest.fixture
def sample_content_draft():
    """Sample content draft for testing."""
    return ContentDraft(
        title="Test Article Title",
        content="This is a test article content with multiple paragraphs.\n\nSecond paragraph here.",
        word_count=15,
        reading_time=1
    )


@pytest.fixture
def sample_content_analysis():
    """Sample content analysis for testing."""
    return ContentAnalysis(
        readability_score=65.0,
        grade_level=8.0,
        keyword_density={"test": 2.5, "article": 1.8},
        suggestions=["Add more examples", "Improve readability"]
    )


@pytest.fixture
def mock_workflow():
    """Mock ContentCreationWorkflow for testing."""
    with patch('main.ChatOllama') as mock_chat:
        mock_chat.return_value = AsyncMock()
        workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
        return workflow


@pytest.fixture
def mock_web_search():
    """Mock web search results."""
    return [
        {
            "title": "Test Result 1",
            "url": "https://example.com/1",
            "snippet": "This is a test search result snippet 1"
        },
        {
            "title": "Test Result 2", 
            "url": "https://example.com/2",
            "snippet": "This is a test search result snippet 2"
        }
    ]


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override outputs directory for tests
        original_cwd = os.getcwd()
        test_outputs = os.path.join(temp_dir, "outputs")
        os.makedirs(test_outputs, exist_ok=True)
        yield test_outputs


@pytest.fixture
def mock_content_state():
    """Mock complete content creation state."""
    return {
        "request": ContentRequest(
            topic="AI in Healthcare",
            content_type=ContentType.ARTICLE,
            target_audience="Healthcare professionals",
            word_count=1000,
            tone="professional",
            keywords=["AI", "healthcare", "medical"],
            special_requirements=""
        ),
        "research_data": ResearchData(
            sources=["https://medical-ai.com", "https://health-tech.org"],
            key_facts=["AI reduces diagnosis time", "Machine learning improves accuracy"],
            statistics=["30% faster diagnosis", "95% accuracy rate"],
            quotes=["AI is transforming healthcare", "The future is now"],
            related_topics=["Medical imaging", "Predictive analytics"]
        ),
        "content_plan": ContentPlan(
            title="AI Revolutionizing Healthcare",
            outline=["Introduction", "Current Applications", "Benefits", "Challenges", "Future"],
            key_points=["Diagnosis improvement", "Cost reduction", "Patient outcomes"],
            target_keywords=["AI healthcare", "medical technology", "patient care"],
            estimated_length=1000
        ),
        "draft": ContentDraft(
            title="AI Revolutionizing Healthcare",
            content="# AI Revolutionizing Healthcare\n\nArtificial Intelligence is transforming healthcare...",
            word_count=1000,
            reading_time=4
        ),
        "analysis": ContentAnalysis(
            readability_score=72.0,
            grade_level=9.0,
            keyword_density={"AI": 3.2, "healthcare": 2.8, "medical": 2.1},
            suggestions=["Add more examples", "Include statistics"]
        ),
        "final_content": "# AI Revolutionizing Healthcare\n\nFinal polished content here...",
        "feedback_history": ["Initial draft created", "SEO optimized"],
        "revision_count": 1,
        "metadata": {
            "created_at": "2024-01-01T10:00:00",
            "total_time": 180,
            "model_used": "llama3.1:8b",
            "seo_score": 85
        }
    }


class MockLLMResponse:
    """Mock LLM response for testing."""
    
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self):
        return self.content


@pytest.fixture
def mock_llm_responses():
    """Mock LLM responses for different agents."""
    return {
        "research": MockLLMResponse("""
            Based on research, here are key findings about the topic:
            - Key fact 1: Important information discovered
            - Key fact 2: Another crucial insight
            - Statistics: 75% improvement shown in studies
            - Expert quote: "This is a significant development"
        """),
        "planning": MockLLMResponse("""
            Content Plan:
            Title: Comprehensive Guide to the Topic
            Outline:
            1. Introduction and Overview
            2. Key Benefits and Applications
            3. Implementation Strategies
            4. Case Studies and Examples
            5. Conclusion and Next Steps
            
            Target Keywords: primary, secondary, tertiary
            Estimated Length: 1200 words
        """),
        "writing": MockLLMResponse("""
            # Comprehensive Guide to the Topic
            
            ## Introduction
            This article explores the important aspects of our topic...
            
            ## Key Benefits
            The primary advantages include:
            - Benefit 1: Significant improvement in efficiency
            - Benefit 2: Cost reduction of up to 30%
            - Benefit 3: Enhanced user experience
            
            ## Conclusion
            In conclusion, this topic represents a major opportunity...
        """),
        "editing": MockLLMResponse("""
            # Comprehensive Guide to the Topic
            
            ## Introduction
            This article comprehensively explores the critical aspects of our topic...
            
            ## Key Benefits  
            The primary advantages include:
            - **Enhanced Efficiency**: Significant improvement in operational efficiency
            - **Cost Reduction**: Demonstrable cost reduction of up to 30%
            - **User Experience**: Substantially enhanced user experience
            
            ## Conclusion
            In conclusion, this topic represents a transformative opportunity...
        """),
        "seo": MockLLMResponse("""
            SEO Analysis Complete:
            - Target keywords properly distributed
            - Title optimized for search engines
            - Meta description suggestions added
            - Internal linking opportunities identified
            - Content length appropriate for topic
            
            SEO Score: 88/100
        """),
        "qa": MockLLMResponse("""
            Quality Assurance Review Complete:
            - Grammar and spelling checked
            - Factual accuracy verified
            - Tone consistency maintained
            - Call-to-action added
            - Final formatting applied
            
            Content approved for publication.
        """)
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["OLLAMA_MODEL"] = "test-model"
    os.environ["OLLAMA_BASE_URL"] = "http://test:11434"
    yield
    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]