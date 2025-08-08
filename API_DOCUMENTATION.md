# API Documentation - Content Creation Multi-Agent System

This document provides comprehensive API documentation for the Content Creation Multi-Agent System, including class interfaces, function signatures, and usage examples.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Agent Classes](#agent-classes)
3. [Security API](#security-api)
4. [Resilience API](#resilience-api)
5. [Tools API](#tools-api)
6. [Web Interface API](#web-interface-api)
7. [Monitoring API](#monitoring-api)
8. [Examples](#examples)

---

## Core Classes

### ContentRequest

Represents a content creation request with all necessary parameters.

```python
@dataclass
class ContentRequest:
    topic: str                           # The main topic/subject for content
    content_type: ContentType            # Type of content to generate
    target_audience: str                 # Intended audience description
    word_count: int                      # Desired word count
    tone: str = "professional"           # Writing tone and style
    keywords: List[str] = None           # SEO keywords to include
    special_requirements: str = ""       # Additional requirements/constraints
```

**Usage Example:**
```python
request = ContentRequest(
    topic="The Future of Artificial Intelligence in Healthcare",
    content_type=ContentType.BLOG_POST,
    target_audience="Healthcare professionals and tech enthusiasts",
    word_count=1500,
    tone="informative and engaging",
    keywords=["AI", "healthcare", "medical technology", "innovation"],
    special_requirements="Include recent statistics and real-world case studies"
)
```

### ContentType

Enumeration of supported content types.

```python
class ContentType(Enum):
    BLOG_POST = "blog_post"              # Blog post format
    ARTICLE = "article"                  # Long-form article
    SOCIAL_MEDIA = "social_media"        # Short social media content
    NEWSLETTER = "newsletter"            # Newsletter format
    MARKETING_COPY = "marketing_copy"    # Marketing materials
```

### ContentCreationState

The state object that flows through the multi-agent pipeline.

```python
class ContentCreationState(TypedDict):
    request: Optional[ContentRequest]           # Original request
    research_data: Optional[ResearchData]       # Research findings
    content_plan: Optional[ContentPlan]         # Content structure plan
    draft: Optional[ContentDraft]               # Generated content draft
    analysis: Optional[ContentAnalysis]         # Content analysis results
    final_content: Optional[str]                # Final polished content
    feedback_history: List[str]                 # Process feedback
    revision_count: int                         # Number of revisions
    metadata: Dict[str, Any]                    # Additional metadata
```

### ContentCreationWorkflow

Main orchestrator class for the multi-agent content creation system.

```python
class ContentCreationWorkflow:
    def __init__(self, model_name: str = "llama3.1:8b", 
                 base_url: str = "http://localhost:11434"):
        """Initialize the workflow with Ollama configuration."""
    
    async def create_content(self, content_request: ContentRequest) -> ContentCreationState:
        """Execute the complete content creation workflow."""
```

**Usage Example:**
```python
workflow = ContentCreationWorkflow(
    model_name="llama3.1:8b",
    base_url="http://localhost:11434"
)

result = await workflow.create_content(content_request)
print(result["final_content"])
```

---

## Agent Classes

### ResearchAgent

Handles information gathering and research data collection.

```python
class ResearchAgent:
    def __init__(self, llm: ChatOllama):
        """Initialize with language model."""
    
    async def research(self, state: ContentCreationState) -> ContentCreationState:
        """Conduct research on the given topic."""
```

**Capabilities:**
- Web search integration
- Data extraction and validation
- Source compilation
- Fact verification

### PlanningAgent

Creates content structure and strategic planning.

```python
class PlanningAgent:
    def __init__(self, llm: ChatOllama):
        """Initialize with language model."""
    
    async def plan_content(self, state: ContentCreationState) -> ContentCreationState:
        """Create detailed content plan and structure."""
```

**Capabilities:**
- Content outline creation
- Keyword strategy planning
- Length estimation
- Structure optimization

### WriterAgent

Generates the initial content draft.

```python
class WriterAgent:
    def __init__(self, llm: ChatOllama):
        """Initialize with language model."""
    
    async def write_content(self, state: ContentCreationState) -> ContentCreationState:
        """Generate content based on research and plan."""
```

**Capabilities:**
- Content generation
- Style adaptation
- Research integration
- Draft creation

### EditorAgent

Refines and improves content quality.

```python
class EditorAgent:
    def __init__(self, llm: ChatOllama):
        """Initialize with language model."""
    
    async def edit_content(self, state: ContentCreationState) -> ContentCreationState:
        """Edit and improve content quality."""
```

**Capabilities:**
- Readability improvement
- Grammar and style correction
- Content enhancement
- Quality analysis

### SEOAgent

Optimizes content for search engines.

```python
class SEOAgent:
    def __init__(self, llm: ChatOllama):
        """Initialize with language model."""
    
    async def optimize_seo(self, state: ContentCreationState) -> ContentCreationState:
        """Optimize content for SEO."""
```

**Capabilities:**
- Keyword optimization
- SEO scoring
- Meta description generation
- Search optimization

### QualityAssuranceAgent

Performs final validation and quality checks.

```python
class QualityAssuranceAgent:
    def __init__(self, llm: ChatOllama):
        """Initialize with language model."""
    
    async def finalize_content(self, state: ContentCreationState) -> ContentCreationState:
        """Perform final quality assurance and save content."""
```

**Capabilities:**
- Final quality validation
- Content formatting
- File output management
- Process completion

---

## Security API

### Input Validation

```python
from security_utils import validate_content_request, validate_and_sanitize_text

def validate_content_request(request: ContentRequest) -> ValidationResult:
    """Validate and sanitize content request."""

def validate_and_sanitize_text(text: str, content_type: str = "general") -> ValidationResult:
    """Validate and sanitize text content."""
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    is_valid: bool                          # Whether input passes validation
    sanitized_input: Optional[Any] = None   # Cleaned input data
    errors: List[str] = None                # Validation errors
    warnings: List[str] = None              # Validation warnings
    threat_level: str = "none"              # Threat level: none, low, medium, high
```

### Content Filtering

```python
from security_utils import filter_content

def filter_content(content: str) -> Tuple[str, List[str]]:
    """Filter inappropriate content and return cleaned version."""
    # Returns: (filtered_content, list_of_filtered_items)
```

### Security Logging

```python
from security_utils import SecurityLogger

class SecurityLogger:
    def log_security_event(self, event_type: SecurityThreat, message: str, 
                          context: Dict[str, Any] = None, severity: str = "warning"):
        """Log security event with structured data."""
```

---

## Resilience API

### Retry Decorator

Add automatic retry logic with configurable backoff strategies.

```python
from resilience_utils import retry_with_backoff, RetryConfig, RetryStrategy

@retry_with_backoff(RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    timeout=30.0
))
async def my_function():
    """Function with retry logic."""
    pass
```

### Timeout Decorator

Add timeout protection to functions.

```python
from resilience_utils import timeout

@timeout(30.0)  # 30 second timeout
async def my_function():
    """Function with timeout protection."""
    pass
```

### Circuit Breaker

Implement circuit breaker pattern for service protection.

```python
from resilience_utils import circuit_breaker, CircuitBreakerConfig

@circuit_breaker("my_service", CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=3,
    timeout=60.0
))
async def my_function():
    """Function with circuit breaker protection."""
    pass
```

### Performance Monitoring

```python
from resilience_utils import get_performance_stats, get_system_health

# Get performance statistics
stats = get_performance_stats("operation_name", time_window_minutes=60)

# Get overall system health
health = get_system_health()
```

---

## Tools API

### Web Search Tool

```python
@tool
def web_search_tool(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Performs web search to gather information for content research."""
```

### Content Analysis Tool

```python
@tool
def content_analysis_tool(content: str) -> Dict[str, Any]:
    """Analyzes content for readability, SEO, and quality metrics."""
```

### SEO Optimization Tool

```python
@tool
def seo_optimization_tool(content: str, target_keywords: List[str]) -> Dict[str, Any]:
    """Provides SEO optimization suggestions for content."""
```

### Save Content Tool

```python
@tool
def save_content_tool(content: str, filename: str) -> Dict[str, str]:
    """Saves content to a file."""
```

---

## Web Interface API

### Streamlit Application

The web interface provides a user-friendly way to interact with the system.

```bash
# Launch web interface
streamlit run streamlit_app.py
```

**Key Features:**
- Interactive content creation form
- Real-time progress tracking
- Content preview and download
- System health monitoring
- Generation history

### Session State Management

```python
# Initialize session state
init_session_state()

# Access session state
st.session_state.workflow
st.session_state.generation_history
st.session_state.current_result
```

---

## Monitoring API

### System Health

```python
from resilience_utils import get_system_health

health = get_system_health()
# Returns:
{
    "timestamp": "2024-01-01T10:00:00",
    "circuit_breakers": {
        "service_name": {
            "state": "closed",
            "failure_count": 0,
            "success_count": 10
        }
    },
    "performance_stats": {
        "operation_name": {
            "total_operations": 50,
            "success_rate": 0.96,
            "average_duration": 2.5
        }
    }
}
```

### Performance Statistics

```python
from resilience_utils import get_performance_stats

stats = get_performance_stats("content_creation", 60)
# Returns:
{
    "operation": "content_creation",
    "time_window_minutes": 60,
    "total_operations": 25,
    "successful_operations": 24,
    "failed_operations": 1,
    "success_rate": 0.96,
    "average_duration": 3.2,
    "min_duration": 1.5,
    "max_duration": 5.8,
    "average_attempts": 1.2,
    "error_types": {
        "TimeoutError": 1
    }
}
```

---

## Examples

### Basic Content Generation

```python
import asyncio
from main import ContentCreationWorkflow, ContentRequest, ContentType

async def generate_blog_post():
    # Initialize workflow
    workflow = ContentCreationWorkflow()
    
    # Create request
    request = ContentRequest(
        topic="10 Benefits of Remote Work",
        content_type=ContentType.BLOG_POST,
        target_audience="Remote workers and managers",
        word_count=1200,
        keywords=["remote work", "benefits", "productivity"]
    )
    
    # Generate content
    result = await workflow.create_content(request)
    
    # Access results
    print("Title:", result["draft"].title)
    print("Word Count:", result["draft"].word_count)
    print("SEO Score:", result["metadata"].get("seo_score"))
    print("\nContent:")
    print(result["final_content"])

# Run
asyncio.run(generate_blog_post())
```

### Security Validation

```python
from security_utils import validate_content_request, ValidationError
from main import ContentRequest, ContentType

# Create request
request = ContentRequest(
    topic="<script>alert('test')</script>AI in Healthcare",  # Malicious input
    content_type=ContentType.ARTICLE,
    target_audience="Healthcare professionals",
    word_count=1000
)

# Validate request
validation = validate_content_request(request)

if not validation.is_valid:
    print("Validation failed:")
    for error in validation.errors:
        print(f"- {error}")
    print(f"Threat level: {validation.threat_level}")
else:
    # Use sanitized request
    clean_request = validation.sanitized_input
    print("Request validated and sanitized")
```

### Performance Monitoring

```python
from resilience_utils import PerformanceMonitor, get_performance_stats
import time

# Monitor function performance
monitor = PerformanceMonitor()

def monitored_function():
    start_time = datetime.now()
    try:
        # Your function logic here
        time.sleep(2)
        success = True
        error = None
    except Exception as e:
        success = False
        error = e
    finally:
        end_time = datetime.now()
        monitor.record_operation(
            "my_function", start_time, end_time, success, 1, error
        )

# Run function
monitored_function()

# Get statistics
stats = get_performance_stats("my_function", 60)
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average duration: {stats['average_duration']:.2f}s")
```

### Custom Agent Creation

```python
from langchain_core.messages import HumanMessage
from typing import Dict, Any

class CustomAgent:
    def __init__(self, llm):
        self.llm = llm
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Custom agent processing logic."""
        
        # Get request details
        request = state["request"]
        
        # Create prompt
        prompt = f"""
        You are a custom content agent.
        Topic: {request.topic}
        Audience: {request.target_audience}
        
        Perform your custom processing here.
        """
        
        # Call LLM
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        # Process response and update state
        state["custom_data"] = response.content
        
        return state
```

### Error Handling

```python
import asyncio
from main import ContentCreationWorkflow, ValidationError

async def safe_content_generation(request):
    workflow = ContentCreationWorkflow()
    
    try:
        result = await workflow.create_content(request)
        return result
        
    except ValidationError as e:
        print(f"Input validation failed: {e}")
        return None
        
    except asyncio.TimeoutError:
        print("Content generation timed out")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

---

## Error Codes and Responses

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `VALIDATION_ERROR` | Input validation failed | Check input parameters and sanitize |
| `TIMEOUT_ERROR` | Operation timed out | Increase timeout or reduce complexity |
| `CIRCUIT_BREAKER_OPEN` | Service temporarily unavailable | Wait for circuit breaker reset |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait before retrying |
| `OLLAMA_CONNECTION_ERROR` | Cannot connect to Ollama | Check Ollama server status |

### HTTP Status Codes (Web Interface)

| Code | Description |
|------|-------------|
| `200` | Success |
| `400` | Bad Request - Invalid input |
| `429` | Too Many Requests |
| `500` | Internal Server Error |
| `503` | Service Temporarily Unavailable |

---

## Current Implementation Files

### Core API Files
- **main.py** - Main ContentCreationWorkflow class and orchestration logic
- **types_shared.py** - All TypedDict definitions, ContentRequest, ContentType, and state structures
- **demo.py** - CLI interface for testing and demonstration with interactive scenarios

### Agent API Implementations
- **agents/research_agent.py** - ResearchAgent class with web search integration
- **agents/planning_agent.py** - PlanningAgent class with content structure creation
- **agents/writer_agent.py** - WriterAgent class with Ollama LLM content generation
- **agents/editor_agent.py** - EditorAgent class with NLTK readability analysis
- **agents/seo_agent.py** - SEOAgent class with keyword optimization
- **agents/qa_agent.py** - QualityAssuranceAgent class with final validation

### Security and Resilience APIs
- **security_utils.py** - All security validation, filtering, and logging functions
- **resilience_utils.py** - Retry decorators, circuit breakers, timeout protection, monitoring

### Web Interface API
- **streamlit_app.py** - Streamlit web application with interactive UI components

### Testing APIs
- **test_agents.py** (root) - Main test suite with comprehensive API testing
- **tests/test_agents.py** - Individual agent API testing
- **tests/test_integration.py** - Multi-agent workflow API testing
- **tests/test_e2e.py** - End-to-end API validation
- **tests/test_tools.py** - Tool API testing
- **tests/conftest.py** - PyTest fixtures and test configuration

## API Configuration

### Environment Variables
```env
OLLAMA_MODEL=llama3.1:8b          # Default model for all agents
OLLAMA_BASE_URL=http://localhost:11434  # Ollama server endpoint
OLLAMA_TEMPERATURE=0.7            # LLM temperature setting
OLLAMA_TOP_P=0.9                  # Top-p sampling parameter
OLLAMA_NUM_PREDICT=4096           # Maximum tokens to predict
```

### API Configuration Files
- **requirements.txt** - Python dependencies for API functionality
- **.env.sample** - Environment variables template
- **pytest.ini** - Test configuration for API testing
- **.coveragerc** - Code coverage configuration

## API Usage Patterns

### Async/Await Pattern
All agent APIs are asynchronous for optimal performance:
```python
async def main():
    workflow = ContentCreationWorkflow()
    result = await workflow.create_content(request)
    return result
```

### State Management Pattern
All agents follow the ContentCreationState pattern:
```python
async def agent_function(state: ContentCreationState) -> ContentCreationState:
    # Process current state
    # Update with new data
    # Return enhanced state
    return state
```

### Error Handling Pattern
All APIs implement comprehensive error handling:
```python
try:
    result = await api_function()
except ValidationError as e:
    handle_validation_error(e)
except TimeoutError as e:
    handle_timeout_error(e)
```

### Tool Integration Pattern
All tools use the @tool decorator:
```python
@tool
def tool_function(input: str) -> Dict[str, Any]:
    """Tool description for LLM usage"""
    return {"result": processed_data}
```

For more detailed examples and advanced usage patterns, refer to the test files in the `tests/` directory and the main documentation in `README.md`.