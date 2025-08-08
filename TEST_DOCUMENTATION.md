# Test Documentation

## Overview

The Content Creation Multi-Agent System uses a comprehensive test suite with PyTest to ensure reliability, performance, and correctness across all components. The testing framework includes unit tests, integration tests, end-to-end tests, and tool validation tests.

## Test Structure

### Test Organization
```
tests/
├── conftest.py           # Shared fixtures and configuration
├── test_agents.py        # Individual agent unit tests
├── test_integration.py   # Multi-agent workflow tests
├── test_e2e.py          # Complete pipeline tests
├── test_tools.py        # External tool integration tests
└── __init__.py          # Package initialization

test_agents.py           # Root-level simple integration test
pytest.ini               # PyTest configuration
```

### Test Categories

#### 1. Unit Tests (`@pytest.mark.unit`)
- **Purpose**: Test individual agent functionality in isolation
- **Coverage**: 15 tests across 6 agents + error handling
- **Focus**: Agent initialization, core methods, exception handling
- **Execution Time**: Fast (~1-5s per test)

#### 2. Integration Tests (`@pytest.mark.integration`)
- **Purpose**: Test agent-to-agent interactions and state flow
- **Coverage**: 14 tests covering pipeline stages and state management
- **Focus**: Data flow, state consistency, feedback accumulation
- **Execution Time**: Medium (~5-15s per test)

#### 3. End-to-End Tests (`@pytest.mark.e2e`)
- **Purpose**: Test complete content creation workflows
- **Coverage**: 9 tests covering real-world scenarios
- **Focus**: Full pipeline execution, different content types
- **Execution Time**: Slow (~30-120s per test)

#### 4. Tool Tests (`@pytest.mark.unit`)
- **Purpose**: Test external tool integrations
- **Coverage**: 13 tests covering web search, analysis, SEO, file operations
- **Focus**: API integration, error handling, data validation
- **Execution Time**: Fast-Medium (~2-10s per test)

## Test Configuration

### PyTest Configuration (`pytest.ini`)
```ini
[tool:pytest]
minversion = 6.0
addopts = -ra -q --strict-markers --disable-warnings
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
markers =
    unit: Unit tests for individual components
    integration: Integration tests for component interactions  
    e2e: End-to-end tests for complete workflows
    slow: Tests that take longer to run
    requires_ollama: Tests that require Ollama to be running
```

### Test Environment Setup
- **Automatic Environment**: Tests run with `TESTING=true` environment variable
- **Mock LLM**: Ollama calls are mocked to avoid actual model dependencies
- **Temporary Directories**: File operations use temporary directories
- **Async Support**: Full async/await support for agent testing

## Fixtures and Mocking

### Core Fixtures (`conftest.py`)

#### Mock LLM and Workflow
```python
@pytest.fixture
def mock_ollama_llm():
    """Mock Ollama LLM for testing without actual model calls."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock()
    return mock_llm

@pytest.fixture
def mock_workflow():
    """Mock ContentCreationWorkflow for testing."""
    with patch('main.ChatOllama') as mock_chat:
        workflow = ContentCreationWorkflow(model_name="test-model")
        return workflow
```

#### Sample Data Fixtures
```python
@pytest.fixture
def sample_content_request():
    return ContentRequest(
        topic="Test Topic",
        content_type=ContentType.BLOG_POST,
        target_audience="Test audience",
        word_count=500,
        tone="professional",
        keywords=["test", "sample"]
    )

@pytest.fixture
def sample_research_data():
    return ResearchData(
        sources=["https://example.com/1", "https://example.com/2"],
        key_facts=["Fact 1", "Fact 2", "Fact 3"],
        statistics=["50% increase", "2x improvement"],
        quotes=["Expert quote here"],
        related_topics=["Related topic 1", "Related topic 2"]
    )
```

#### Mock LLM Responses
```python
@pytest.fixture
def mock_llm_responses():
    """Mock realistic LLM responses for different agents."""
    return {
        "research": MockLLMResponse("Research findings with key facts..."),
        "planning": MockLLMResponse("Content plan with title and outline..."),
        "writing": MockLLMResponse("# Article Title\n\nContent body..."),
        "editing": MockLLMResponse("Improved and polished content..."),
        "seo": MockLLMResponse("SEO Analysis: Score 88/100..."),
        "qa": MockLLMResponse("Quality check passed, ready for publication...")
    }
```

## Test Examples and Results

### Unit Test Example: Research Agent
```python
@pytest.mark.unit
class TestResearchAgent:
    @pytest.mark.asyncio
    async def test_research_method_with_valid_state(self, research_agent, sample_content_request):
        # Setup
        initial_state = {
            "request": sample_content_request,
            "research_data": None,
            "metadata": {}
        }
        
        # Execute
        result = await research_agent.research(initial_state)
        
        # Assertions
        assert "research_data" in result
        assert result["research_data"] is not None
        assert isinstance(result["research_data"].sources, list)
        assert len(result["research_data"].key_facts) > 0
```

**Expected Output:**
```
[PASSED] - Research agent processes request correctly
[PASSED] - Research data structure is valid
[PASSED] - Sources and facts are populated
```

### Integration Test Example: Pipeline Flow
```python
@pytest.mark.integration
class TestAgentPipeline:
    @pytest.mark.asyncio
    async def test_research_to_planning_flow(self, mock_workflow, sample_content_request):
        # Execute research phase
        research_result = await mock_workflow.research_agent.research({
            "request": sample_content_request
        })
        
        # Execute planning phase with research output
        planning_result = await mock_workflow.planning_agent.plan_content(research_result)
        
        # Verify data flow
        assert planning_result["content_plan"].title is not None
        assert len(planning_result["content_plan"].outline) > 0
        assert planning_result["content_plan"].target_keywords
```

**Expected Output:**
```
[PASSED] - Research data flows to planning agent
[PASSED] - Content plan structure is created
[PASSED] - Keywords are extracted from research
```

### E2E Test Example: Complete Workflow
```python
@pytest.mark.e2e
class TestCompleteWorkflow:
    @pytest.mark.asyncio
    async def test_full_content_creation_pipeline(self, mock_workflow):
        request = ContentRequest(
            topic="Benefits of Exercise",
            content_type=ContentType.ARTICLE,
            word_count=500
        )
        
        # Execute full workflow
        result = await mock_workflow.create_content(request)
        
        # Verify complete output
        assert result["final_content"] is not None
        assert result["metadata"]["total_time"] > 0
        assert result["draft"].word_count > 0
```

**Expected Output:**
```
Testing Multi-Agent Content Creation System
==================================================
[OK] Workflow initialized successfully
Testing with topic: Benefits of Exercise
Target length: 500 words
👥 Audience: General public
--------------------------------------------------

Agent Information:
1. Research Agent: ResearchAgent
2. Planning Agent: PlanningAgent
3. Writer Agent: WriterAgent
4. Editor Agent: EditorAgent
5. SEO Agent: SEOAgent
6. QA Agent: QualityAssuranceAgent

Starting content creation workflow...

[OK] Content Creation Completed!
Final word count: 487
Reading time: 2 minutes
📁 Saved to: outputs/Benefits_of_Exercise_20250101_120000.md
SEO Score: 85
Quality Score: 92
```

## Test Execution Commands

### Run All Tests
```bash
# Basic test run
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html

# Quiet mode
pytest tests/ -q
```

### Run by Category
```bash
# Unit tests only
pytest tests/ -m unit -v

# Integration tests only  
pytest tests/ -m integration -v

# End-to-end tests only
pytest tests/ -m e2e -v

# Exclude slow tests
pytest tests/ -m "not slow" -v
```

### Run Specific Test Files
```bash
# Agent tests only
pytest tests/test_agents.py -v

# Tool tests only
pytest tests/test_tools.py -v

# Integration tests only
pytest tests/test_integration.py -v

# E2E tests only
pytest tests/test_e2e.py -v
```

### Run Simple Root-Level Test
```bash
# Quick integration test
python test_agents.py
```

## Test Results and Performance

### Typical Test Execution Results

#### Full Test Suite
```
======================== test session starts ========================
platform darwin -- Python 3.10.9, pytest-7.4.0
rootdir: /content-creation-multi-agent-system-capstone
configfile: pytest.ini
plugins: asyncio-0.21.0, anyio-4.9.0, cov-4.1.0
asyncio: mode=strict
collected 51 items

tests/test_agents.py::TestResearchAgent::test_research_agent_initialization PASSED [  2%]
tests/test_agents.py::TestResearchAgent::test_research_method_with_valid_state PASSED [  4%]
tests/test_agents.py::TestResearchAgent::test_research_method_exception_handling PASSED [  6%]
tests/test_agents.py::TestPlanningAgent::test_planning_agent_initialization PASSED [  8%]
tests/test_agents.py::TestPlanningAgent::test_plan_content_method PASSED [ 10%]
tests/test_agents.py::TestWriterAgent::test_writer_agent_initialization PASSED [ 12%]
tests/test_agents.py::TestWriterAgent::test_write_content_method PASSED [ 14%]
tests/test_agents.py::TestEditorAgent::test_editor_agent_initialization PASSED [ 16%]
tests/test_agents.py::TestEditorAgent::test_edit_content_method PASSED [ 18%]
tests/test_agents.py::TestSEOAgent::test_seo_agent_initialization PASSED [ 20%]
tests/test_agents.py::TestSEOAgent::test_optimize_seo_method PASSED [ 22%]
tests/test_agents.py::TestQualityAssuranceAgent::test_qa_agent_initialization PASSED [ 24%]
tests/test_agents.py::TestQualityAssuranceAgent::test_finalize_content_method PASSED [ 26%]
tests/test_agents.py::TestAgentErrorHandling::test_agent_llm_timeout PASSED [ 28%]
tests/test_agents.py::TestAgentErrorHandling::test_agent_network_error PASSED [ 30%]

tests/test_tools.py::TestWebSearchTool::test_web_search_success PASSED [ 32%]
tests/test_tools.py::TestWebSearchTool::test_web_search_empty_results PASSED [ 34%]
tests/test_tools.py::TestWebSearchTool::test_web_search_exception PASSED [ 36%]
tests/test_tools.py::TestContentAnalysisTool::test_content_analysis_success PASSED [ 38%]
tests/test_tools.py::TestContentAnalysisTool::test_content_analysis_nltk_download PASSED [ 40%]
tests/test_tools.py::TestSEOOptimizationTool::test_seo_optimization_basic PASSED [ 42%]
tests/test_tools.py::TestSEOOptimizationTool::test_seo_optimization_missing_keywords PASSED [ 44%]
tests/test_tools.py::TestSaveContentTool::test_save_content_success PASSED [ 46%]
tests/test_tools.py::TestToolIntegration::test_search_to_analysis_pipeline PASSED [ 48%]

tests/test_integration.py::TestAgentPipeline::test_research_to_planning_flow PASSED [ 50%]
tests/test_integration.py::TestAgentPipeline::test_planning_to_writing_flow PASSED [ 52%]
tests/test_integration.py::TestAgentPipeline::test_writing_to_editing_flow PASSED [ 54%]
tests/test_integration.py::TestAgentPipeline::test_editing_to_seo_flow PASSED [ 56%]
tests/test_integration.py::TestAgentPipeline::test_seo_to_qa_flow PASSED [ 58%]
tests/test_integration.py::TestWorkflowStateManagement::test_state_consistency_through_pipeline PASSED [ 60%]
tests/test_integration.py::TestWorkflowStateManagement::test_feedback_history_accumulation PASSED [ 62%]
tests/test_integration.py::TestWorkflowStateManagement::test_metadata_accumulation PASSED [ 64%]

tests/test_e2e.py::TestCompleteWorkflow::test_full_content_creation_pipeline PASSED [ 66%]
tests/test_e2e.py::TestCompleteWorkflow::test_workflow_with_different_content_types PASSED [ 68%]
tests/test_e2e.py::TestCompleteWorkflow::test_workflow_error_resilience PASSED [ 70%]
tests/test_e2e.py::TestDemoFunctionality::test_demo_content_creation_function PASSED [ 72%]
tests/test_e2e.py::TestRealWorldScenarios::test_blog_post_creation_scenario PASSED [ 74%]
tests/test_e2e.py::TestRealWorldScenarios::test_technical_article_scenario PASSED [ 76%]
tests/test_e2e.py::TestRealWorldScenarios::test_short_content_scenario PASSED [ 78%]

======================== 51 tests passed in 23.45s ========================
```

#### Test Performance Metrics
- **Unit Tests**: ~0.1-0.5s per test (fast)
- **Integration Tests**: ~1-5s per test (medium)
- **E2E Tests**: ~10-60s per test (slow)
- **Total Suite**: ~20-30s (mocked) / ~5-10min (with Ollama)

### Code Coverage Results
```
Name                    Stmts   Miss  Cover
-------------------------------------------
main.py                   245     12    95%
agents/__init__.py          8      0   100%
agents/research_agent.py   67      5    93%
agents/planning_agent.py   59      3    95%
agents/writer_agent.py     64      4    94%
agents/editor_agent.py     71      6    92%
agents/seo_agent.py        78      7    91%
agents/qa_agent.py         52      2    96%
types_shared.py            42      0   100%
resilience_utils.py        38      2    95%
security_utils.py          29      1    97%
-------------------------------------------
TOTAL                     753     42    94%
```

## Error Handling and Edge Cases

### Common Test Scenarios

#### Network/API Failures
```python
@pytest.mark.asyncio
async def test_web_search_exception(self):
    """Test web search tool handles exceptions gracefully."""
    with patch('duckduckgo_search.DDGS') as mock_ddgs:
        mock_ddgs.side_effect = Exception("Network error")
        
        result = await web_search_tool.ainvoke("test query")
        
        # Should return empty results, not crash
        assert result == []
```

#### LLM Timeout/Errors
```python
@pytest.mark.asyncio  
async def test_agent_llm_timeout(self, research_agent):
    """Test agent handles LLM timeout gracefully."""
    research_agent.llm.ainvoke.side_effect = asyncio.TimeoutError()
    
    result = await research_agent.research({"request": sample_request})
    
    # Should return fallback response
    assert "research_data" in result
    assert "error" in result["metadata"]
```

#### Invalid Input Data
```python
@pytest.mark.asyncio
async def test_agent_invalid_input(self, writer_agent):
    """Test agent handles invalid input gracefully.""" 
    invalid_state = {"invalid": "data"}
    
    result = await writer_agent.write_content(invalid_state)
    
    # Should handle gracefully with error message
    assert "error" in result["metadata"]
    assert result["draft"] is not None  # Fallback content
```

## Test Maintenance and Best Practices

### Writing New Tests

#### Test Structure Template
```python
@pytest.mark.unit  # or integration, e2e
class TestNewFeature:
    """Test cases for new feature functionality."""
    
    @pytest.fixture
    def feature_setup(self):
        """Setup specific to this feature."""
        return FeatureClass()
    
    @pytest.mark.asyncio
    async def test_feature_basic_functionality(self, feature_setup):
        """Test basic feature operation."""
        # Arrange
        input_data = create_test_data()
        
        # Act  
        result = await feature_setup.process(input_data)
        
        # Assert
        assert result is not None
        assert result.status == "success"
    
    @pytest.mark.asyncio
    async def test_feature_error_handling(self, feature_setup):
        """Test feature handles errors gracefully."""
        with pytest.raises(ExpectedError):
            await feature_setup.process(invalid_data)
```

### Test Data Management
- Use fixtures for reusable test data
- Create realistic but minimal test cases
- Mock external dependencies consistently
- Use temporary files/directories for file tests

### Performance Considerations
- Mock expensive operations (LLM calls, web requests)
- Use `@pytest.mark.slow` for tests >10 seconds
- Parallel test execution where possible
- Clean up resources after tests

### Debugging Failed Tests
```bash
# Verbose output with full traceback
pytest tests/test_agents.py::TestResearchAgent::test_research_method -v -s --tb=long

# Run single test with debugging
pytest tests/test_agents.py::TestResearchAgent::test_research_method -v -s --pdb

# Show local variables in failure output
pytest tests/test_agents.py --tb=long --showlocals
```

## Continuous Integration

### GitHub Actions Integration
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          python -c "import nltk; nltk.download('punkt')"
      - name: Run tests
        run: |
          pytest tests/ --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Local Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

#### NLTK Data Missing
```bash
# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"
```

#### Ollama Connection in Tests
```bash
# Tests should run without Ollama, but if needed:
ollama serve
ollama pull llama3.1:8b
```

#### Import Errors
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Async Test Issues
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Check pytest.ini has asyncio_mode = auto
```

### Test Environment Variables
```bash
# For testing without external services
export TESTING=true
export OLLAMA_MODEL=test-model
export OLLAMA_BASE_URL=http://test:11434

# Run tests
pytest tests/
```

This comprehensive test suite ensures the Content Creation Multi-Agent System maintains high quality, reliability, and performance across all components and use cases.