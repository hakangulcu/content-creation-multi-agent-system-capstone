# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Content Creation Multi-Agent System built with LangGraph and local Ollama models. The system orchestrates 6 specialized agents through a sequential pipeline to generate high-quality, SEO-optimized content completely offline.

## Essential Commands

### Setup and Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"

# Setup Ollama (if not installed)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llama3.1:8b
```

### Running the System
```bash
# Basic demo
python main.py

# Interactive demo with multiple scenarios
python demo.py

# Run specific components for testing
python -c "from agents.research_agent import ResearchAgent; print('Research agent imported successfully')"
```

### Testing
```bash
# Run all tests
pytest test_agents.py -v

# Run specific test categories
pytest test_agents.py::TestTools -v
pytest test_agents.py::TestAgents -v
pytest test_agents.py::TestWorkflow -v

# Run with coverage
pytest test_agents.py --cov=. --cov-report=html
```

### Code Quality
```bash
# Format code
python -m black .

# Lint code
python -m flake8

# Type checking (if mypy is installed)
mypy main.py agents/
```

## System Architecture

### Core Components
- **main.py** - Primary workflow orchestrator using LangGraph StateGraph
- **agents/** - 6 specialized agent modules (research, planning, writing, editing, SEO, QA)
- **demo.py** - Interactive demonstration system with 4 predefined scenarios
- **test_agents.py** - Comprehensive test suite with unit, integration, and performance tests

### Agent Pipeline Flow
1. **ResearchAgent** - Web search via DuckDuckGo, data gathering
2. **PlanningAgent** - Content structure, outline, keyword strategy  
3. **WriterAgent** - Content generation using Ollama LLM
4. **EditorAgent** - Quality improvement, readability optimization via NLTK
5. **SEOAgent** - Keyword analysis, SEO scoring
6. **QualityAssuranceAgent** - Final validation, file output

### State Management
The system uses a `ContentCreationState` TypedDict that flows through all agents:
```python
state = {
    "request": ContentRequest,
    "research_data": ResearchData, 
    "content_plan": ContentPlan,
    "draft": ContentDraft,
    "analysis": ContentAnalysis,
    "final_content": str,
    "feedback_history": List[str],
    "revision_count": int,
    "metadata": Dict[str, Any]
}
```

## Configuration

### Ollama Models
- Default: `llama3.1:8b` (8GB RAM required)
- Fast: `phi3:mini` (4GB RAM required)  
- High-quality: `llama3.1:70b` (64GB RAM required)

### Environment Variables (.env)
```env
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEMPERATURE=0.7
OLLAMA_NUM_PREDICT=4096
```

## Development Patterns

### Adding New Agents
1. Create new agent file in `agents/` directory
2. Inherit common patterns from existing agents
3. Add to `agents/__init__.py` imports
4. Register in workflow graph in `main.py:_build_workflow()`
5. Add corresponding tests in `test_agents.py`

### Tool Development
Tools use `@tool` decorator from langchain_core:
```python
@tool
def custom_tool(input: str) -> dict:
    """Tool description for LLM"""
    # Implementation
    return {"result": "processed"}
```

### Error Handling
- All agents include try/catch blocks with fallback responses
- Web search tools handle network failures gracefully  
- File operations create directories automatically
- LLM calls have timeout and retry logic

### Testing Strategy
- Mock Ollama responses for consistent unit testing
- Use `pytest-asyncio` for async agent testing
- Performance tests measure end-to-end pipeline timing
- Edge case tests cover invalid inputs and resource constraints

## Common Issues

### Ollama Connection
If "Connection refused" errors occur:
```bash
# Check Ollama status
ollama list
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Memory Issues
For "Out of memory" errors, switch to smaller model:
```bash
ollama pull phi3:mini
# Update .env: OLLAMA_MODEL=phi3:mini
```

### NLTK Data Missing
If content analysis fails:
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Detailed File Structure

### Root Directory Files

#### Documentation Files
- **CLAUDE.md** - AI assistant guidance and project instructions (this file)
- **README.md** - Comprehensive project overview and setup instructions
- **API_DOCUMENTATION.md** - API reference and integration details
- **AGENT_DOCUMENTATION.md** - Detailed agent specifications and workflows
- **PROJECT_STRUCTURE.md** - Project architecture and component breakdown
- **SYSTEM_ARCHITECTURE.md** - System design and technical architecture
- **SUBMISSION_CHECKLIST.md** - Project completion and submission requirements
- **OLLAMA_SETUP_GUIDE.md** - Step-by-step Ollama installation and configuration
- **LICENSE** - MIT license for the project

#### Core System Files
- **main.py** - Primary workflow orchestrator using LangGraph StateGraph
- **demo.py** - Interactive demonstration system with 4 predefined scenarios
- **streamlit_app.py** - Web-based user interface for non-technical users
- **requirements.txt** - Python dependencies optimized for Ollama integration
- **types_shared.py** - Shared type definitions and data structures
- **resilience_utils.py** - Retry logic, circuit breakers, error handling utilities
- **security_utils.py** - Input validation, content filtering, security functions

#### Setup and Configuration Files
- **setup_local.sh** - Automated local development environment setup script
- **setup_project.py** - Python-based project initialization and configuration
- **resolve_conflicts.py** - Conflict resolution utilities for concurrent operations
- **pytest.ini** - PyTest configuration for test execution
- **.env.sample** - Environment variables template
- **.coveragerc** - Code coverage configuration

#### Testing Files
- **test_agents.py** - Main comprehensive test suite (unit, integration, performance)
- **test_tool_async.py** - Asynchronous tool testing utilities
- **test_run_file.txt** - Test execution logs and results

### Directory Structure

#### agents/
Contains 6 specialized agent modules with consistent architecture:
- **__init__.py** - Agent package initialization and exports
- **research_agent.py** - Web search via DuckDuckGo, data gathering, fact collection
- **planning_agent.py** - Content structure, outline creation, keyword strategy
- **writer_agent.py** - Content generation using Ollama LLM integration
- **editor_agent.py** - Quality improvement, readability optimization via NLTK
- **seo_agent.py** - Keyword analysis, SEO scoring, search optimization
- **qa_agent.py** - Final validation, quality assurance, file output management

#### tests/
Comprehensive testing framework with modular test organization:
- **__init__.py** - Test package initialization
- **conftest.py** - PyTest configuration, fixtures, and shared test utilities
- **test_agents.py** - Individual agent functionality and behavior tests
- **test_integration.py** - Multi-agent workflow and state management tests
- **test_e2e.py** - End-to-end system testing with real scenarios
- **test_tools.py** - External tool integration and API testing

#### outputs/
Generated content with organized file naming and metadata:
- **Artificial_Intelligence_in_Healthcare_20250807_203416.md** - Healthcare AI analysis
- **The_Future_of_Cybersecurity_with_AI_20250807_221834.md** - Cybersecurity trend analysis
- **Benefits_of_Exercise_20250730_131536.md** - Health and fitness content
- **Coding_with_AI_20250714_142625.md** - Programming and AI development
- **Artificial_Intelligence_in_Healthcare:_Transforming_Patient_Care_20250714_105656.md** - Healthcare transformation analysis

#### logs/
System logging and monitoring (auto-created during runtime):
- Agent execution logs
- Performance metrics
- Error tracking
- Workflow state transitions

#### __pycache__/
Python bytecode cache files (auto-generated):
- **main.cpython-310.pyc** - Compiled main module
- **resilience_utils.cpython-310.pyc** - Compiled resilience utilities
- **security_utils.cpython-310.pyc** - Compiled security functions  
- **types_shared.cpython-310.pyc** - Compiled shared type definitions

### File Dependencies and Relationships

#### Core Workflow Chain
1. **main.py** → orchestrates all agents via LangGraph
2. **agents/*.py** → implement specialized functionality
3. **types_shared.py** → provides shared data structures
4. **resilience_utils.py** → handles error recovery
5. **security_utils.py** → ensures safe operations

#### Testing Infrastructure
1. **pytest.ini** → configures test execution
2. **tests/conftest.py** → provides test fixtures
3. **tests/test_*.py** → implement test cases
4. **test_agents.py** → main test suite (root level)

#### Documentation Hierarchy
1. **README.md** → user-facing documentation
2. **CLAUDE.md** → AI assistant instructions
3. **API_DOCUMENTATION.md** → technical API reference
4. **AGENT_DOCUMENTATION.md** → agent specifications
5. **SYSTEM_ARCHITECTURE.md** → design documentation

### File Size and Complexity Overview

#### Large/Complex Files (>500 lines)
- **README.md** (~548 lines) - Comprehensive user documentation
- **main.py** (~400+ lines) - Core workflow orchestration
- **test_agents.py** (~300+ lines) - Comprehensive test suite

#### Medium Files (100-500 lines)  
- **agents/*.py** - Individual agent implementations
- **demo.py** - Interactive demonstration system
- **streamlit_app.py** - Web interface implementation

#### Small Files (<100 lines)
- **types_shared.py** - Type definitions
- **resilience_utils.py** - Utility functions
- **security_utils.py** - Security helpers
- Configuration and setup files

## Performance Expectations

Typical timing for 1500-word article on modern hardware:
- Research: 30-45s
- Planning: 15-20s  
- Writing: 60-90s
- Editing: 30-45s
- SEO: 20-30s
- QA: 15-25s
- **Total: 3-5 minutes**

## Development Notes

- System uses async/await throughout for performance
- All agents maintain state immutability principles
- LangGraph handles workflow orchestration and error recovery
- Content is validated at each pipeline stage
- Output files include comprehensive metadata for tracking
- System designed for offline operation with zero API costs