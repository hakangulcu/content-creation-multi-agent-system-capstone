"""Unit tests for individual agents."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from main import ContentRequest, ContentType
from agents import (
    ResearchAgent,
    PlanningAgent, 
    WriterAgent,
    EditorAgent,
    SEOAgent,
    QualityAssuranceAgent
)


@pytest.mark.unit
class TestResearchAgent:
    """Test cases for ResearchAgent."""
    
    @pytest.fixture
    def research_agent(self, mock_ollama_llm):
        """Create ResearchAgent instance for testing."""
        return ResearchAgent(mock_ollama_llm)
    
    @pytest.mark.asyncio
    async def test_research_agent_initialization(self, mock_ollama_llm):
        """Test ResearchAgent initialization."""
        agent = ResearchAgent(mock_ollama_llm)
        assert agent.llm is mock_ollama_llm
        assert hasattr(agent, 'research')
    
    @pytest.mark.asyncio
    async def test_research_method_with_valid_state(self, research_agent, sample_content_request, mock_web_search):
        """Test research method with valid content state."""
        # Setup
        initial_state = {
            "request": sample_content_request,
            "research_data": None,
            "content_plan": None,
            "draft": None,
            "analysis": None,
            "final_content": None,
            "feedback_history": [],
            "revision_count": 0,
            "metadata": {}
        }
        
        # Mock LLM response for research
        mock_response = MagicMock()
        mock_response.content = """
        Research findings:
        - Key fact: AI improves diagnostic accuracy by 30%
        - Statistic: Healthcare AI market expected to reach $102 billion
        - Quote: "AI is transforming patient care" - Dr. Smith
        - Related: Machine learning, Medical imaging
        """
        research_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Mock web search
        with patch('main.web_search_tool', return_value=mock_web_search):
            result = await research_agent.research(initial_state)
        
        # Verify
        assert result["research_data"] is not None
        assert isinstance(result["research_data"].sources, list)
        assert isinstance(result["research_data"].key_facts, list)
        assert len(result["research_data"].key_facts) > 0
    
    @pytest.mark.asyncio
    async def test_research_method_exception_handling(self, research_agent, sample_content_request):
        """Test research method exception handling."""
        initial_state = {
            "request": sample_content_request,
            "research_data": None,
            "content_plan": None,
            "draft": None,
            "analysis": None,
            "final_content": None,
            "feedback_history": [],
            "revision_count": 0,
            "metadata": {}
        }
        
        # Mock LLM to raise exception
        research_agent.llm.ainvoke = AsyncMock(side_effect=Exception("LLM Error"))
        
        result = await research_agent.research(initial_state)
        
        # Should handle gracefully with fallback
        assert result["research_data"] is not None
        assert "Search failed" in str(result["research_data"].key_facts)


@pytest.mark.unit  
class TestPlanningAgent:
    """Test cases for PlanningAgent."""
    
    @pytest.fixture
    def planning_agent(self, mock_ollama_llm):
        """Create PlanningAgent instance for testing."""
        return PlanningAgent(mock_ollama_llm)
    
    @pytest.mark.asyncio
    async def test_planning_agent_initialization(self, mock_ollama_llm):
        """Test PlanningAgent initialization."""
        agent = PlanningAgent(mock_ollama_llm)
        assert agent.llm is mock_ollama_llm
        assert hasattr(agent, 'plan_content')
    
    @pytest.mark.asyncio
    async def test_plan_content_method(self, planning_agent, sample_content_request, sample_research_data):
        """Test plan_content method."""
        initial_state = {
            "request": sample_content_request,
            "research_data": sample_research_data,
            "content_plan": None,
            "draft": None,
            "analysis": None,
            "final_content": None,
            "feedback_history": [],
            "revision_count": 0,
            "metadata": {}
        }
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = """
        Content Plan:
        Title: Test Article About Sample Topic
        Outline:
        1. Introduction
        2. Main Benefits  
        3. Implementation
        4. Conclusion
        
        Key Points:
        - Point 1: Important insight
        - Point 2: Key benefit
        - Point 3: Implementation tip
        
        Target Keywords: test, sample, article
        Estimated Length: 500 words
        """
        planning_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await planning_agent.plan_content(initial_state)
        
        # Verify
        assert result["content_plan"] is not None
        assert result["content_plan"].title
        assert len(result["content_plan"].outline) > 0
        assert len(result["content_plan"].key_points) > 0


@pytest.mark.unit
class TestWriterAgent:
    """Test cases for WriterAgent."""
    
    @pytest.fixture
    def writer_agent(self, mock_ollama_llm):
        """Create WriterAgent instance for testing."""
        return WriterAgent(mock_ollama_llm)
    
    @pytest.mark.asyncio
    async def test_writer_agent_initialization(self, mock_ollama_llm):
        """Test WriterAgent initialization."""
        agent = WriterAgent(mock_ollama_llm)
        assert agent.llm is mock_ollama_llm
        assert hasattr(agent, 'write_content')
    
    @pytest.mark.asyncio
    async def test_write_content_method(self, writer_agent, sample_content_request, sample_research_data, sample_content_plan):
        """Test write_content method."""
        initial_state = {
            "request": sample_content_request,
            "research_data": sample_research_data,
            "content_plan": sample_content_plan,
            "draft": None,
            "analysis": None,
            "final_content": None,
            "feedback_history": [],
            "revision_count": 0,
            "metadata": {}
        }
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = """
        # Test Article Title
        
        ## Introduction
        This article explores the important aspects of our test topic.
        
        ## Main Content
        Here we dive into the key points and benefits of the subject matter.
        
        ## Conclusion
        In conclusion, this topic offers significant value and opportunities.
        """
        writer_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await writer_agent.write_content(initial_state)
        
        # Verify
        assert result["draft"] is not None
        assert result["draft"].title
        assert result["draft"].content
        assert result["draft"].word_count > 0
        assert result["draft"].reading_time > 0


@pytest.mark.unit
class TestEditorAgent:
    """Test cases for EditorAgent."""
    
    @pytest.fixture
    def editor_agent(self, mock_ollama_llm):
        """Create EditorAgent instance for testing."""
        return EditorAgent(mock_ollama_llm)
    
    @pytest.mark.asyncio
    async def test_editor_agent_initialization(self, mock_ollama_llm):
        """Test EditorAgent initialization."""
        agent = EditorAgent(mock_ollama_llm)
        assert agent.llm is mock_ollama_llm
        assert hasattr(agent, 'edit_content')
    
    @pytest.mark.asyncio
    async def test_edit_content_method(self, editor_agent, sample_content_request, sample_research_data, sample_content_plan, sample_content_draft):
        """Test edit_content method."""
        initial_state = {
            "request": sample_content_request,
            "research_data": sample_research_data,
            "content_plan": sample_content_plan,
            "draft": sample_content_draft,
            "analysis": None,
            "final_content": None,
            "feedback_history": [],
            "revision_count": 0,
            "metadata": {}
        }
        
        # Mock LLM response with improved content
        mock_response = MagicMock()
        mock_response.content = """
        # Enhanced Test Article Title
        
        ## Introduction
        This comprehensive article thoroughly explores the important aspects of our test topic.
        
        ## Main Content
        Here we delve deeply into the key points and substantial benefits of the subject matter.
        
        ## Conclusion
        In conclusion, this topic offers significant value and transformative opportunities.
        """
        editor_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Mock content analysis
        with patch('main.content_analysis_tool') as mock_analysis:
            mock_analysis.return_value = {
                "readability_score": 75.0,
                "grade_level": 8.5,
                "word_count": 45,
                "reading_time": 1,
                "keyword_density": {"test": 2.2, "article": 1.8}
            }
            
            result = await editor_agent.edit_content(initial_state)
        
        # Verify
        assert result["draft"] is not None
        assert result["analysis"] is not None
        assert "Enhanced" in result["draft"].content or len(result["draft"].content) >= len(sample_content_draft.content)


@pytest.mark.unit
class TestSEOAgent:
    """Test cases for SEOAgent."""
    
    @pytest.fixture
    def seo_agent(self, mock_ollama_llm):
        """Create SEOAgent instance for testing."""
        return SEOAgent(mock_ollama_llm)
    
    @pytest.mark.asyncio
    async def test_seo_agent_initialization(self, mock_ollama_llm):
        """Test SEOAgent initialization."""
        agent = SEOAgent(mock_ollama_llm)
        assert agent.llm is mock_ollama_llm
        assert hasattr(agent, 'optimize_seo')
    
    @pytest.mark.asyncio
    async def test_optimize_seo_method(self, seo_agent, sample_content_request, sample_research_data, sample_content_plan, sample_content_draft, sample_content_analysis):
        """Test optimize_seo method."""
        initial_state = {
            "request": sample_content_request,
            "research_data": sample_research_data,
            "content_plan": sample_content_plan,
            "draft": sample_content_draft,
            "analysis": sample_content_analysis,
            "final_content": None,
            "feedback_history": [],
            "revision_count": 0,
            "metadata": {}
        }
        
        # Mock LLM response with SEO-optimized content
        mock_response = MagicMock()
        mock_response.content = """
        # Test Article Title - Complete Guide for Beginners
        
        ## Introduction
        This test article provides comprehensive insights into the sample topic.
        
        ## Main Content
        The test demonstrates key sample techniques and article best practices.
        
        ## Conclusion
        This sample article showcases effective test methodologies.
        """
        seo_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Mock SEO optimization tool
        with patch('main.seo_optimization_tool') as mock_seo:
            mock_seo.return_value = {
                "keyword_analysis": {"test": 3, "sample": 2, "article": 2},
                "suggestions": ["Keywords well distributed", "Title optimized"],
                "seo_score": 85
            }
            
            result = await seo_agent.optimize_seo(initial_state)
        
        # Verify
        assert result["draft"] is not None
        assert "seo_score" in result["metadata"]
        assert result["metadata"]["seo_score"] > 0


@pytest.mark.unit
class TestQualityAssuranceAgent:
    """Test cases for QualityAssuranceAgent."""
    
    @pytest.fixture
    def qa_agent(self, mock_ollama_llm):
        """Create QualityAssuranceAgent instance for testing."""
        return QualityAssuranceAgent(mock_ollama_llm)
    
    @pytest.mark.asyncio
    async def test_qa_agent_initialization(self, mock_ollama_llm):
        """Test QualityAssuranceAgent initialization."""
        agent = QualityAssuranceAgent(mock_ollama_llm)
        assert agent.llm is mock_ollama_llm
        assert hasattr(agent, 'finalize_content')
    
    @pytest.mark.asyncio
    async def test_finalize_content_method(self, qa_agent, sample_content_request, sample_research_data, sample_content_plan, sample_content_draft, sample_content_analysis):
        """Test finalize_content method."""
        initial_state = {
            "request": sample_content_request,
            "research_data": sample_research_data,
            "content_plan": sample_content_plan,
            "draft": sample_content_draft,
            "analysis": sample_content_analysis,
            "final_content": None,
            "feedback_history": [],
            "revision_count": 0,
            "metadata": {"seo_score": 85}
        }
        
        # Mock LLM response with final polished content
        mock_response = MagicMock()
        mock_response.content = """
        # Final Polished Test Article Title
        
        ## Introduction
        This expertly crafted test article provides comprehensive insights into the sample topic with professional polish.
        
        ## Main Content  
        The carefully reviewed test demonstrates key sample techniques and article best practices with quality assurance.
        
        ## Conclusion
        This thoroughly reviewed sample article showcases effective test methodologies with final approval.
        """
        qa_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Mock file save operation
        with patch('main.save_content_tool') as mock_save:
            mock_save.return_value = {
                "status": "success", 
                "filepath": "outputs/test_article.md",
                "message": "Content saved successfully"
            }
            
            result = await qa_agent.finalize_content(initial_state)
        
        # Verify
        assert result["final_content"] is not None
        assert "output_file" in result["metadata"]
        assert result["metadata"]["output_file"]
        assert "Polished" in result["final_content"] or "quality" in result["final_content"].lower()


@pytest.mark.unit
class TestAgentErrorHandling:
    """Test error handling across all agents."""
    
    @pytest.mark.asyncio
    async def test_agent_llm_timeout(self, mock_ollama_llm):
        """Test agent behavior when LLM times out."""
        mock_ollama_llm.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError("LLM timeout"))
        
        agent = ResearchAgent(mock_ollama_llm)
        state = {"request": ContentRequest("test", ContentType.ARTICLE, "test", 100)}
        
        # Should handle timeout gracefully
        result = await agent.research(state)
        assert result is not None
        assert "research_data" in result
    
    @pytest.mark.asyncio
    async def test_agent_network_error(self, mock_ollama_llm):
        """Test agent behavior with network errors."""
        mock_ollama_llm.ainvoke = AsyncMock(side_effect=ConnectionError("Network unavailable"))
        
        agent = PlanningAgent(mock_ollama_llm)
        state = {
            "request": ContentRequest("test", ContentType.ARTICLE, "test", 100),
            "research_data": MagicMock()
        }
        
        result = await agent.plan_content(state)
        assert result is not None
        assert "content_plan" in result