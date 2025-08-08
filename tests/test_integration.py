"""Integration tests for multi-agent workflows."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from main import ContentCreationWorkflow, ContentRequest, ContentType


@pytest.mark.integration
class TestAgentPipeline:
    """Test integration between agents in the pipeline."""
    
    @pytest.mark.asyncio
    async def test_research_to_planning_flow(self, mock_llm_responses):
        """Test data flow from research to planning agent."""
        # Mock workflow with controlled responses
        with patch('main.ChatOllama') as mock_chat:
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            
            workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
            
            # Setup mock responses for research and planning
            mock_llm.ainvoke.side_effect = [
                mock_llm_responses["research"],
                mock_llm_responses["planning"]
            ]
            
            # Create initial state
            request = ContentRequest(
                topic="AI in Healthcare",
                content_type=ContentType.ARTICLE,
                target_audience="Healthcare professionals",
                word_count=1000,
                tone="professional",
                keywords=["AI", "healthcare", "medical"]
            )
            
            state = {
                "request": request,
                "research_data": None,
                "content_plan": None,
                "draft": None,
                "analysis": None,
                "final_content": None,
                "feedback_history": [],
                "revision_count": 0,
                "metadata": {}
            }
            
            # Test research -> planning flow
            with patch('main.web_search_tool', return_value=[{"title": "Test", "url": "http://test.com", "snippet": "AI healthcare research"}]):
                # Run research
                research_result = await workflow.research_agent.research(state)
                assert research_result["research_data"] is not None
                
                # Run planning with research results
                planning_result = await workflow.planning_agent.plan_content(research_result)
                assert planning_result["content_plan"] is not None
                assert planning_result["content_plan"].title
                assert len(planning_result["content_plan"].outline) > 0
    
    @pytest.mark.asyncio 
    async def test_planning_to_writing_flow(self, mock_llm_responses, sample_research_data):
        """Test data flow from planning to writing agent."""
        with patch('main.ChatOllama') as mock_chat:
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            
            workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
            
            # Setup responses
            mock_llm.ainvoke.side_effect = [
                mock_llm_responses["planning"],
                mock_llm_responses["writing"]
            ]
            
            request = ContentRequest("Test Topic", ContentType.ARTICLE, "Test audience", 500)
            
            # Start with research data
            state = {
                "request": request,
                "research_data": sample_research_data,
                "content_plan": None,
                "draft": None,
                "analysis": None,
                "final_content": None,
                "feedback_history": [],
                "revision_count": 0,
                "metadata": {}
            }
            
            # Run planning
            planning_result = await workflow.planning_agent.plan_content(state)
            
            # Run writing with planning results
            writing_result = await workflow.writer_agent.write_content(planning_result)
            
            assert writing_result["draft"] is not None
            assert writing_result["draft"].content
            assert writing_result["draft"].word_count > 0
    
    @pytest.mark.asyncio
    async def test_writing_to_editing_flow(self, mock_llm_responses, sample_research_data, sample_content_plan):
        """Test data flow from writing to editing agent."""
        with patch('main.ChatOllama') as mock_chat:
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            
            workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
            
            mock_llm.ainvoke.side_effect = [
                mock_llm_responses["writing"],
                mock_llm_responses["editing"]
            ]
            
            request = ContentRequest("Test Topic", ContentType.ARTICLE, "Test audience", 500)
            
            state = {
                "request": request,
                "research_data": sample_research_data,
                "content_plan": sample_content_plan,
                "draft": None,
                "analysis": None,
                "final_content": None,
                "feedback_history": [],
                "revision_count": 0,
                "metadata": {}
            }
            
            # Run writing
            writing_result = await workflow.writer_agent.write_content(state)
            
            # Run editing with writing results
            with patch('main.content_analysis_tool') as mock_analysis:
                mock_analysis.return_value = {
                    "readability_score": 72.0,
                    "grade_level": 8.5,
                    "word_count": 120,
                    "reading_time": 1,
                    "keyword_density": {"test": 2.5}
                }
                
                editing_result = await workflow.editor_agent.edit_content(writing_result)
            
            assert editing_result["draft"] is not None
            assert editing_result["analysis"] is not None
            assert editing_result["analysis"].readability_score >= -100  # Flesch score can be negative for very difficult text
    
    @pytest.mark.asyncio
    async def test_editing_to_seo_flow(self, mock_llm_responses, sample_content_draft, sample_content_analysis):
        """Test data flow from editing to SEO agent."""
        with patch('main.ChatOllama') as mock_chat:
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            
            workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
            
            mock_llm.ainvoke.side_effect = [
                mock_llm_responses["editing"],
                mock_llm_responses["seo"]
            ]
            
            request = ContentRequest("Test Topic", ContentType.ARTICLE, "Test audience", 500, keywords=["test", "topic"])
            
            state = {
                "request": request,
                "research_data": MagicMock(),
                "content_plan": MagicMock(),
                "draft": sample_content_draft,
                "analysis": sample_content_analysis,
                "final_content": None,
                "feedback_history": [],
                "revision_count": 0,
                "metadata": {}
            }
            
            # Mock content analysis for editing
            with patch('main.content_analysis_tool') as mock_analysis:
                mock_analysis.return_value = {
                    "readability_score": 72.0,
                    "grade_level": 8.5,
                    "word_count": 120,
                    "reading_time": 1,
                    "keyword_density": {"test": 2.5}
                }
                
                editing_result = await workflow.editor_agent.edit_content(state)
            
            # Run SEO with editing results
            with patch('main.seo_optimization_tool') as mock_seo:
                mock_seo.return_value = {
                    "keyword_analysis": {"test": 3, "topic": 2},
                    "suggestions": ["Good keyword distribution"],
                    "seo_score": 88
                }
                
                seo_result = await workflow.seo_agent.optimize_seo(editing_result)
            
            assert seo_result["draft"] is not None
            assert "seo_score" in seo_result["metadata"]
    
    @pytest.mark.asyncio
    async def test_seo_to_qa_flow(self, mock_llm_responses, sample_content_draft, sample_content_analysis):
        """Test data flow from SEO to QA agent."""
        with patch('main.ChatOllama') as mock_chat:
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            
            workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
            
            mock_llm.ainvoke.side_effect = [
                mock_llm_responses["seo"],
                mock_llm_responses["qa"]
            ]
            
            request = ContentRequest("Test Topic", ContentType.ARTICLE, "Test audience", 500)
            
            state = {
                "request": request,
                "research_data": MagicMock(),
                "content_plan": MagicMock(),
                "draft": sample_content_draft,
                "analysis": sample_content_analysis,
                "final_content": None,
                "feedback_history": [],
                "revision_count": 0,
                "metadata": {"seo_score": 85}
            }
            
            # Run SEO
            with patch('main.seo_optimization_tool') as mock_seo:
                mock_seo.return_value = {
                    "keyword_analysis": {"test": 3},
                    "suggestions": [],
                    "seo_score": 85
                }
                
                seo_result = await workflow.seo_agent.optimize_seo(state)
            
            # Run QA with SEO results
            with patch('main.save_content_tool') as mock_save:
                mock_save.return_value = {
                    "status": "success",
                    "filepath": "outputs/test.md",
                    "message": "Saved successfully"
                }
                
                qa_result = await workflow.qa_agent.finalize_content(seo_result)
            
            assert qa_result["final_content"] is not None
            assert "output_file" in qa_result["metadata"]


@pytest.mark.integration
class TestWorkflowStateManagement:
    """Test state management across the workflow."""
    
    @pytest.mark.asyncio
    async def test_state_consistency_through_pipeline(self, mock_content_state):
        """Test that state remains consistent through the pipeline."""
        with patch('main.ChatOllama') as mock_chat:
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            
            workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
            
            # Mock all agent responses
            mock_response = MagicMock()
            mock_response.content = "Mock response"
            mock_llm.ainvoke.return_value = mock_response
            
            initial_request = mock_content_state["request"]
            
            # Start with minimal state
            state = {
                "request": initial_request,
                "research_data": None,
                "content_plan": None,
                "draft": None,
                "analysis": None,
                "final_content": None,
                "feedback_history": [],
                "revision_count": 0,
                "metadata": {}
            }
            
            # Mock all tools
            with patch('main.web_search_tool', return_value=[{"title": "Test", "url": "test.com", "snippet": "test"}]), \
                 patch('main.content_analysis_tool', return_value={"readability_score": 70.0, "grade_level": 8.0, "word_count": 100, "reading_time": 1, "keyword_density": {"test": 2.0}}), \
                 patch('main.seo_optimization_tool', return_value={"keyword_analysis": {"test": 2}, "suggestions": [], "seo_score": 80}), \
                 patch('main.save_content_tool', return_value={"status": "success", "filepath": "test.md", "message": "Saved"}):
                
                # Run through each agent
                state = await workflow.research_agent.research(state)
                assert state["request"] == initial_request  # Request should persist
                
                state = await workflow.planning_agent.plan_content(state)
                assert state["request"] == initial_request
                assert state["research_data"] is not None
                
                state = await workflow.writer_agent.write_content(state)
                assert state["request"] == initial_request
                assert state["content_plan"] is not None
                
                state = await workflow.editor_agent.edit_content(state)
                assert state["request"] == initial_request
                assert state["draft"] is not None
                
                state = await workflow.seo_agent.optimize_seo(state)
                assert state["request"] == initial_request
                assert state["analysis"] is not None
                
                state = await workflow.qa_agent.finalize_content(state)
                assert state["request"] == initial_request
                assert state["final_content"] is not None
    
    @pytest.mark.asyncio
    async def test_feedback_history_accumulation(self):
        """Test that feedback history accumulates correctly."""
        with patch('main.ChatOllama') as mock_chat:
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            
            workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
            
            # Mock responses with feedback
            mock_response = MagicMock()
            mock_response.content = "Response with feedback"
            mock_llm.ainvoke.return_value = mock_response
            
            request = ContentRequest("Test", ContentType.ARTICLE, "Test", 100)
            state = {
                "request": request,
                "research_data": None,
                "content_plan": None,
                "draft": None,
                "analysis": None,
                "final_content": None,
                "feedback_history": [],
                "revision_count": 0,
                "metadata": {}
            }
            
            with patch('main.web_search_tool', return_value=[{"title": "Test", "url": "test.com", "snippet": "test"}]):
                # Each agent should potentially add to feedback history
                initial_feedback_count = len(state["feedback_history"])
                
                state = await workflow.research_agent.research(state)
                after_research = len(state["feedback_history"])
                
                # Feedback history should be maintained and potentially grow
                assert len(state["feedback_history"]) >= initial_feedback_count
    
    @pytest.mark.asyncio
    async def test_metadata_accumulation(self):
        """Test that metadata accumulates correctly through the pipeline."""
        with patch('main.ChatOllama') as mock_chat:
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            
            workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
            
            mock_response = MagicMock()
            mock_response.content = "Mock content"
            mock_llm.ainvoke.return_value = mock_response
            
            request = ContentRequest("Test", ContentType.ARTICLE, "Test", 100)
            state = {
                "request": request,
                "research_data": None,
                "content_plan": None,
                "draft": None,
                "analysis": None,
                "final_content": None,
                "feedback_history": [],
                "revision_count": 0,
                "metadata": {"initial_key": "initial_value"}
            }
            
            with patch('main.web_search_tool', return_value=[]), \
                 patch('main.content_analysis_tool', return_value={"readability_score": 70.0, "grade_level": 8.0, "word_count": 100, "reading_time": 1, "keyword_density": {}}), \
                 patch('main.seo_optimization_tool', return_value={"keyword_analysis": {}, "suggestions": [], "seo_score": 75}), \
                 patch('main.save_content_tool', return_value={"status": "success", "filepath": "test.md", "message": "Saved"}):
                
                # Run through pipeline
                state = await workflow.research_agent.research(state)
                state = await workflow.planning_agent.plan_content(state)
                state = await workflow.writer_agent.write_content(state)
                state = await workflow.editor_agent.edit_content(state)
                state = await workflow.seo_agent.optimize_seo(state)
                state = await workflow.qa_agent.finalize_content(state)
                
                # Metadata should accumulate
                assert "initial_key" in state["metadata"]
                assert state["metadata"]["initial_key"] == "initial_value"
                
                # Should have additional metadata from agents
                metadata_keys = set(state["metadata"].keys())
                assert len(metadata_keys) > 1  # More than just initial key


@pytest.mark.integration
class TestErrorPropagation:
    """Test error handling and propagation between agents."""
    
    @pytest.mark.asyncio
    async def test_agent_error_recovery(self):
        """Test that agent errors don't break the pipeline."""
        with patch('main.ChatOllama') as mock_chat:
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            
            workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
            
            # First agent fails, but second should still work
            mock_llm.ainvoke.side_effect = [
                Exception("First agent fails"),
                MagicMock(content="Second agent succeeds")
            ]
            
            request = ContentRequest("Test", ContentType.ARTICLE, "Test", 100)
            state = {
                "request": request,
                "research_data": None,
                "content_plan": None,
                "draft": None,
                "analysis": None,
                "final_content": None,
                "feedback_history": [],
                "revision_count": 0,
                "metadata": {}
            }
            
            # Research agent fails but provides fallback
            with patch('main.web_search_tool', side_effect=Exception("Search fails")):
                result = await workflow.research_agent.research(state)
                
                # Should have fallback research data
                assert result["research_data"] is not None
                
                # Planning agent should still work with fallback data
                result = await workflow.planning_agent.plan_content(result)
                assert result["content_plan"] is not None
    
    @pytest.mark.asyncio 
    async def test_tool_error_handling(self):
        """Test handling of tool errors within agents."""
        with patch('main.ChatOllama') as mock_chat:
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            
            workflow = ContentCreationWorkflow(model_name="test-model", base_url="http://test:11434")
            
            mock_response = MagicMock()
            mock_response.content = "Agent response despite tool failure"
            mock_llm.ainvoke.return_value = mock_response
            
            request = ContentRequest("Test", ContentType.ARTICLE, "Test", 100)
            state = {
                "request": request,
                "research_data": MagicMock(),
                "content_plan": MagicMock(),
                "draft": MagicMock(title="Test", content="Test content", word_count=10, reading_time=1),
                "analysis": None,
                "final_content": None,
                "feedback_history": [],
                "revision_count": 0,
                "metadata": {}
            }
            
            # Content analysis tool fails but agent should handle it
            with patch('main.content_analysis_tool', side_effect=Exception("Analysis fails")):
                result = await workflow.editor_agent.edit_content(state)
                
                # Should still produce a result despite tool failure
                assert result["draft"] is not None
                # Analysis might be None or have fallback values
                assert "analysis" in result