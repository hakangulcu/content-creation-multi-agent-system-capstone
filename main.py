#!/usr/bin/env python3
"""
Content Creation Multi-Agent System
AAIDC Module 3 Project

This script demonstrates a sophisticated multi-agent system for automated content creation.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Additional imports for tools
import requests
import re
from urllib.parse import urlparse
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
import logging
import structlog

# Production imports for security and resilience
from security_utils import (
    validate_content_request,
    validate_and_sanitize_text,
    filter_content,
    ValidationError,
    SecurityLogger
)
from resilience_utils import (
    retry_with_backoff,
    timeout,
    circuit_breaker,
    RetryConfig,
    CircuitBreakerConfig,
    resilience_manager
)

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
security_logger = SecurityLogger()

# Import shared types
from types_shared import (
    ContentType,
    ContentRequest,
    ResearchData,
    ContentPlan,
    ContentDraft,
    ContentAnalysis
)

class ContentCreationState(TypedDict):
    """State object that flows through the multi-agent pipeline"""
    request: Optional[ContentRequest]
    research_data: Optional[ResearchData]
    content_plan: Optional[ContentPlan]
    draft: Optional[ContentDraft]
    analysis: Optional[ContentAnalysis]
    final_content: Optional[str]
    feedback_history: List[str]
    revision_count: int
    metadata: Dict[str, Any]

# =============================================================================
# TOOLS DEFINITION
# =============================================================================

@tool
@retry_with_backoff(RetryConfig(max_attempts=3, base_delay=1.0, timeout=30.0))
@circuit_breaker("web_search", CircuitBreakerConfig(failure_threshold=3, timeout=60.0))
def web_search_tool(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs web search to gather information for content research.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and snippet
    """
    try:
        # Validate and sanitize query
        validation_result = validate_and_sanitize_text(query, "search_query")
        if not validation_result.is_valid:
            logger.warning(
                "web_search_invalid_query",
                query_hash=query[:50],
                errors=validation_result.errors
            )
            return [{"title": "Error", "url": "", "snippet": "Invalid search query"}]
        
        sanitized_query = validation_result.sanitized_input
        
        search = DuckDuckGoSearchRun()
        results = search.run(sanitized_query)
        
        # Parse results (simplified parsing)
        parsed_results = []
        lines = results.split('\n')
        for i, line in enumerate(lines[:max_results]):
            if line.strip():
                # Filter content for safety
                filtered_snippet, filtered_items = filter_content(line.strip())
                
                parsed_results.append({
                    "title": f"Result {i+1}",
                    "url": "https://example.com",  # Placeholder
                    "snippet": filtered_snippet
                })
        
        logger.info(
            "web_search_completed",
            query_length=len(sanitized_query),
            results_count=len(parsed_results)
        )
        
        return parsed_results
    except Exception as e:
        logger.error(
            "web_search_error",
            error=str(e),
            error_type=type(e).__name__,
            query_length=len(query)
        )
        # Graceful fallback
        return [{
            "title": "Search Unavailable", 
            "url": "", 
            "snippet": "Search service temporarily unavailable. Please try again later."
        }]

@tool
@retry_with_backoff(RetryConfig(max_attempts=2, base_delay=0.5))
@timeout(15.0)
def content_analysis_tool(content: str) -> Dict[str, Any]:
    """
    Analyzes content for readability, SEO, and quality metrics.
    
    Args:
        content: The content text to analyze
        
    Returns:
        Dictionary containing analysis metrics
    """
    try:
        # Validate content first
        validation_result = validate_and_sanitize_text(content, "analysis_content")
        if not validation_result.is_valid:
            logger.warning(
                "content_analysis_invalid_content",
                content_length=len(content),
                errors=validation_result.errors
            )
            return {
                "error": "Invalid content for analysis",
                "validation_errors": validation_result.errors
            }
        
        sanitized_content = validation_result.sanitized_input
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Calculate readability metrics
        readability = flesch_reading_ease(sanitized_content)
        grade_level = flesch_kincaid_grade(sanitized_content)
        
        # Word count and reading time
        word_count = len(sanitized_content.split())
        reading_time = max(1, word_count // 200)  # ~200 words per minute
        
        # Basic keyword density (simplified)
        words = sanitized_content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only count words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate density for top words
        keyword_density = {}
        total_words = len(words)
        for word, count in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            keyword_density[word] = round((count / total_words) * 100, 2)
        
        result = {
            "readability_score": readability,
            "grade_level": grade_level,
            "word_count": word_count,
            "reading_time": reading_time,
            "keyword_density": keyword_density,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            "content_analysis_completed",
            word_count=word_count,
            readability_score=readability,
            grade_level=grade_level
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "content_analysis_error",
            error=str(e),
            error_type=type(e).__name__,
            content_length=len(content)
        )
        # Graceful fallback
        return {
            "error": "Analysis temporarily unavailable",
            "readability_score": 60.0,  # Reasonable default
            "grade_level": 8.0,
            "word_count": len(content.split()),
            "reading_time": max(1, len(content.split()) // 200),
            "keyword_density": {},
            "analysis_timestamp": datetime.now().isoformat()
        }

@tool
def seo_optimization_tool(content: str, target_keywords: List[str]) -> Dict[str, Any]:
    """
    Provides SEO optimization suggestions for content.
    
    Args:
        content: The content to optimize
        target_keywords: List of target keywords
        
    Returns:
        SEO analysis and suggestions
    """
    try:
        suggestions = []
        content_lower = content.lower()
        
        # Check keyword presence
        keyword_analysis = {}
        for keyword in target_keywords:
            count = content_lower.count(keyword.lower())
            keyword_analysis[keyword] = count
            
            if count == 0:
                suggestions.append(f"Consider adding the keyword '{keyword}' to your content")
            elif count > 10:
                suggestions.append(f"Keyword '{keyword}' may be overused ({count} times)")
        
        # Check title and headings
        lines = content.split('\n')
        has_title = any(line.startswith('#') for line in lines)
        if not has_title:
            suggestions.append("Add a compelling title using # markdown")
        
        # Check content length
        word_count = len(content.split())
        if word_count < 300:
            suggestions.append("Content is quite short for SEO - consider expanding")
        elif word_count > 3000:
            suggestions.append("Content is very long - consider breaking into sections")
        
        return {
            "keyword_analysis": keyword_analysis,
            "suggestions": suggestions,
            "word_count": word_count,
            "seo_score": min(100, max(0, 70 - len(suggestions) * 5))  # Simple scoring
        }
    except Exception as e:
        logger.error(f"SEO analysis error: {e}")
        return {"error": str(e)}

@tool
def save_content_tool(content: str, filename: str) -> Dict[str, str]:
    """
    Saves content to a file.
    
    Args:
        content: Content to save
        filename: Name of the file
        
    Returns:
        Save operation result
    """
    try:
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        filepath = os.path.join("outputs", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "status": "success",
            "filepath": filepath,
            "message": f"Content saved to {filepath}"
        }
    except Exception as e:
        logger.error(f"Save error: {e}")
        return {
            "status": "error",
            "message": f"Failed to save: {str(e)}"
        }

# =============================================================================
# AGENT IMPORTS - Import from specialized agent modules
# =============================================================================

from agents import (
    ResearchAgent,
    PlanningAgent,
    WriterAgent,
    EditorAgent,
    SEOAgent,
    QualityAssuranceAgent
)

# =============================================================================
# LANGGRAPH WORKFLOW DEFINITION
# =============================================================================

class ContentCreationWorkflow:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        # Initialize local Ollama LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
            base_url=base_url,
            # Additional parameters for better performance
            num_predict=4096,  # Max tokens to generate
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
        
        # Initialize specialized agents from agent modules
        self.research_agent = ResearchAgent(self.llm)
        self.planning_agent = PlanningAgent(self.llm)
        self.writer_agent = WriterAgent(self.llm)
        self.editor_agent = EditorAgent(self.llm)
        self.seo_agent = SEOAgent(self.llm)
        self.qa_agent = QualityAssuranceAgent(self.llm)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(ContentCreationState)
        
        # Add nodes (specialized agents with distinct capabilities)
        workflow.add_node("research", self.research_agent.research)
        workflow.add_node("planning", self.planning_agent.plan_content)
        workflow.add_node("writing", self.writer_agent.write_content)
        workflow.add_node("editing", self.editor_agent.edit_content)
        workflow.add_node("seo_optimization", self.seo_agent.optimize_seo)
        workflow.add_node("quality_assurance", self.qa_agent.finalize_content)
        
        # Define the workflow edges
        workflow.add_edge("research", "planning")
        workflow.add_edge("planning", "writing")
        workflow.add_edge("writing", "editing")
        workflow.add_edge("editing", "seo_optimization")
        workflow.add_edge("seo_optimization", "quality_assurance")
        workflow.add_edge("quality_assurance", END)
        
        # Set entry point
        workflow.set_entry_point("research")
        
        # Compile the workflow
        return workflow.compile()
    
    async def create_content(self, content_request: ContentRequest, progress_callback=None) -> ContentCreationState:
        """Execute the complete content creation workflow with security validation"""
        
        # Validate and sanitize input request
        validation_result = validate_content_request(content_request)
        if not validation_result.is_valid:
            error_msg = f"Invalid content request: {', '.join(validation_result.errors)}"
            logger.error(
                "content_request_validation_failed",
                errors=validation_result.errors,
                warnings=validation_result.warnings,
                threat_level=validation_result.threat_level
            )
            raise ValidationError(error_msg)
        
        # Use sanitized request
        sanitized_request = validation_result.sanitized_input
        
        # Initialize state as a dictionary
        state: ContentCreationState = {
            "request": sanitized_request,
            "research_data": None,
            "content_plan": None,
            "draft": None,
            "analysis": None,
            "final_content": None,
            "feedback_history": [],
            "revision_count": 0,
            "metadata": {
                "workflow_start_time": datetime.now().isoformat(),
                "security_validation": {
                    "validated": True,
                    "warnings": validation_result.warnings,
                    "threat_level": validation_result.threat_level
                }
            }
        }
        
        logger.info(
            "content_creation_started",
            topic=sanitized_request.topic,
            content_type=sanitized_request.content_type.value,
            word_count=sanitized_request.word_count,
            threat_level=validation_result.threat_level
        )
        
        try:
            # Execute workflow with timeout protection and progress tracking
            workflow_timeout = float(os.getenv("WORKFLOW_TIMEOUT", "300"))  # 5 minutes default
            
            if progress_callback:
                final_state = await asyncio.wait_for(
                    self._execute_workflow_with_progress(state, progress_callback),
                    timeout=workflow_timeout
                )
            else:
                final_state = await asyncio.wait_for(
                    self.workflow.ainvoke(state),
                    timeout=workflow_timeout
                )
            
            # Add completion metadata
            final_state["metadata"]["workflow_end_time"] = datetime.now().isoformat()
            final_state["metadata"]["total_duration"] = (
                datetime.fromisoformat(final_state["metadata"]["workflow_end_time"]) - 
                datetime.fromisoformat(final_state["metadata"]["workflow_start_time"])
            ).total_seconds()
            
            logger.info(
                "content_creation_completed",
                topic=sanitized_request.topic,
                duration=final_state["metadata"]["total_duration"],
                revision_count=final_state["revision_count"]
            )
            
            return final_state
            
        except asyncio.TimeoutError:
            logger.error(
                "content_creation_timeout",
                topic=sanitized_request.topic,
                timeout=workflow_timeout
            )
            raise Exception(f"Content creation timed out after {workflow_timeout} seconds")
            
        except Exception as e:
            logger.error(
                "content_creation_failed",
                topic=sanitized_request.topic,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    async def _execute_workflow_with_progress(self, state: ContentCreationState, progress_callback) -> ContentCreationState:
        """Execute workflow with progress tracking for each agent"""
        
        # Define the agent execution order and their display names
        agent_steps = [
            ("research", "Research & Data Gathering"),
            ("planning", "Content Planning"), 
            ("writing", "Writing Content"),
            ("editing", "Editing & Refinement"),
            ("seo_optimization", "SEO Optimization"),
            ("quality_assurance", "Quality Assurance")
        ]
        
        current_state = state
        
        for i, (agent_key, display_name) in enumerate(agent_steps):
            # Update progress before each agent
            progress = i / len(agent_steps)
            progress_callback(progress, display_name, i + 1, len(agent_steps))
            
            # Execute the specific agent
            if agent_key == "research":
                current_state = await self._execute_research(current_state)
            elif agent_key == "planning":
                current_state = await self._execute_planning(current_state)
            elif agent_key == "writing":
                current_state = await self._execute_writing(current_state)
            elif agent_key == "editing":
                current_state = await self._execute_editing(current_state)
            elif agent_key == "seo_optimization":
                current_state = await self._execute_seo(current_state)
            elif agent_key == "quality_assurance":
                current_state = await self._execute_qa(current_state)
        
        # Final progress update
        progress_callback(1.0, "Content generation completed", len(agent_steps), len(agent_steps))
        
        return current_state
    
    async def _execute_research(self, state: ContentCreationState) -> ContentCreationState:
        """Execute research agent"""
        from agents.research_agent import ResearchAgent
        agent = ResearchAgent(self.llm)
        return await agent.research(state)
    
    async def _execute_planning(self, state: ContentCreationState) -> ContentCreationState:
        """Execute planning agent"""
        from agents.planning_agent import PlanningAgent
        agent = PlanningAgent(self.llm)
        return await agent.plan_content(state)
    
    async def _execute_writing(self, state: ContentCreationState) -> ContentCreationState:
        """Execute writing agent"""
        from agents.writer_agent import WriterAgent
        agent = WriterAgent(self.llm)
        return await agent.write_content(state)
    
    async def _execute_editing(self, state: ContentCreationState) -> ContentCreationState:
        """Execute editing agent"""
        from agents.editor_agent import EditorAgent
        agent = EditorAgent(self.llm)
        return await agent.edit_content(state)
    
    async def _execute_seo(self, state: ContentCreationState) -> ContentCreationState:
        """Execute SEO agent"""
        from agents.seo_agent import SEOAgent
        agent = SEOAgent(self.llm)
        return await agent.optimize_seo(state)
    
    async def _execute_qa(self, state: ContentCreationState) -> ContentCreationState:
        """Execute QA agent"""
        from agents.qa_agent import QualityAssuranceAgent
        agent = QualityAssuranceAgent(self.llm)
        return await agent.finalize_content(state)

# =============================================================================
# DEMO AND TESTING
# =============================================================================

async def demo_content_creation():
    """Demo function to showcase the multi-agent system"""
    
    # Get Ollama configuration from environment (with defaults)
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"Using Ollama model: {model_name}")
    print(f"Ollama server: {base_url}")
    
    # Create workflow
    try:
        workflow = ContentCreationWorkflow(model_name=model_name, base_url=base_url)
        print("Ollama connection established")
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        print(f"And the model is installed: ollama pull {model_name}")
        return
    
    # Define a content request
    request = ContentRequest(
        topic="Artificial Intelligence in Healthcare",
        content_type=ContentType.BLOG_POST,
        target_audience="Healthcare professionals and technology leaders",
        word_count=1500,
        tone="professional yet accessible",
        keywords=["AI in healthcare", "medical AI", "healthcare technology", "patient care"],
        special_requirements="Include recent statistics and real-world examples"
    )
    
    print("Starting Multi-Agent Content Creation System")
    print(f"Topic: {request.topic}")
    print(f"Target: {request.target_audience}")
    print(f"Length: {request.word_count} words")
    print("-" * 60)
    
    try:
        # Execute workflow
        result = await workflow.create_content(request)
        
        print("\nContent Creation Completed Successfully!")
        print(f"Final word count: {result['draft'].word_count}")
        print(f"Reading time: {result['draft'].reading_time} minutes")
        print(f"Saved to: {result['metadata'].get('output_file', 'N/A')}")
        print(f"SEO Score: {result['metadata'].get('seo_score', 'N/A')}")
        
        # Display content preview
        if result["final_content"]:
            print("\nContent Preview:")
            print("-" * 40)
            preview = result["final_content"][:500] + "..." if len(result["final_content"]) > 500 else result["final_content"]
            print(preview)
        
    except Exception as e:
        print(f"Error during content creation: {e}")
        logger.error(f"Content creation failed: {e}")

# =============================================================================
# TEST-FRIENDLY FUNCTION VERSIONS (without decorators)
# =============================================================================

def _web_search_function(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Raw web search function for testing"""
    try:
        validation_result = validate_and_sanitize_text(query, "search_query")
        if not validation_result.is_valid:
            return [{"title": "Error", "url": "", "snippet": "Invalid search query"}]
        
        sanitized_query = validation_result.sanitized_input
        search = DuckDuckGoSearchRun()
        results = search.run(sanitized_query)
        
        parsed_results = []
        lines = results.split('\n')
        for i, line in enumerate(lines[:max_results]):
            if line.strip():
                filtered_snippet, _ = filter_content(line.strip())
                parsed_results.append({
                    "title": f"Result {i+1}",
                    "url": "https://example.com",
                    "snippet": filtered_snippet
                })
        return parsed_results
    except Exception as e:
        return [{"title": "Error", "url": "", "snippet": "Search Unavailable"}]

def _content_analysis_function(content: str) -> Dict[str, Any]:
    """Raw content analysis function for testing"""
    try:
        validation_result = validate_and_sanitize_text(content, "analysis_content")
        if not validation_result.is_valid:
            return {"error": "Invalid content for analysis"}
        
        sanitized_content = validation_result.sanitized_input
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        readability = flesch_reading_ease(sanitized_content)
        grade_level = flesch_kincaid_grade(sanitized_content)
        word_count = len(sanitized_content.split())
        reading_time = max(1, word_count // 200)
        
        words = sanitized_content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        keyword_density = {}
        total_words = len(words)
        for word, count in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            keyword_density[word] = round((count / total_words) * 100, 2)
        
        return {
            "readability_score": readability,
            "grade_level": grade_level,
            "word_count": word_count,
            "reading_time": reading_time,
            "keyword_density": keyword_density,
            "analysis_timestamp": datetime.now().isoformat()
        }
    except Exception:
        return {
            "error": "Analysis temporarily unavailable",
            "readability_score": 60.0,
            "grade_level": 8.0,
            "word_count": len(content.split()),
            "reading_time": max(1, len(content.split()) // 200),
            "keyword_density": {},
            "analysis_timestamp": datetime.now().isoformat()
        }

def _seo_optimization_function(content: str, target_keywords: List[str]) -> Dict[str, Any]:
    """Raw SEO optimization function for testing"""
    try:
        validation_result = validate_and_sanitize_text(content, "seo_content")
        if not validation_result.is_valid:
            return {"error": "Invalid content for SEO analysis"}
        
        sanitized_content = validation_result.sanitized_input.lower()
        
        keyword_analysis = {}
        for keyword in target_keywords:
            count = sanitized_content.count(keyword.lower())
            keyword_analysis[keyword] = count
        
        suggestions = []
        for keyword, count in keyword_analysis.items():
            if count == 0:
                suggestions.append(f"Consider adding '{keyword}' to improve SEO")
            elif count > 10:
                suggestions.append(f"'{keyword}' might be overused ({count} times)")
        
        has_title = content.strip().startswith('#')
        if not has_title:
            suggestions.append("Consider adding a title with # to improve SEO")
        
        word_count = len(content.split())
        if word_count < 300:
            suggestions.append("Content might be too short for good SEO (consider 300+ words)")
        elif word_count > 2000:
            suggestions.append("Content might be too long for optimal readability")
        
        seo_score = max(0, 100 - len(suggestions) * 10)
        
        return {
            "keyword_analysis": keyword_analysis,
            "suggestions": suggestions,
            "seo_score": seo_score,
            "word_count": word_count,
            "has_title": has_title
        }
    except Exception:
        return {
            "error": "SEO analysis temporarily unavailable",
            "keyword_analysis": {},
            "suggestions": ["Analysis failed"],
            "seo_score": 50,
            "word_count": 0,
            "has_title": False
        }

def _save_content_function(content: str, filename: str) -> Dict[str, str]:
    """Raw save content function for testing"""
    try:
        os.makedirs("outputs", exist_ok=True)
        filepath = os.path.join("outputs", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return {
            "status": "success",
            "filepath": filepath,
            "message": f"Content saved to {filepath}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save: {str(e)}"
        }

def main():
    """Main entry point"""
    print("Content Creation Multi-Agent System")
    print("AAIDC Module 3 Project")
    print("=" * 50)
    
    # Run demo
    asyncio.run(demo_content_creation())

if __name__ == "__main__":
    main()