#!/usr/bin/env python3
"""
Content Creation Multi-Agent System
AAIDC Module 2 Project

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    SOCIAL_MEDIA = "social_media"
    NEWSLETTER = "newsletter"
    MARKETING_COPY = "marketing_copy"

@dataclass
class ContentRequest:
    topic: str
    content_type: ContentType
    target_audience: str
    word_count: int
    tone: str = "professional"
    keywords: List[str] = None
    special_requirements: str = ""

@dataclass
class ResearchData:
    sources: List[str]
    key_facts: List[str]
    statistics: List[str]
    quotes: List[str]
    related_topics: List[str]

@dataclass
class ContentPlan:
    title: str
    outline: List[str]
    key_points: List[str]
    target_keywords: List[str]
    estimated_length: int

@dataclass
class ContentDraft:
    title: str
    content: str
    word_count: int
    reading_time: int

@dataclass
class ContentAnalysis:
    readability_score: float
    grade_level: float
    keyword_density: Dict[str, float]
    suggestions: List[str]

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
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        
        # Parse results (simplified parsing)
        parsed_results = []
        lines = results.split('\n')
        for i, line in enumerate(lines[:max_results]):
            if line.strip():
                parsed_results.append({
                    "title": f"Result {i+1}",
                    "url": "https://example.com",  # Placeholder
                    "snippet": line.strip()
                })
        
        return parsed_results
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return [{"title": "Error", "url": "", "snippet": f"Search failed: {str(e)}"}]

@tool
def content_analysis_tool(content: str) -> Dict[str, Any]:
    """
    Analyzes content for readability, SEO, and quality metrics.
    
    Args:
        content: The content text to analyze
        
    Returns:
        Dictionary containing analysis metrics
    """
    try:
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Calculate readability metrics
        readability = flesch_reading_ease(content)
        grade_level = flesch_kincaid_grade(content)
        
        # Word count and reading time
        word_count = len(content.split())
        reading_time = max(1, word_count // 200)  # ~200 words per minute
        
        # Basic keyword density (simplified)
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only count words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate density for top words
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
    except Exception as e:
        logger.error(f"Content analysis error: {e}")
        return {"error": str(e)}

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
    
    async def create_content(self, content_request: ContentRequest) -> ContentCreationState:
        """Execute the complete content creation workflow"""
        
        # Initialize state as a dictionary
        state: ContentCreationState = {
            "request": content_request,
            "research_data": None,
            "content_plan": None,
            "draft": None,
            "analysis": None,
            "final_content": None,
            "feedback_history": [],
            "revision_count": 0,
            "metadata": {}
        }
        
        logger.info(f"Starting content creation for: {content_request.topic}")
        
        # Execute workflow
        final_state = await self.workflow.ainvoke(state)
        
        logger.info("Content creation workflow completed successfully")
        return final_state

# =============================================================================
# DEMO AND TESTING
# =============================================================================

async def demo_content_creation():
    """Demo function to showcase the multi-agent system"""
    
    # Get Ollama configuration from environment (with defaults)
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"🤖 Using Ollama model: {model_name}")
    print(f"🌐 Ollama server: {base_url}")
    
    # Create workflow
    try:
        workflow = ContentCreationWorkflow(model_name=model_name, base_url=base_url)
        print("✅ Ollama connection established")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        print(f"💡 And the model is installed: ollama pull {model_name}")
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
    
    print("🚀 Starting Multi-Agent Content Creation System")
    print(f"📝 Topic: {request.topic}")
    print(f"🎯 Target: {request.target_audience}")
    print(f"📊 Length: {request.word_count} words")
    print("-" * 60)
    
    try:
        # Execute workflow
        result = await workflow.create_content(request)
        
        print("\n✅ Content Creation Completed Successfully!")
        print(f"📄 Final word count: {result['draft'].word_count}")
        print(f"⏱️ Reading time: {result['draft'].reading_time} minutes")
        print(f"📁 Saved to: {result['metadata'].get('output_file', 'N/A')}")
        print(f"🔍 SEO Score: {result['metadata'].get('seo_score', 'N/A')}")
        
        # Display content preview
        if result["final_content"]:
            print("\n📖 Content Preview:")
            print("-" * 40)
            preview = result["final_content"][:500] + "..." if len(result["final_content"]) > 500 else result["final_content"]
            print(preview)
        
    except Exception as e:
        print(f"❌ Error during content creation: {e}")
        logger.error(f"Content creation failed: {e}")

def main():
    """Main entry point"""
    print("Content Creation Multi-Agent System")
    print("AAIDC Module 2 Project")
    print("=" * 50)
    
    # Run demo
    asyncio.run(demo_content_creation())

if __name__ == "__main__":
    main()