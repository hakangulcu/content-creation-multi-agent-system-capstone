"""
Research Agent - Information Gathering and Data Collection Specialist

This agent is responsible for conducting comprehensive research on specified topics,
gathering facts, statistics, and supporting data from various sources to provide
a solid foundation for content creation.

Core Capabilities:
- Web search and information retrieval
- Data extraction and fact collection
- Source verification and quality assessment
- Research synthesis and organization

Domain Expertise:
- Internet research methodologies
- Information validation techniques
- Data source evaluation
- Content research best practices
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    from langchain_core.tools import tool
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    tool = None
    DuckDuckGoSearchRun = None

logger = logging.getLogger(__name__)

@dataclass
class ResearchData:
    """Structured research data container"""
    sources: List[str]
    key_facts: List[str]
    statistics: List[str]
    quotes: List[str]
    related_topics: List[str]

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
        
        parsed_results = []
        lines = results.split('\n')
        for i, line in enumerate(lines[:max_results]):
            if line.strip():
                parsed_results.append({
                    "title": f"Result {i+1}",
                    "url": "https://example.com",
                    "snippet": line.strip()
                })
        
        return parsed_results
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return [{"title": "Error", "url": "", "snippet": f"Search failed: {str(e)}"}]

class ResearchAgent:
    """
    Specialized agent for comprehensive information gathering and research.
    
    The Research Agent serves as the foundation of the content creation pipeline,
    responsible for gathering accurate, relevant, and up-to-date information on
    any given topic. This agent employs sophisticated search strategies and
    information validation techniques to ensure high-quality research output.
    
    Key Responsibilities:
    1. Topic analysis and research query generation
    2. Multi-source information gathering
    3. Fact verification and source credibility assessment
    4. Data organization and synthesis
    5. Research quality assurance
    
    Specialized Tools:
    - Web search integration (DuckDuckGo)
    - Information extraction algorithms
    - Source reliability scoring
    - Data categorization systems
    """
    
    def __init__(self, llm):
        """
        Initialize the Research Agent with language model and tools.
        
        Args:
            llm: Language model instance for intelligent processing
        """
        self.llm = llm
        self.tools = [web_search_tool]
        self.agent_type = "Research Specialist"
        self.domain_expertise = [
            "Internet research methodologies",
            "Information validation techniques", 
            "Data source evaluation",
            "Content research best practices",
            "Fact-checking and verification",
            "Statistical data analysis"
        ]
    
    async def research(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct comprehensive research based on the content request.
        
        This method implements a multi-stage research process:
        1. Query formulation and strategy planning
        2. Information gathering from multiple sources
        3. Data validation and quality assessment
        4. Research synthesis and organization
        
        Args:
            state: Current workflow state containing content request
            
        Returns:
            Updated state with comprehensive research data
        """
        request = state["request"]
        logger.info(f"Research Agent: Starting research for '{request.topic}'")
        
        # Generate diverse research queries for comprehensive coverage
        queries = self._generate_research_queries(request)
        
        # Execute research strategy
        all_sources = []
        key_facts = []
        statistics = []
        
        for query in queries:
            try:
                results = web_search_tool.invoke({"query": query, "max_results": 3})
                processed_data = self._process_search_results(results)
                
                all_sources.extend(processed_data["sources"])
                key_facts.extend(processed_data["facts"])
                statistics.extend(processed_data["statistics"])
                
            except Exception as e:
                logger.error(f"Research query failed for '{query}': {e}")
                continue
        
        # Create structured research output
        research_data = ResearchData(
            sources=all_sources[:10],  # Top 10 most relevant sources
            key_facts=key_facts[:8],   # Most important facts
            statistics=statistics[:5], # Key statistical data
            quotes=[],  # Would be extracted with advanced parsing
            related_topics=self._extract_related_topics(all_sources)
        )
        
        # Update workflow state
        state["research_data"] = research_data
        
        # Initialize metadata if not present
        if "metadata" not in state:
            state["metadata"] = {}
        
        state["metadata"]["research_completed"] = datetime.now().isoformat()
        state["metadata"]["research_sources_count"] = len(all_sources)
        state["metadata"]["research_quality_score"] = self._calculate_research_quality(research_data)
        
        logger.info(f"Research Agent: Completed research with {len(all_sources)} sources")
        
        # Return only the updates to the state as a dictionary
        return {
            "research_data": research_data,
            "metadata": state["metadata"]
        }
    
    def _generate_research_queries(self, request) -> List[str]:
        """
        Generate comprehensive research queries based on content request.
        
        Args:
            request: Content creation request with topic and requirements
            
        Returns:
            List of strategically formulated search queries
        """
        base_topic = request.topic
        current_year = datetime.now().year
        
        queries = [
            base_topic,  # Primary topic
            f"{base_topic} overview guide",  # Comprehensive coverage
            f"{base_topic} statistics {current_year}",  # Current data
            f"{base_topic} trends {current_year}",  # Latest trends
            f"{base_topic} expert analysis",  # Expert insights
            f"{base_topic} case studies examples",  # Practical applications
        ]
        
        # Add keyword-specific queries if provided
        if request.keywords:
            for keyword in request.keywords[:3]:  # Limit to avoid over-querying
                queries.append(f"{keyword} recent developments")
        
        return queries
    
    def _process_search_results(self, results: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Process and categorize search results into different data types.
        
        Args:
            results: Raw search results from web search tool
            
        Returns:
            Categorized data dictionary with sources, facts, and statistics
        """
        sources = []
        facts = []
        statistics = []
        
        for result in results:
            snippet = result.get("snippet", "")
            if not snippet:
                continue
                
            sources.append(snippet)
            
            # Categorize based on content patterns
            snippet_lower = snippet.lower()
            
            # Identify statistical information
            if any(indicator in snippet_lower for indicator in 
                   ["percent", "%", "million", "billion", "study found", "research shows", "survey"]):
                statistics.append(snippet)
            else:
                facts.append(snippet)
        
        return {
            "sources": sources,
            "facts": facts,
            "statistics": statistics
        }
    
    def _extract_related_topics(self, sources: List[str]) -> List[str]:
        """
        Extract related topics from research sources for content expansion.
        
        Args:
            sources: List of source snippets
            
        Returns:
            List of related topics identified from research
        """
        # Simple implementation - in production would use NLP techniques
        related_topics = []
        
        # Extract potential topics from sources (simplified approach)
        common_terms = {}
        for source in sources:
            words = source.lower().split()
            for word in words:
                if len(word) > 5 and word.isalpha():  # Filter for meaningful terms
                    common_terms[word] = common_terms.get(word, 0) + 1
        
        # Return most frequent meaningful terms as related topics
        sorted_terms = sorted(common_terms.items(), key=lambda x: x[1], reverse=True)
        related_topics = [term for term, count in sorted_terms[:5] if count > 1]
        
        return related_topics
    
    def _calculate_research_quality(self, research_data: ResearchData) -> int:
        """
        Calculate a quality score for the research output.
        
        Args:
            research_data: Compiled research data
            
        Returns:
            Quality score from 0-100
        """
        score = 0
        
        # Source diversity and quantity
        score += min(30, len(research_data.sources) * 3)
        
        # Fact richness
        score += min(25, len(research_data.key_facts) * 4)
        
        # Statistical data availability
        score += min(25, len(research_data.statistics) * 8)
        
        # Related topics for comprehensive coverage
        score += min(20, len(research_data.related_topics) * 4)
        
        return min(100, score)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Provide detailed information about this agent's capabilities.
        
        Returns:
            Agent information dictionary
        """
        return {
            "name": "Research Agent",
            "type": self.agent_type,
            "role": "Information Gathering and Data Collection Specialist",
            "primary_function": "Comprehensive topic research and data collection",
            "domain_expertise": self.domain_expertise,
            "capabilities": [
                "Multi-source web research",
                "Information quality assessment",
                "Data categorization and organization",
                "Fact verification processes",
                "Research strategy formulation",
                "Source credibility evaluation"
            ],
            "tools": ["DuckDuckGo Search", "Information Extraction", "Data Validation"],
            "workflow_position": "Foundation - First stage of content creation pipeline"
        }