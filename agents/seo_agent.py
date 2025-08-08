"""
SEO Agent - Search Engine Optimization Specialist

This agent is responsible for optimizing content for search engines while
maintaining quality and readability, ensuring maximum visibility and
discoverability without compromising user experience.

Core Capabilities:
- SEO analysis and optimization
- Keyword integration and density management
- Content structure optimization for search
- SEO performance assessment

Domain Expertise:
- Search engine optimization techniques
- Keyword research and analysis
- Content SEO best practices
- Search visibility optimization
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
import re

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool
except ImportError:
    HumanMessage = None
    SystemMessage = None
    tool = None

logger = logging.getLogger(__name__)

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
        
        # Check keyword presence and density
        keyword_analysis = {}
        for keyword in target_keywords:
            count = content_lower.count(keyword.lower())
            keyword_analysis[keyword] = count
            
            if count == 0:
                suggestions.append(f"Consider adding the keyword '{keyword}' to your content")
            elif count > 10:
                suggestions.append(f"Keyword '{keyword}' may be overused ({count} times)")
        
        # Check title and headings structure
        lines = content.split('\n')
        headings = [line for line in lines if line.startswith('#')]
        
        if not headings:
            suggestions.append("Add a compelling title using # markdown")
        elif len(headings) < 3:
            suggestions.append("Consider adding more subheadings (H2, H3) to improve structure")
        
        # Check content length for SEO
        word_count = len(content.split())
        if word_count < 300:
            suggestions.append("Content is quite short for SEO - consider expanding to at least 300 words")
        elif word_count > 3000:
            suggestions.append("Content is very long - consider breaking into sections or multiple pages")
        
        # Check for meta-descriptions or summaries
        if not any("summary" in line.lower() or "overview" in line.lower() for line in lines[:5]):
            suggestions.append("Consider adding a summary or overview section near the beginning")
        
        # Calculate SEO score based on various factors
        seo_score = 70  # Base score
        seo_score += min(15, len(headings) * 3)  # Reward good structure
        seo_score += min(10, len([k for k in keyword_analysis.values() if k > 0]) * 2)  # Keyword presence
        seo_score -= len(suggestions) * 3  # Deduct for issues found
        seo_score = max(0, min(100, seo_score))
        
        return {
            "keyword_analysis": keyword_analysis,
            "suggestions": suggestions,
            "word_count": word_count,
            "heading_count": len(headings),
            "seo_score": seo_score
        }
    except Exception as e:
        logger.error(f"SEO analysis error: {e}")
        return {"error": str(e)}

class SEOAgent:
    """
    Specialized agent for search engine optimization and content visibility enhancement.
    
    The SEO Agent focuses on optimizing content for search engines while maintaining
    quality and user experience. This agent implements best practices for SEO,
    keyword optimization, and content structure to maximize search visibility
    and organic discovery.
    
    Key Responsibilities:
    1. SEO analysis and assessment
    2. Keyword optimization and integration
    3. Content structure optimization for search
    4. Meta-information optimization
    5. Search visibility enhancement
    
    Specialized Capabilities:
    - Advanced SEO analysis and optimization
    - Keyword research and density management
    - Content structure optimization
    - Search engine best practices implementation
    - Organic visibility enhancement
    """
    
    def __init__(self, llm):
        """
        Initialize the SEO Agent with language model and optimization tools.
        
        Args:
            llm: Language model instance for SEO optimization
        """
        self.llm = llm
        self.tools = [seo_optimization_tool]
        self.agent_type = "SEO Optimization Specialist"
        self.domain_expertise = [
            "Search engine optimization techniques",
            "Keyword research and analysis",
            "Content SEO best practices",
            "Search visibility optimization",
            "Technical SEO implementation",
            "Content structure optimization"
        ]
    
    async def optimize_seo(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize content for search engines while maintaining quality and readability.
        
        This method implements comprehensive SEO optimization:
        1. Content SEO analysis and assessment
        2. Keyword optimization and integration
        3. Structure enhancement for search visibility
        4. Meta-information optimization
        5. Performance validation and improvement
        
        Args:
            state: Current workflow state with content draft
            
        Returns:
            Updated state with SEO-optimized content
        """
        draft = state["draft"]
        plan = state["content_plan"]
        request = state["request"]
        
        logger.info(f"SEO Agent: Optimizing content for search engines")
        
        # Perform comprehensive SEO analysis
        try:
            from main import _seo_optimization_function
            seo_analysis = _seo_optimization_function(draft.content, plan.target_keywords)
        except ImportError:
            # Fallback to local tool if import fails
            seo_analysis = seo_optimization_tool.invoke({
                "content": draft.content,
                "target_keywords": plan.target_keywords
            })
        
        # Determine if optimization is needed
        optimization_needed = self._assess_optimization_needs(seo_analysis, request)
        
        if optimization_needed:
            # Create SEO optimization prompt
            optimization_prompt = self._create_seo_prompt(draft, plan, seo_analysis, request)
            
            messages = [
                SystemMessage(content=self._get_system_message()),
                HumanMessage(content=optimization_prompt)
            ]
            
            # Generate SEO-optimized content
            response = await self.llm.ainvoke(messages)
            optimized_content = response.content
            
            # Post-process optimized content
            refined_content = self._post_process_seo_content(optimized_content, draft, plan)
            
            # Update content metrics
            word_count = len(refined_content.split())
            reading_time = max(1, word_count // 200)
            
            # Create updated draft
            from .writer_agent import ContentDraft
            updated_draft = ContentDraft(
                title=draft.title,
                content=refined_content,
                word_count=word_count,
                reading_time=reading_time
            )
            
            state["draft"] = updated_draft
            
            # Perform final SEO analysis
            try:
                from main import _seo_optimization_function
                final_seo_analysis = _seo_optimization_function(refined_content, plan.target_keywords)
            except ImportError:
                # Fallback to local tool if import fails
                final_seo_analysis = seo_optimization_tool.invoke({
                    "content": refined_content,
                    "target_keywords": plan.target_keywords
                })
            
            logger.info(f"SEO Agent: Content optimized - SEO score improved to {final_seo_analysis.get('seo_score', 'N/A')}")
        else:
            final_seo_analysis = seo_analysis
            logger.info(f"SEO Agent: No optimization needed - current SEO score: {seo_analysis.get('seo_score', 'N/A')}")
        
        # Update workflow state with SEO metrics
        # Initialize metadata if not present
        if "metadata" not in state:
            state["metadata"] = {}
        
        state["metadata"]["seo_optimization_completed"] = datetime.now().isoformat()
        state["metadata"]["seo_score"] = final_seo_analysis.get("seo_score", 0)
        state["metadata"]["keyword_coverage"] = self._calculate_keyword_coverage(
            final_seo_analysis.get("keyword_analysis", {}), plan.target_keywords
        )
        state["metadata"]["seo_improvement_score"] = self._calculate_seo_improvement(
            seo_analysis.get("seo_score", 0), final_seo_analysis.get("seo_score", 0)
        )
        
        return state
    
    def _assess_optimization_needs(self, seo_analysis: Dict[str, Any], request) -> bool:
        """
        Assess whether SEO optimization is needed based on analysis results.
        
        Args:
            seo_analysis: Current SEO analysis results
            request: Original content request
            
        Returns:
            Boolean indicating if optimization is needed
        """
        seo_score = seo_analysis.get("seo_score", 0)
        suggestions = seo_analysis.get("suggestions", [])
        keyword_analysis = seo_analysis.get("keyword_analysis", {})
        
        # Optimization needed if:
        # 1. SEO score is below threshold
        if seo_score < 70:
            return True
        
        # 2. Multiple suggestions for improvement
        if len(suggestions) > 2:
            return True
        
        # 3. Poor keyword coverage
        keywords_missing = sum(1 for count in keyword_analysis.values() if count == 0)
        if keywords_missing > len(keyword_analysis) * 0.3:  # More than 30% missing
            return True
        
        return False
    
    def _create_seo_prompt(self, draft, plan, seo_analysis, request) -> str:
        """
        Create comprehensive SEO optimization prompt.
        
        Args:
            draft: Current content draft
            plan: Content plan with keywords
            seo_analysis: SEO analysis results
            request: Original content request
            
        Returns:
            Detailed SEO optimization prompt
        """
        keyword_status = self._format_keyword_status(seo_analysis.get("keyword_analysis", {}))
        suggestions_text = self._format_seo_suggestions(seo_analysis.get("suggestions", []))
        
        return f"""
        Optimize the following content for search engines while maintaining high quality and readability.
        
        CURRENT CONTENT:
        {draft.content}
        
        SEO ANALYSIS RESULTS:
        - Current SEO Score: {seo_analysis.get('seo_score', 'Unknown')}/100
        - Word Count: {seo_analysis.get('word_count', 'Unknown')}
        - Heading Count: {seo_analysis.get('heading_count', 'Unknown')}
        
        TARGET KEYWORDS AND CURRENT STATUS:
        {keyword_status}
        
        SEO IMPROVEMENT RECOMMENDATIONS:
        {suggestions_text}
        
        CONTENT REQUIREMENTS:
        - Target Audience: {request.target_audience}
        - Content Type: {request.content_type.value}
        - Tone: {request.tone}
        - Target Word Count: {request.word_count}
        
        SEO OPTIMIZATION OBJECTIVES:
        1. KEYWORD OPTIMIZATION
           - Naturally integrate all target keywords throughout the content
           - Ensure primary keywords appear in title and headings
           - Maintain optimal keyword density (1-3% per keyword)
           - Use semantic variations and related terms
        
        2. CONTENT STRUCTURE
           - Optimize heading hierarchy (H1, H2, H3) for search engines
           - Create clear, descriptive headings that include keywords
           - Improve content scannability with proper formatting
           - Add internal structure that search engines can understand
        
        3. CONTENT QUALITY FOR SEO
           - Ensure comprehensive coverage of the topic
           - Add valuable, unique insights that users seek
           - Include related topics and questions users might have
           - Maintain natural, engaging writing style
        
        4. TECHNICAL SEO ELEMENTS
           - Optimize content length for search visibility
           - Ensure proper use of markdown formatting
           - Create content that encourages engagement and sharing
           - Structure for featured snippets when possible
        
        OPTIMIZATION GUIDELINES:
        - Maintain the original content quality and value
        - Keep the tone and style appropriate for the target audience
        - Ensure keyword integration feels natural and not forced
        - Preserve all important information and key messages
        - Focus on user experience while optimizing for search engines
        - Use markdown formatting effectively for structure
        
        CRITICAL REQUIREMENTS:
        - Do not sacrifice readability for SEO
        - Maintain engaging, valuable content for human readers
        - Ensure all keyword integration is contextually appropriate
        - Preserve the original content structure and flow
        
        Return the fully SEO-optimized content that addresses all recommendations
        while maintaining high quality and reader engagement:
        """
    
    def _get_system_message(self) -> str:
        """Get system message defining the SEO agent's role."""
        return """
        You are an expert SEO specialist with deep knowledge of search engine
        optimization, content marketing, and organic visibility strategies.
        
        Your expertise includes:
        - Advanced SEO techniques and best practices
        - Keyword research and optimization strategies
        - Content structure optimization for search engines
        - Technical SEO implementation
        - User experience optimization for SEO
        - Content marketing for organic growth
        
        Your SEO philosophy:
        - Always prioritize user experience over search engine manipulation
        - Create content that serves both users and search engines effectively
        - Implement white-hat SEO techniques exclusively
        - Focus on long-term organic visibility and authority building
        - Balance optimization with content quality and readability
        - Understand that great content is the foundation of great SEO
        
        Your goal is to optimize content for maximum search visibility while
        maintaining or improving content quality, user engagement, and value delivery.
        """
    
    def _format_keyword_status(self, keyword_analysis: Dict[str, int]) -> str:
        """
        Format keyword analysis results for optimization guidance.
        
        Args:
            keyword_analysis: Dictionary of keywords and their counts
            
        Returns:
            Formatted keyword status report
        """
        if not keyword_analysis:
            return "No keyword analysis available"
        
        status_lines = []
        for keyword, count in keyword_analysis.items():
            if count == 0:
                status_lines.append(f"• '{keyword}': NOT FOUND - Needs integration")
            elif count <= 2:
                status_lines.append(f"• '{keyword}': {count} times - Could use more mentions")
            elif count <= 8:
                status_lines.append(f"• '{keyword}': {count} times - Good coverage")
            else:
                status_lines.append(f"• '{keyword}': {count} times - May be overused")
        
        return "\n".join(status_lines)
    
    def _format_seo_suggestions(self, suggestions: List[str]) -> str:
        """
        Format SEO suggestions for optimization prompt.
        
        Args:
            suggestions: List of SEO improvement suggestions
            
        Returns:
            Formatted suggestions text
        """
        if not suggestions:
            return "No specific suggestions - content appears well-optimized"
        
        return "\n".join([f"• {suggestion}" for suggestion in suggestions])
    
    def _post_process_seo_content(self, optimized_content: str, original_draft, plan) -> str:
        """
        Post-process SEO-optimized content for final quality assurance.
        
        Args:
            optimized_content: SEO-optimized content from LLM
            original_draft: Original content draft
            plan: Content plan with keywords
            
        Returns:
            Final refined SEO-optimized content
        """
        # Clean up the optimized content
        refined = optimized_content.strip()
        
        # Ensure title preservation
        if not refined.startswith('#'):
            refined = f"# {original_draft.title}\n\n{refined}"
        
        # Validate keyword integration (basic check)
        refined_lower = refined.lower()
        for keyword in plan.target_keywords:
            if keyword.lower() not in refined_lower:
                logger.warning(f"SEO optimization may have missed keyword: {keyword}")
        
        # Clean up formatting
        lines = refined.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line == '':
                if not prev_empty:
                    cleaned_lines.append('')
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        return '\n'.join(cleaned_lines)
    
    def _calculate_keyword_coverage(self, keyword_analysis: Dict[str, int], target_keywords: List[str]) -> float:
        """
        Calculate keyword coverage percentage.
        
        Args:
            keyword_analysis: Analysis results with keyword counts
            target_keywords: List of target keywords
            
        Returns:
            Coverage percentage
        """
        try:
            if not target_keywords or len(target_keywords) == 0:
                return 100.0
        except (TypeError, AttributeError):
            # Handle MagicMock or other non-list objects
            return 100.0
        
        covered_keywords = sum(1 for keyword in target_keywords 
                             if keyword_analysis.get(keyword, 0) > 0)
        
        return round((covered_keywords / len(target_keywords)) * 100, 2)
    
    def _calculate_seo_improvement(self, before_score: int, after_score: int) -> float:
        """
        Calculate SEO improvement percentage.
        
        Args:
            before_score: SEO score before optimization
            after_score: SEO score after optimization
            
        Returns:
            Improvement percentage
        """
        if before_score == 0:
            return 0.0
        
        improvement = ((after_score - before_score) / before_score) * 100
        return round(improvement, 2)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Provide detailed information about this agent's capabilities.
        
        Returns:
            Agent information dictionary
        """
        return {
            "name": "SEO Agent",
            "type": self.agent_type,
            "role": "Search Engine Optimization Specialist",
            "primary_function": "Content optimization for search engine visibility and organic discovery",
            "domain_expertise": self.domain_expertise,
            "capabilities": [
                "Advanced SEO analysis and optimization",
                "Keyword research and density management",
                "Content structure optimization for search",
                "Meta-information optimization",
                "Search visibility enhancement",
                "Technical SEO implementation"
            ],
            "tools": ["SEO Analysis Tool", "Keyword Optimization", "Search Visibility Assessment"],
            "workflow_position": "Optimization - Fifth stage, enhances content for search engine visibility"
        }