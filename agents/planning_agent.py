"""
Planning Agent - Content Strategy and Structure Specialist

This agent is responsible for creating detailed content plans, defining structure,
and establishing strategic frameworks for content creation based on research findings
and target requirements.

Core Capabilities:
- Content strategy development
- Structural planning and organization
- Keyword strategy formulation
- Content flow optimization

Domain Expertise:
- Content architecture design
- Editorial planning methodologies
- SEO content strategy
- Audience-focused content planning
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    HumanMessage = None
    SystemMessage = None

logger = logging.getLogger(__name__)

@dataclass
class ContentPlan:
    """Structured content plan container"""
    title: str
    outline: List[str]
    key_points: List[str]
    target_keywords: List[str]
    estimated_length: int

class PlanningAgent:
    """
    Specialized agent for content strategy development and structural planning.
    
    The Planning Agent transforms raw research data into actionable content strategies,
    creating comprehensive plans that guide the content creation process. This agent
    combines editorial expertise with SEO strategy and audience analysis to produce
    optimized content frameworks.
    
    Key Responsibilities:
    1. Content strategy formulation
    2. Structural planning and organization
    3. Keyword integration strategy
    4. Content flow optimization
    5. Target audience alignment
    
    Specialized Capabilities:
    - Editorial planning methodologies
    - Content architecture design
    - SEO-focused planning
    - Audience persona integration
    - Content performance prediction
    """
    
    def __init__(self, llm):
        """
        Initialize the Planning Agent with language model capabilities.
        
        Args:
            llm: Language model instance for strategic content planning
        """
        self.llm = llm
        self.agent_type = "Content Strategy Specialist"
        self.domain_expertise = [
            "Content architecture design",
            "Editorial planning methodologies",
            "SEO content strategy",
            "Audience-focused content planning",
            "Content structure optimization",
            "Keyword integration strategies"
        ]
    
    async def plan_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive content plan based on research and requirements.
        
        This method implements a strategic planning process:
        1. Research analysis and synthesis
        2. Audience and requirement assessment
        3. Content architecture development
        4. SEO strategy integration
        5. Quality and performance optimization
        
        Args:
            state: Current workflow state with research data and requirements
            
        Returns:
            Updated state with detailed content plan
        """
        request = state["request"]
        research = state["research_data"]
        
        logger.info(f"Planning Agent: Creating content plan for '{request.topic}'")
        
        # Generate strategic planning prompt
        planning_prompt = self._create_planning_prompt(request, research)
        
        messages = [
            SystemMessage(content=self._get_system_message()),
            HumanMessage(content=planning_prompt)
        ]
        
        # Generate content plan using LLM
        try:
            response = await self.llm.ainvoke(messages)
            content_plan = self._parse_plan_response(response.content, request)
        except Exception as e:
            logger.error(f"Planning Agent: LLM error - {e}")
            # Create fallback plan
            content_plan = self._create_fallback_plan(request)
        
        # Enhance plan with strategic analysis
        enhanced_plan = self._enhance_content_plan(content_plan, request, research)
        
        # Update workflow state
        state["content_plan"] = enhanced_plan
        
        # Initialize metadata if not present
        if "metadata" not in state:
            state["metadata"] = {}
        
        state["metadata"]["planning_completed"] = datetime.now().isoformat()
        state["metadata"]["plan_complexity_score"] = self._calculate_plan_complexity(enhanced_plan)
        state["metadata"]["estimated_sections"] = len(enhanced_plan.outline)
        
        logger.info(f"Planning Agent: Content plan created - '{enhanced_plan.title}'")
        
        # Update state with content plan while preserving existing state
        updated_state = state.copy()
        updated_state["content_plan"] = enhanced_plan
        return updated_state
    
    def _create_planning_prompt(self, request, research) -> str:
        """
        Generate comprehensive planning prompt incorporating all requirements.
        
        Args:
            request: Content creation request
            research: Research data from research agent
            
        Returns:
            Detailed planning prompt for LLM
        """
        research_summary = self._summarize_research(research)
        
        return f"""
        Create a comprehensive content plan for a {request.content_type.value} about "{request.topic}".
        
        CONTENT REQUIREMENTS:
        - Target Audience: {request.target_audience}
        - Word Count: {request.word_count}
        - Tone: {request.tone}
        - Keywords: {request.keywords or 'None specified'}
        - Special Requirements: {request.special_requirements or 'None'}
        
        AVAILABLE RESEARCH DATA:
        {research_summary}
        
        PLANNING OBJECTIVES:
        1. Create an engaging, compelling title
        2. Develop a logical, well-structured outline (6-8 main sections)
        3. Define key points that must be covered in each section
        4. Identify target keywords for SEO optimization
        5. Estimate word count distribution across sections
        6. Ensure content flow serves the target audience
        
        OUTPUT FORMAT (JSON):
        {{
            "title": "Compelling, SEO-optimized title",
            "outline": ["Section 1", "Section 2", "Section 3", ...],
            "key_points": ["Key point 1", "Key point 2", ...],
            "target_keywords": ["keyword1", "keyword2", ...],
            "estimated_length": {request.word_count},
            "section_breakdown": {{
                "Section 1": {{
                    "word_count": 200,
                    "focus": "Introduction and hook",
                    "key_elements": ["Hook", "Overview", "Value proposition"]
                }},
                ...
            }}
        }}
        
        Focus on creating a plan that maximizes engagement, provides value to the target audience,
        and incorporates research findings effectively.
        """
    
    def _get_system_message(self) -> str:
        """Get the system message defining the agent's role."""
        return """
        You are an expert content strategist and editorial planner with deep expertise in:
        - Content architecture and information design
        - Audience-focused content strategy
        - SEO content optimization
        - Editorial workflow management
        - Content performance optimization
        
        Your role is to transform research data and requirements into actionable,
        strategic content plans that guide successful content creation.
        
        Always consider:
        - Target audience needs and preferences
        - Content flow and logical progression
        - SEO optimization opportunities
        - Engagement and retention strategies
        - Value delivery and actionable insights
        """
    
    def _summarize_research(self, research) -> str:
        """
        Create a concise summary of research data for planning context.
        
        Args:
            research: Research data object
            
        Returns:
            Formatted research summary
        """
        if not research:
            return "No research data available"
        
        summary_parts = []
        
        if research.key_facts:
            summary_parts.append(f"Key Facts ({len(research.key_facts)}):")
            for i, fact in enumerate(research.key_facts[:3], 1):
                summary_parts.append(f"  {i}. {fact[:100]}...")
        
        if research.statistics:
            summary_parts.append(f"\nStatistics ({len(research.statistics)}):")
            for i, stat in enumerate(research.statistics[:2], 1):
                summary_parts.append(f"  {i}. {stat[:100]}...")
        
        if research.related_topics:
            summary_parts.append(f"\nRelated Topics: {', '.join(research.related_topics[:5])}")
        
        return "\n".join(summary_parts)
    
    def _parse_plan_response(self, response_content: str, request) -> ContentPlan:
        """
        Parse LLM response into structured ContentPlan object.
        
        Args:
            response_content: Raw LLM response
            request: Original content request
            
        Returns:
            Structured ContentPlan object
        """
        try:
            # Extract JSON from response
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0]
            else:
                json_str = response_content
            
            plan_data = json.loads(json_str)
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse plan JSON, using fallback: {e}")
            # Fallback plan structure
            plan_data = self._create_fallback_plan(request)
        
        return ContentPlan(
            title=plan_data.get("title", f"Complete Guide to {request.topic}"),
            outline=plan_data.get("outline", []),
            key_points=plan_data.get("key_points", []),
            target_keywords=plan_data.get("target_keywords", request.keywords or [request.topic]),
            estimated_length=plan_data.get("estimated_length", request.word_count)
        )
    
    def _create_fallback_plan(self, request) -> ContentPlan:
        """
        Create a fallback content plan when LLM parsing fails.
        
        Args:
            request: Content creation request
            
        Returns:
            Basic ContentPlan object
        """
        topic = request.topic
        content_type = request.content_type.value
        
        return ContentPlan(
            title=f"The Complete Guide to {topic}",
            outline=[
                "Introduction and Overview",
                f"Understanding {topic}: Fundamentals",
                "Current State and Trends",
                "Key Benefits and Applications",
                "Best Practices and Strategies",
                "Challenges and Solutions",
                "Future Outlook and Predictions",
                "Conclusion and Next Steps"
            ],
            key_points=[
                f"Provide comprehensive overview of {topic}",
                "Include current statistics and trends",
                "Offer actionable insights and recommendations",
                "Address common challenges and solutions",
                "Maintain engaging, accessible tone"
            ],
            target_keywords=request.keywords or [topic],
            estimated_length=request.word_count
        )
    
    def _enhance_content_plan(self, plan: ContentPlan, request, research) -> ContentPlan:
        """
        Enhance content plan with additional strategic elements.
        
        Args:
            plan: Basic content plan
            request: Content creation request
            research: Research data
            
        Returns:
            Enhanced ContentPlan with strategic improvements
        """
        # Enhance keywords with research insights
        enhanced_keywords = list(plan.target_keywords)
        if research and research.related_topics:
            enhanced_keywords.extend(research.related_topics[:3])
        
        # Remove duplicates while preserving order
        seen = set()
        enhanced_keywords = [x for x in enhanced_keywords if not (x in seen or seen.add(x))]
        
        # Enhance key points with research-driven insights
        enhanced_key_points = list(plan.key_points)
        if research and research.statistics:
            enhanced_key_points.append("Incorporate latest statistical data and research findings")
        
        return ContentPlan(
            title=plan.title,
            outline=plan.outline,
            key_points=enhanced_key_points,
            target_keywords=enhanced_keywords[:8],  # Limit to 8 keywords
            estimated_length=plan.estimated_length
        )
    
    def _calculate_plan_complexity(self, plan: ContentPlan) -> int:
        """
        Calculate complexity score for the content plan.
        
        Args:
            plan: Content plan to analyze
            
        Returns:
            Complexity score from 0-100
        """
        score = 0
        
        # Outline depth and structure
        score += min(30, len(plan.outline) * 4)
        
        # Key points comprehensiveness
        score += min(25, len(plan.key_points) * 3)
        
        # Keyword strategy depth
        score += min(25, len(plan.target_keywords) * 3)
        
        # Title quality (simple heuristic)
        title_score = min(20, len(plan.title.split()) * 2)
        score += title_score
        
        return min(100, score)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Provide detailed information about this agent's capabilities.
        
        Returns:
            Agent information dictionary
        """
        return {
            "name": "Planning Agent",
            "type": self.agent_type,
            "role": "Content Strategy and Structure Specialist",
            "primary_function": "Strategic content planning and architectural design",
            "domain_expertise": self.domain_expertise,
            "capabilities": [
                "Content strategy formulation",
                "Structural planning and organization",
                "SEO strategy integration",
                "Audience-focused planning",
                "Editorial workflow design",
                "Content architecture optimization"
            ],
            "tools": ["Strategic Analysis", "Content Architecture", "SEO Planning"],
            "workflow_position": "Strategy - Second stage, transforms research into actionable plans"
        }