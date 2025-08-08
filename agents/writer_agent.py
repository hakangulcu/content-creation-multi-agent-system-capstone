"""
Writer Agent - Content Creation and Composition Specialist

This agent is responsible for transforming content plans and research data into
well-crafted, engaging written content that meets specific requirements and
maintains consistent quality throughout the creation process.

Core Capabilities:
- Content composition and writing
- Tone and style consistency
- Research integration and synthesis
- Narrative flow optimization

Domain Expertise:
- Professional writing techniques
- Content structure and flow
- Audience-appropriate communication
- Research-based content development
"""

import logging
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    HumanMessage = None
    SystemMessage = None

logger = logging.getLogger(__name__)

@dataclass
class ContentDraft:
    """Structured content draft container"""
    title: str
    content: str
    word_count: int
    reading_time: int

class WriterAgent:
    """
    Specialized agent for content creation and composition.
    
    The Writer Agent is the creative core of the content creation pipeline,
    responsible for transforming strategic plans and research data into compelling,
    well-structured written content. This agent combines advanced writing techniques
    with audience awareness and research integration to produce high-quality drafts.
    
    Key Responsibilities:
    1. Content composition and creation
    2. Research integration and synthesis
    3. Tone and style consistency maintenance
    4. Narrative flow and structure optimization
    5. Audience engagement optimization
    
    Specialized Capabilities:
    - Advanced content composition techniques
    - Multi-format writing expertise
    - Research-driven content development
    - Audience-specific communication
    - SEO-conscious writing practices
    """
    
    def __init__(self, llm):
        """
        Initialize the Writer Agent with language model capabilities.
        
        Args:
            llm: Language model instance for content generation
        """
        self.llm = llm
        self.agent_type = "Content Creation Specialist"
        self.domain_expertise = [
            "Professional writing techniques",
            "Content structure and flow optimization",
            "Audience-appropriate communication",
            "Research-based content development",
            "Multi-format content creation",
            "Narrative construction and storytelling"
        ]
    
    async def write_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate high-quality content based on plan and research data.
        
        This method implements a comprehensive writing process:
        1. Content planning analysis and interpretation
        2. Research data integration strategy
        3. Audience-focused content creation
        4. Quality and coherence optimization
        5. Format-specific adaptation
        
        Args:
            state: Current workflow state with plan and research data
            
        Returns:
            Updated state with completed content draft
        """
        request = state["request"]
        plan = state["content_plan"]
        research = state["research_data"]
        
        logger.info(f"Writer Agent: Creating content for '{plan.title}'")
        
        # Generate comprehensive writing prompt
        writing_prompt = self._create_writing_prompt(request, plan, research)
        
        messages = [
            SystemMessage(content=self._get_system_message(request)),
            HumanMessage(content=writing_prompt)
        ]
        
        # Generate content using LLM
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        # Post-process and enhance content
        processed_content = self._post_process_content(content, request, plan)
        
        # Calculate content metrics
        word_count = len(processed_content.split())
        reading_time = max(1, word_count // 200)  # ~200 words per minute
        
        # Create content draft
        draft = ContentDraft(
            title=plan.title,
            content=processed_content,
            word_count=word_count,
            reading_time=reading_time
        )
        
        # Update workflow state
        state["draft"] = draft
        
        # Initialize metadata if not present
        if "metadata" not in state:
            state["metadata"] = {}
        
        state["metadata"]["writing_completed"] = datetime.now().isoformat()
        state["metadata"]["content_quality_score"] = self._assess_content_quality(draft, request)
        state["metadata"]["research_integration_score"] = self._assess_research_integration(draft, research)
        
        logger.info(f"Writer Agent: Content created - {word_count} words, {reading_time} min read")
        
        # Return only the updates to the state as a dictionary
        return {
            "draft": draft,
            "metadata": state["metadata"]
        }
    
    def _create_writing_prompt(self, request, plan, research) -> str:
        """
        Generate comprehensive writing prompt incorporating all elements.
        
        Args:
            request: Content creation request
            plan: Content plan from planning agent
            research: Research data from research agent
            
        Returns:
            Detailed writing prompt for content generation
        """
        research_context = self._format_research_context(research)
        outline_structure = self._format_outline_structure(plan)
        
        return f"""
        Create a high-quality {request.content_type.value} based on the comprehensive plan and research provided.
        
        CONTENT SPECIFICATIONS:
        - Title: {plan.title}
        - Target Length: {request.word_count} words
        - Target Audience: {request.target_audience}
        - Tone: {request.tone}
        - Content Type: {request.content_type.value}
        - Special Requirements: {request.special_requirements or 'None'}
        
        CONTENT STRUCTURE TO FOLLOW:
        {outline_structure}
        
        KEY POINTS TO INCORPORATE:
        {self._format_key_points(plan.key_points)}
        
        RESEARCH DATA TO INTEGRATE:
        {research_context}
        
        TARGET KEYWORDS TO INCLUDE NATURALLY:
        {', '.join(plan.target_keywords)}
        
        WRITING REQUIREMENTS:
        1. Create engaging, well-structured content that flows naturally
        2. Include all sections from the provided outline in logical order
        3. Integrate research data and statistics seamlessly into the narrative
        4. Maintain consistent tone and voice throughout
        5. Use markdown formatting for headings and structure
        6. Write in a style appropriate for the target audience
        7. Include compelling introductions and strong conclusions
        8. Ensure each section provides value and advances the overall narrative
        9. Target approximately {request.word_count} words total
        10. Make the content actionable and practical where appropriate
        
        CONTENT QUALITY STANDARDS:
        - Clear, engaging writing that captures and maintains reader attention
        - Logical flow between sections with smooth transitions
        - Evidence-based content supported by research findings
        - Practical insights and actionable recommendations
        - Professional formatting and presentation
        
        Write the complete {request.content_type.value} now, ensuring it meets all requirements
        and provides exceptional value to the target audience:
        """
    
    def _get_system_message(self, request) -> str:
        """
        Generate system message defining the agent's writing persona.
        
        Args:
            request: Content creation request for context
            
        Returns:
            System message for writing agent
        """
        content_type = request.content_type.value
        
        return f"""
        You are an expert {content_type} writer with extensive experience in creating
        high-quality, engaging content across multiple industries and formats.
        
        Your expertise includes:
        - Advanced writing techniques and storytelling
        - Audience analysis and targeted communication
        - Research integration and fact-based writing
        - SEO-conscious content creation
        - Multi-format content adaptation
        - Professional editing and refinement
        
        Your writing philosophy:
        - Always prioritize value delivery to the reader
        - Create content that is both informative and engaging
        - Support claims with research and evidence
        - Maintain clarity and accessibility
        - Build logical, compelling narratives
        - Optimize for both human readers and search engines
        
        For this {content_type}, focus on creating content that serves the target
        audience's needs while achieving the specified objectives and requirements.
        """
    
    def _format_research_context(self, research) -> str:
        """
        Format research data for integration guidance.
        
        Args:
            research: Research data object
            
        Returns:
            Formatted research context for writing prompt
        """
        if not research:
            return "No specific research data available - rely on general knowledge"
        
        context_parts = []
        
        if research.key_facts:
            context_parts.append("FACTS TO INCORPORATE:")
            for i, fact in enumerate(research.key_facts[:5], 1):
                context_parts.append(f"  {i}. {fact}")
        
        if research.statistics:
            context_parts.append("\nSTATISTICS TO INCLUDE:")
            for i, stat in enumerate(research.statistics[:3], 1):
                context_parts.append(f"  {i}. {stat}")
        
        if research.related_topics:
            context_parts.append(f"\nRELATED TOPICS TO CONSIDER: {', '.join(research.related_topics[:5])}")
        
        return "\n".join(context_parts) if context_parts else "General research context available"
    
    def _format_outline_structure(self, plan) -> str:
        """
        Format content outline for writing guidance.
        
        Args:
            plan: Content plan object
            
        Returns:
            Formatted outline structure
        """
        if not plan.outline:
            return "No specific outline provided - use standard structure"
        
        structure_parts = []
        for i, section in enumerate(plan.outline, 1):
            # Estimate word count per section
            section_words = plan.estimated_length // len(plan.outline)
            structure_parts.append(f"{i}. {section} (~{section_words} words)")
        
        return "\n".join(structure_parts)
    
    def _format_key_points(self, key_points) -> str:
        """
        Format key points for writing guidance.
        
        Args:
            key_points: List of key points to incorporate
            
        Returns:
            Formatted key points
        """
        if not key_points:
            return "No specific key points provided"
        
        return "\n".join([f"- {point}" for point in key_points])
    
    def _post_process_content(self, content: str, request, plan) -> str:
        """
        Post-process generated content for quality and consistency.
        
        Args:
            content: Raw generated content
            request: Content creation request
            plan: Content plan
            
        Returns:
            Post-processed, refined content
        """
        # Basic post-processing operations
        processed = content.strip()
        
        # Ensure proper markdown formatting
        if not processed.startswith('#'):
            processed = f"# {plan.title}\n\n{processed}"
        
        # Clean up excessive whitespace
        lines = processed.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            if line.strip() == '':
                if not prev_empty:
                    cleaned_lines.append('')
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        return '\n'.join(cleaned_lines)
    
    def _assess_content_quality(self, draft: ContentDraft, request) -> int:
        """
        Assess the quality of generated content.
        
        Args:
            draft: Content draft to assess
            request: Original content request
            
        Returns:
            Quality score from 0-100
        """
        score = 0
        
        # Word count accuracy
        target_words = request.word_count
        actual_words = draft.word_count
        word_accuracy = 1 - abs(target_words - actual_words) / target_words
        score += int(word_accuracy * 30)
        
        # Content structure (basic heuristics)
        lines = draft.content.split('\n')
        headings = [line for line in lines if line.startswith('#')]
        score += min(25, len(headings) * 4)  # Reward good structure
        
        # Content length appropriateness
        if 500 <= draft.word_count <= 5000:  # Reasonable range
            score += 25
        
        # Title quality
        if draft.title and len(draft.title.split()) >= 4:
            score += 20
        
        return min(100, score)
    
    def _assess_research_integration(self, draft: ContentDraft, research) -> int:
        """
        Assess how well research data was integrated into content.
        
        Args:
            draft: Content draft to assess
            research: Original research data
            
        Returns:
            Integration score from 0-100
        """
        if not research:
            return 50  # Neutral score when no research available
        
        score = 0
        content_lower = draft.content.lower()
        
        # Check for research fact integration
        if research.key_facts:
            integrated_facts = 0
            for fact in research.key_facts:
                # Simple keyword matching (would be more sophisticated in production)
                fact_words = fact.lower().split()[:5]  # First 5 words as key identifiers
                for word in fact_words:
                    if len(word) > 4 and word in content_lower:
                        integrated_facts += 1
                        break
            score += min(40, (integrated_facts / len(research.key_facts)) * 40)
        
        # Check for statistics integration
        if research.statistics:
            integrated_stats = 0
            for stat in research.statistics:
                # Look for numerical patterns
                if any(char.isdigit() for char in stat):
                    stat_numbers = ''.join(c for c in stat if c.isdigit() or c == '.')
                    if stat_numbers and stat_numbers in draft.content:
                        integrated_stats += 1
            score += min(30, (integrated_stats / len(research.statistics)) * 30)
        
        # Check for related topics integration
        if research.related_topics:
            integrated_topics = 0
            for topic in research.related_topics:
                if topic.lower() in content_lower:
                    integrated_topics += 1
            score += min(30, (integrated_topics / len(research.related_topics)) * 30)
        
        return min(100, score)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Provide detailed information about this agent's capabilities.
        
        Returns:
            Agent information dictionary
        """
        return {
            "name": "Writer Agent",
            "type": self.agent_type,
            "role": "Content Creation and Composition Specialist",
            "primary_function": "High-quality content creation and composition",
            "domain_expertise": self.domain_expertise,
            "capabilities": [
                "Advanced content composition",
                "Research integration and synthesis",
                "Audience-focused writing",
                "Multi-format content creation",
                "Tone and style consistency",
                "Narrative flow optimization"
            ],
            "tools": ["Advanced Language Generation", "Content Synthesis", "Style Adaptation"],
            "workflow_position": "Creation - Third stage, transforms plans into actual content"
        }