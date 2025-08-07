"""
Editor Agent - Content Refinement and Quality Enhancement Specialist

This agent is responsible for reviewing, editing, and improving content quality,
ensuring clarity, readability, and adherence to style guidelines while maintaining
the original intent and message effectiveness.

Core Capabilities:
- Content editing and refinement
- Readability optimization
- Style and tone consistency
- Quality assurance and improvement

Domain Expertise:
- Professional editing techniques
- Readability analysis and optimization
- Content flow and structure improvement
- Grammar and style enhancement
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

try:
    import nltk
except ImportError:
    nltk = None
    
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
except ImportError:
    flesch_reading_ease = None
    flesch_kincaid_grade = None
    
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool
except ImportError:
    HumanMessage = None
    SystemMessage = None
    tool = None

logger = logging.getLogger(__name__)

@dataclass
class ContentAnalysis:
    """Content analysis results container"""
    readability_score: float
    grade_level: float
    keyword_density: Dict[str, float]
    suggestions: List[str]

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
        # Download required NLTK data if available
        if nltk:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
        
        # Calculate readability metrics if available
        if flesch_reading_ease and flesch_kincaid_grade:
            readability = flesch_reading_ease(content)
            grade_level = flesch_kincaid_grade(content)
        else:
            readability = 50.0  # Default fallback
            grade_level = 10.0  # Default fallback
        
        # Word count and reading time
        word_count = len(content.split())
        reading_time = max(1, word_count // 200)  # ~200 words per minute
        
        # Basic keyword density analysis
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

class EditorAgent:
    """
    Specialized agent for content editing and quality enhancement.
    
    The Editor Agent serves as the quality control specialist in the content creation
    pipeline, focusing on improving readability, clarity, and overall content quality.
    This agent combines advanced editing techniques with analytical tools to ensure
    content meets professional standards and audience expectations.
    
    Key Responsibilities:
    1. Content quality assessment and improvement
    2. Readability analysis and optimization
    3. Style and tone consistency enforcement
    4. Grammar and language enhancement
    5. Content flow and structure refinement
    
    Specialized Capabilities:
    - Advanced editing and proofreading techniques
    - Readability analysis and optimization
    - Content structure enhancement
    - Audience-appropriate language adjustment
    - Quality metrics assessment and improvement
    """
    
    def __init__(self, llm):
        """
        Initialize the Editor Agent with language model and analysis tools.
        
        Args:
            llm: Language model instance for content editing
        """
        self.llm = llm
        self.tools = [content_analysis_tool]
        self.agent_type = "Content Quality Specialist"
        self.domain_expertise = [
            "Professional editing techniques",
            "Readability analysis and optimization",
            "Content flow and structure improvement",
            "Grammar and style enhancement",
            "Quality assurance methodologies",
            "Audience-focused content refinement"
        ]
    
    async def edit_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Edit and improve content quality through comprehensive analysis and refinement.
        
        This method implements a multi-stage editing process:
        1. Content quality analysis and assessment
        2. Readability evaluation and optimization
        3. Style and tone consistency review
        4. Structural improvement and flow enhancement
        5. Final quality validation and metrics
        
        Args:
            state: Current workflow state with content draft
            
        Returns:
            Updated state with edited, improved content
        """
        draft = state["draft"]
        request = state["request"]
        
        logger.info(f"Editor Agent: Editing content - {draft.word_count} words")
        
        # Analyze current content quality
        analysis_result = content_analysis_tool.invoke({"content": draft.content})
        
        # Create comprehensive editing prompt
        editing_prompt = self._create_editing_prompt(draft, request, analysis_result)
        
        messages = [
            SystemMessage(content=self._get_system_message()),
            HumanMessage(content=editing_prompt)
        ]
        
        # Generate edited content
        response = await self.llm.ainvoke(messages)
        edited_content = response.content
        
        # Post-process edited content
        refined_content = self._post_process_edited_content(edited_content, draft, request)
        
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
        
        # Perform final analysis on edited content
        final_analysis_result = content_analysis_tool.invoke({"content": refined_content})
        
        # Create content analysis object
        content_analysis = ContentAnalysis(
            readability_score=final_analysis_result.get('readability_score', 0),
            grade_level=final_analysis_result.get('grade_level', 0),
            keyword_density=final_analysis_result.get('keyword_density', {}),
            suggestions=self._generate_improvement_suggestions(final_analysis_result, request)
        )
        
        # Update workflow state
        state["draft"] = updated_draft
        state["analysis"] = content_analysis
        state["metadata"]["editing_completed"] = datetime.now().isoformat()
        state["metadata"]["readability_improvement"] = self._calculate_readability_improvement(
            analysis_result, final_analysis_result
        )
        state["metadata"]["editing_quality_score"] = self._calculate_editing_quality_score(
            updated_draft, content_analysis, request
        )
        
        logger.info(f"Editor Agent: Editing completed - {word_count} words, "
                   f"readability: {final_analysis_result.get('readability_score', 'N/A')}")
        
        return state
    
    def _create_editing_prompt(self, draft, request, analysis_result) -> str:
        """
        Create comprehensive editing prompt with specific improvement targets.
        
        Args:
            draft: Current content draft
            request: Original content request
            analysis_result: Content analysis metrics
            
        Returns:
            Detailed editing prompt for content improvement
        """
        readability_feedback = self._interpret_readability_scores(analysis_result)
        improvement_areas = self._identify_improvement_areas(analysis_result, request)
        
        return f"""
        Please edit and significantly improve the following content to meet professional standards.
        
        CURRENT CONTENT:
        {draft.content}
        
        CURRENT CONTENT ANALYSIS:
        - Word count: {analysis_result.get('word_count', 'Unknown')}
        - Readability score: {analysis_result.get('readability_score', 'Unknown')} {readability_feedback}
        - Grade level: {analysis_result.get('grade_level', 'Unknown')}
        - Reading time: {analysis_result.get('reading_time', 'Unknown')} minutes
        
        TARGET REQUIREMENTS:
        - Target word count: {request.word_count} words
        - Target audience: {request.target_audience}
        - Desired tone: {request.tone}
        - Content type: {request.content_type.value}
        
        SPECIFIC IMPROVEMENT AREAS:
        {improvement_areas}
        
        EDITING OBJECTIVES:
        1. CLARITY & READABILITY
           - Improve sentence structure and flow
           - Eliminate unnecessary jargon or complexity
           - Ensure clear, logical progression of ideas
           - Optimize for target audience comprehension
        
        2. ENGAGEMENT & STYLE
           - Enhance opening hooks and compelling introductions
           - Improve transitions between sections
           - Maintain consistent tone throughout
           - Add engaging elements appropriate for content type
        
        3. CONTENT QUALITY
           - Strengthen key arguments and supporting evidence
           - Improve factual accuracy and precision
           - Enhance practical value and actionability
           - Ensure comprehensive coverage of topic
        
        4. STRUCTURE & ORGANIZATION
           - Optimize content flow and logical sequence
           - Improve headings and subheadings for clarity
           - Ensure balanced section lengths
           - Enhance overall readability and scannability
        
        5. LENGTH OPTIMIZATION
           - Adjust content length to target word count
           - Eliminate redundant or unnecessary content
           - Add valuable content if under target length
           - Maintain quality while meeting length requirements
        
        EDITING GUIDELINES:
        - Preserve the original meaning and key messages
        - Maintain markdown formatting and structure
        - Keep all important facts and statistics
        - Ensure content remains valuable and actionable
        - Focus on improvements that enhance reader experience
        
        Return the fully edited and improved content:
        """
    
    def _get_system_message(self) -> str:
        """Get system message defining the editor agent's role."""
        return """
        You are a professional content editor with extensive experience in improving
        written content across multiple formats and industries.
        
        Your expertise includes:
        - Advanced editing and proofreading techniques
        - Readability analysis and optimization
        - Content structure and flow enhancement
        - Audience-focused content refinement
        - Quality assurance and improvement
        - Style guide adherence and consistency
        
        Your editing philosophy:
        - Always preserve the core message and intent
        - Prioritize clarity and reader comprehension
        - Enhance engagement while maintaining professionalism
        - Improve structure and logical flow
        - Optimize for both readability and search engines
        - Ensure content serves the target audience effectively
        
        Focus on making substantial improvements that enhance content quality,
        readability, and overall effectiveness while respecting the original
        author's voice and intent.
        """
    
    def _interpret_readability_scores(self, analysis_result) -> str:
        """
        Interpret readability scores and provide context.
        
        Args:
            analysis_result: Content analysis results
            
        Returns:
            Human-readable interpretation of readability scores
        """
        readability = analysis_result.get('readability_score', 0)
        
        if readability >= 80:
            return "(Very Easy - 5th grade level)"
        elif readability >= 70:
            return "(Easy - 6th grade level)"
        elif readability >= 60:
            return "(Standard - 7th-8th grade level)"
        elif readability >= 50:
            return "(Fairly Difficult - 9th-10th grade level)"
        elif readability >= 30:
            return "(Difficult - College level)"
        else:
            return "(Very Difficult - Graduate level)"
    
    def _identify_improvement_areas(self, analysis_result, request) -> str:
        """
        Identify specific areas for content improvement.
        
        Args:
            analysis_result: Content analysis results
            request: Original content request
            
        Returns:
            Formatted improvement recommendations
        """
        improvements = []
        
        # Word count analysis
        current_words = analysis_result.get('word_count', 0)
        target_words = request.word_count
        word_diff = abs(current_words - target_words)
        
        if word_diff > target_words * 0.15:  # More than 15% difference
            if current_words < target_words:
                improvements.append(f"• EXPAND CONTENT: Add {target_words - current_words} words to reach target length")
            else:
                improvements.append(f"• REDUCE CONTENT: Remove {current_words - target_words} words to meet target length")
        
        # Readability analysis
        readability = analysis_result.get('readability_score', 0)
        if readability < 50:
            improvements.append("• IMPROVE READABILITY: Simplify complex sentences and reduce jargon")
        elif readability > 80 and "professional" in request.tone.lower():
            improvements.append("• BALANCE TONE: Add professional depth while maintaining accessibility")
        
        # Grade level analysis
        grade_level = analysis_result.get('grade_level', 0)
        if grade_level > 12:
            improvements.append("• REDUCE COMPLEXITY: Lower grade level for broader audience appeal")
        
        return "\n".join(improvements) if improvements else "• Content appears well-balanced, focus on general quality improvements"
    
    def _post_process_edited_content(self, edited_content: str, original_draft, request) -> str:
        """
        Post-process edited content for final quality assurance.
        
        Args:
            edited_content: Edited content from LLM
            original_draft: Original content draft
            request: Content creation request
            
        Returns:
            Final refined content
        """
        # Clean up the edited content
        refined = edited_content.strip()
        
        # Ensure title preservation
        if not refined.startswith('#'):
            refined = f"# {original_draft.title}\n\n{refined}"
        
        # Clean up excessive whitespace and formatting
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
    
    def _generate_improvement_suggestions(self, analysis_result, request) -> List[str]:
        """
        Generate specific suggestions for further content improvement.
        
        Args:
            analysis_result: Final content analysis results
            request: Original content request
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Readability suggestions
        readability = analysis_result.get('readability_score', 0)
        if readability < 60:
            suggestions.append("Consider simplifying complex sentences for better readability")
        
        # Word count suggestions
        word_count = analysis_result.get('word_count', 0)
        target_words = request.word_count
        
        if abs(word_count - target_words) > target_words * 0.1:
            if word_count < target_words:
                suggestions.append("Content could be expanded with additional examples or details")
            else:
                suggestions.append("Content could be condensed for better focus")
        
        # Grade level suggestions
        grade_level = analysis_result.get('grade_level', 0)
        if grade_level > 14:
            suggestions.append("Consider reducing technical complexity for broader accessibility")
        
        return suggestions
    
    def _calculate_readability_improvement(self, before_analysis, after_analysis) -> float:
        """
        Calculate readability improvement percentage.
        
        Args:
            before_analysis: Analysis before editing
            after_analysis: Analysis after editing
            
        Returns:
            Improvement percentage
        """
        before_score = before_analysis.get('readability_score', 0)
        after_score = after_analysis.get('readability_score', 0)
        
        if before_score == 0:
            return 0.0
        
        improvement = ((after_score - before_score) / before_score) * 100
        return round(improvement, 2)
    
    def _calculate_editing_quality_score(self, draft, analysis, request) -> int:
        """
        Calculate overall editing quality score.
        
        Args:
            draft: Edited content draft
            analysis: Content analysis results
            request: Original content request
            
        Returns:
            Quality score from 0-100
        """
        score = 0
        
        # Word count accuracy (30 points)
        word_accuracy = 1 - abs(draft.word_count - request.word_count) / request.word_count
        score += int(word_accuracy * 30)
        
        # Readability score (25 points)
        readability = analysis.readability_score
        if 50 <= readability <= 80:  # Optimal range
            score += 25
        elif 40 <= readability <= 90:  # Good range
            score += 20
        else:
            score += 10
        
        # Grade level appropriateness (20 points)
        grade_level = analysis.grade_level
        if 8 <= grade_level <= 12:  # Appropriate for general audience
            score += 20
        elif 6 <= grade_level <= 14:  # Acceptable range
            score += 15
        else:
            score += 5
        
        # Content structure (25 points)
        lines = draft.content.split('\n')
        headings = [line for line in lines if line.startswith('#')]
        if len(headings) >= 3:  # Good structure
            score += 25
        elif len(headings) >= 1:
            score += 15
        else:
            score += 5
        
        return min(100, score)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Provide detailed information about this agent's capabilities.
        
        Returns:
            Agent information dictionary
        """
        return {
            "name": "Editor Agent",
            "type": self.agent_type,
            "role": "Content Refinement and Quality Enhancement Specialist",
            "primary_function": "Content editing, quality improvement, and readability optimization",
            "domain_expertise": self.domain_expertise,
            "capabilities": [
                "Advanced content editing and proofreading",
                "Readability analysis and optimization",
                "Content structure enhancement",
                "Style and tone consistency enforcement",
                "Quality metrics assessment",
                "Grammar and language improvement"
            ],
            "tools": ["Content Analysis Tool", "Readability Assessment", "Quality Metrics"],
            "workflow_position": "Refinement - Fourth stage, improves content quality and readability"
        }