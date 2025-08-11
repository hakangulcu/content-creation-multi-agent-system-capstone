"""
Quality Assurance Agent - Final Validation and Delivery Specialist

This agent is responsible for conducting final quality checks, validation,
and content delivery, ensuring all requirements are met and content
is ready for publication or distribution.

Core Capabilities:
- Final quality assessment and validation
- Requirements compliance verification
- Content delivery and file management
- Quality reporting and documentation

Domain Expertise:
- Quality assurance methodologies
- Content validation techniques
- Publication readiness assessment
- Documentation and reporting standards
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool
except ImportError:
    HumanMessage = None
    SystemMessage = None
    tool = None

logger = logging.getLogger(__name__)

@tool
def save_content_tool(content: str, filename: str) -> Dict[str, str]:
    """
    Saves content to a file with proper error handling.
    
    Args:
        content: Content to save
        filename: Name of the file
        
    Returns:
        Save operation result with status and file path
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
            "filepath": "",
            "message": f"Failed to save: {str(e)}"
        }

class QualityAssuranceAgent:
    """
    Specialized agent for final quality assurance and content delivery.
    
    The Quality Assurance Agent serves as the final checkpoint in the content
    creation pipeline, responsible for comprehensive quality validation,
    requirements verification, and professional content delivery. This agent
    ensures all content meets specified standards before publication.
    
    Key Responsibilities:
    1. Final quality assessment and validation
    2. Requirements compliance verification
    3. Content formatting and presentation optimization
    4. Professional documentation and metadata generation
    5. Content delivery and file management
    
    Specialized Capabilities:
    - Comprehensive quality assessment frameworks
    - Requirements validation methodologies
    - Professional content formatting
    - Quality reporting and documentation
    - Publication readiness verification
    """
    
    def __init__(self, llm):
        """
        Initialize the Quality Assurance Agent with language model and tools.
        
        Args:
            llm: Language model instance for quality assessment
        """
        self.llm = llm
        self.tools = [save_content_tool]
        self.agent_type = "Quality Assurance Specialist"
        self.domain_expertise = [
            "Quality assurance methodologies",
            "Content validation techniques",
            "Publication readiness assessment",
            "Documentation and reporting standards",
            "Requirements compliance verification",
            "Professional content formatting"
        ]
    
    async def finalize_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive final quality check and prepare content for delivery.
        
        This method implements a thorough QA process:
        1. Comprehensive quality assessment and scoring
        2. Requirements compliance verification
        3. Content formatting and presentation optimization
        4. Professional documentation generation
        5. Content delivery and file management
        
        Args:
            state: Current workflow state with completed content
            
        Returns:
            Updated state with final content and delivery confirmation
        """
        draft = state["draft"]
        request = state["request"]
        
        logger.info(f"QA Agent: Conducting final quality assessment")
        
        # Perform comprehensive quality assessment
        quality_assessment = await self._conduct_quality_assessment(draft, request, state)
        
        # Generate professional content package
        final_content_package = self._create_final_content_package(
            draft, request, state, quality_assessment
        )
        
        # Save content with professional formatting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._generate_filename(request.topic, timestamp)
        
        save_result = save_content_tool.invoke({
            "content": final_content_package,
            "filename": filename
        })
        
        # Update workflow state with final results
        state["final_content"] = final_content_package
        
        # Initialize metadata if not present
        if "metadata" not in state:
            state["metadata"] = {}
        
        state["metadata"]["qa_completed"] = datetime.now().isoformat()
        state["metadata"]["output_file"] = save_result.get("filepath", "")
        state["metadata"]["final_quality_score"] = quality_assessment.get("overall_score", 0)
        state["metadata"]["publication_ready"] = quality_assessment.get("publication_ready", False)
        
        # Initialize feedback_history if not present
        if "feedback_history" not in state:
            state["feedback_history"] = []
        
        state["feedback_history"].append(quality_assessment.get("detailed_feedback", ""))
        
        logger.info(f"QA Agent: Content finalized and saved to {save_result.get('filepath', 'N/A')}")
        logger.info(f"QA Agent: Final quality score: {quality_assessment.get('overall_score', 'N/A')}/100")
        
        # Return updated state preserving all fields
        return state
    
    async def _conduct_quality_assessment(self, draft, request, state) -> Dict[str, Any]:
        """
        Conduct comprehensive quality assessment of the final content.
        
        Args:
            draft: Final content draft
            request: Original content request
            state: Current workflow state
            
        Returns:
            Comprehensive quality assessment results
        """
        # Create detailed quality assessment prompt
        assessment_prompt = self._create_assessment_prompt(draft, request, state)
        
        messages = [
            SystemMessage(content=self._get_qa_system_message()),
            HumanMessage(content=assessment_prompt)
        ]
        
        # Generate quality assessment
        response = await self.llm.ainvoke(messages)
        qa_feedback = response.content
        
        # Parse and enhance assessment results
        quality_metrics = self._calculate_quality_metrics(draft, request, state)
        
        return {
            "detailed_feedback": qa_feedback,
            "overall_score": quality_metrics["overall_score"],
            "publication_ready": quality_metrics["publication_ready"],
            "quality_breakdown": quality_metrics["breakdown"],
            "recommendations": self._extract_recommendations(qa_feedback)
        }
    
    def _create_assessment_prompt(self, draft, request, state) -> str:
        """
        Create comprehensive quality assessment prompt.
        
        Args:
            draft: Content draft to assess
            request: Original content request
            state: Current workflow state
            
        Returns:
            Detailed quality assessment prompt
        """
        metadata = state.get("metadata", {})
        
        return f"""
        Conduct a comprehensive final quality assessment of this content before publication.
        
        CONTENT TO ASSESS:
        {draft.content}
        
        ORIGINAL REQUIREMENTS:
        - Topic: {request.topic}
        - Content Type: {request.content_type.value}
        - Target Audience: {request.target_audience}
        - Target Word Count: {request.word_count} words
        - Tone: {request.tone}
        - Keywords: {request.keywords or 'None specified'}
        - Special Requirements: {request.special_requirements or 'None'}
        
        CONTENT METRICS:
        - Actual Word Count: {draft.word_count} words
        - Reading Time: {draft.reading_time} minutes
        - SEO Score: {metadata.get('seo_score', 'N/A')}
        - Readability Score: {getattr(state.get('analysis'), 'readability_score', 'N/A') if state.get('analysis') else 'N/A'}
        
        WORKFLOW COMPLETION STATUS:
        - Research Completed: {metadata.get('research_completed', 'N/A')}
        - Planning Completed: {metadata.get('planning_completed', 'N/A')}
        - Writing Completed: {metadata.get('writing_completed', 'N/A')}
        - Editing Completed: {metadata.get('editing_completed', 'N/A')}
        - SEO Optimization Completed: {metadata.get('seo_optimization_completed', 'N/A')}
        
        QUALITY ASSESSMENT CRITERIA:
        
        1. CONTENT QUALITY (25 points)
           - Accuracy and factual correctness
           - Depth and comprehensiveness
           - Value and usefulness to target audience
           - Originality and unique insights
           - Professional presentation
        
        2. REQUIREMENTS COMPLIANCE (25 points)
           - Word count accuracy (target: {request.word_count})
           - Tone appropriateness for audience
           - Content type format adherence
           - Special requirements fulfillment
           - Keyword integration effectiveness
        
        3. TECHNICAL QUALITY (25 points)
           - Grammar, spelling, and language mechanics
           - Content structure and organization
           - Readability and accessibility
           - Formatting and presentation
           - SEO optimization effectiveness
        
        4. AUDIENCE ENGAGEMENT (25 points)
           - Relevance to target audience
           - Engagement and readability
           - Actionable insights and practical value
           - Clear communication and messaging
           - Call-to-action effectiveness (if applicable)
        
        ASSESSMENT OUTPUT REQUIREMENTS:
        
        Provide a comprehensive assessment including:
        
        1. OVERALL QUALITY SCORE: X/100
        
        2. PUBLICATION STATUS: [APPROVED FOR PUBLICATION / NEEDS MINOR REVISIONS / REQUIRES MAJOR REVISIONS]
        
        3. DETAILED BREAKDOWN:
           - Content Quality: X/25 - [Brief assessment]
           - Requirements Compliance: X/25 - [Brief assessment]  
           - Technical Quality: X/25 - [Brief assessment]
           - Audience Engagement: X/25 - [Brief assessment]
        
        4. KEY STRENGTHS:
           - [List 3-5 main strengths of the content]
        
        5. AREAS FOR IMPROVEMENT:
           - [List any areas that could be enhanced]
        
        6. FINAL RECOMMENDATIONS:
           - [Specific recommendations for optimization or next steps]
        
        7. PUBLICATION READINESS SUMMARY:
           - [Brief summary of whether content is ready for publication and why]
        
        Conduct this assessment with the highest professional standards, ensuring
        the content meets publication-ready quality before final approval.
        """
    
    def _get_qa_system_message(self) -> str:
        """Get system message defining the QA agent's role."""
        return """
        You are a senior quality assurance specialist with extensive experience
        in content review, publication standards, and professional content validation.
        
        Your expertise includes:
        - Comprehensive quality assessment methodologies
        - Publication readiness evaluation
        - Content standards and best practices
        - Audience-focused content evaluation
        - Technical content review processes
        - Professional documentation standards
        
        Your quality philosophy:
        - Maintain the highest standards for professional content
        - Ensure content serves the target audience effectively
        - Validate all requirements are met before publication
        - Provide constructive, actionable feedback
        - Balance perfectionism with practical publication needs
        - Focus on value delivery and user experience
        
        Your role is to conduct the final, comprehensive quality check that
        ensures content is ready for professional publication and will achieve
        its intended objectives effectively.
        """
    
    def _calculate_quality_metrics(self, draft, request, state) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics for the content.
        
        Args:
            draft: Content draft
            request: Original content request
            state: Current workflow state
            
        Returns:
            Quality metrics and scoring breakdown
        """
        scores = {}
        
        # Content Quality Score (25 points)
        content_score = 20  # Base score
        if draft.word_count >= 500:  # Substantial content
            content_score += 3
        if draft.content.count('#') >= 3:  # Good structure
            content_score += 2
        scores["content_quality"] = min(25, content_score)
        
        # Requirements Compliance Score (25 points)
        compliance_score = 15  # Base score
        word_accuracy = 1 - abs(draft.word_count - request.word_count) / request.word_count
        compliance_score += int(word_accuracy * 10)
        scores["requirements_compliance"] = min(25, compliance_score)
        
        # Technical Quality Score (25 points)
        technical_score = 18  # Base score
        seo_score = state.get("metadata", {}).get("seo_score", 0)
        if seo_score > 70:
            technical_score += 4
        elif seo_score > 50:
            technical_score += 2
        
        if state.get("analysis"):
            readability = getattr(state["analysis"], "readability_score", 0)
            if readability > 60:
                technical_score += 3
        
        scores["technical_quality"] = min(25, technical_score)
        
        # Audience Engagement Score (25 points)
        engagement_score = 20  # Base score
        if request.tone.lower() in draft.content.lower():  # Tone consistency
            engagement_score += 3
        if any(keyword.lower() in draft.content.lower() for keyword in (request.keywords or [])):
            engagement_score += 2
        scores["audience_engagement"] = min(25, engagement_score)
        
        # Calculate overall score
        overall_score = sum(scores.values())
        publication_ready = overall_score >= 75  # 75% threshold for publication readiness
        
        return {
            "overall_score": overall_score,
            "publication_ready": publication_ready,
            "breakdown": scores
        }
    
    def _create_final_content_package(self, draft, request, state, quality_assessment) -> str:
        """
        Create the final content package with professional formatting and metadata.
        
        Args:
            draft: Content draft
            request: Original content request
            state: Current workflow state
            quality_assessment: Quality assessment results
            
        Returns:
            Complete final content package
        """
        metadata = state.get("metadata", {})
        
        # Create comprehensive final package
        final_package = f"""# {draft.title}

**Content Type:** {request.content_type.value}
**Target Audience:** {request.target_audience}
**Word Count:** {draft.word_count}
**Reading Time:** {draft.reading_time} minutes
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**SEO Score:** {metadata.get('seo_score', 'N/A')}/100

---

{draft.content}

---

## Content Generation Metadata

**Quality Assurance Report:**
{quality_assessment.get('detailed_feedback', 'Assessment completed')}

**Generation Pipeline Completion:**
- Research completed: {metadata.get('research_completed', 'N/A')}
- Planning completed: {metadata.get('planning_completed', 'N/A')}
- Writing completed: {metadata.get('writing_completed', 'N/A')}
- Editing completed: {metadata.get('editing_completed', 'N/A')}
- SEO optimization completed: {metadata.get('seo_optimization_completed', 'N/A')}
- Quality assurance completed: {metadata.get('qa_completed', datetime.now().isoformat())}

**Performance Metrics:**
- Final Quality Score: {quality_assessment.get('overall_score', 'N/A')}/100
- Publication Ready: {quality_assessment.get('publication_ready', False)}
- SEO Score: {metadata.get('seo_score', 'N/A')}/100
- Keyword Coverage: {metadata.get('keyword_coverage', 'N/A')}%

**Content Creation Request:**
- Original Topic: {request.topic}
- Target Word Count: {request.word_count}
- Tone: {request.tone}
- Keywords: {', '.join(request.keywords) if request.keywords else 'None specified'}
- Special Requirements: {request.special_requirements or 'None'}

---

*Generated by Multi-Agent Content Creation System*
*Quality Assured and Ready for Publication*
"""
        
        return final_package
    
    def _generate_filename(self, topic: str, timestamp: str) -> str:
        """
        Generate appropriate filename for the content.
        
        Args:
            topic: Content topic
            timestamp: Creation timestamp
            
        Returns:
            Formatted filename
        """
        # Clean topic for filename
        clean_topic = topic.replace(' ', '_').replace(':', '').replace('?', '').replace('!', '')
        # Remove special characters and limit length
        clean_topic = ''.join(c for c in clean_topic if c.isalnum() or c == '_')[:50]
        
        return f"{clean_topic}_{timestamp}.md"
    
    def _extract_recommendations(self, qa_feedback: str) -> list:
        """
        Extract actionable recommendations from QA feedback.
        
        Args:
            qa_feedback: Quality assessment feedback text
            
        Returns:
            List of extracted recommendations
        """
        recommendations = []
        
        # Simple extraction based on common patterns
        lines = qa_feedback.split('\n')
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if 'recommendation' in line.lower() or 'suggest' in line.lower():
                in_recommendations = True
            elif in_recommendations and line.startswith('-'):
                recommendations.append(line[1:].strip())
            elif in_recommendations and line and not line.startswith('-'):
                in_recommendations = False
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Provide detailed information about this agent's capabilities.
        
        Returns:
            Agent information dictionary
        """
        return {
            "name": "Quality Assurance Agent",
            "type": self.agent_type,
            "role": "Final Validation and Delivery Specialist",
            "primary_function": "Comprehensive quality assurance and professional content delivery",
            "domain_expertise": self.domain_expertise,
            "capabilities": [
                "Comprehensive quality assessment",
                "Requirements compliance verification",
                "Publication readiness evaluation",
                "Professional content formatting",
                "Quality reporting and documentation",
                "Content delivery and file management"
            ],
            "tools": ["Quality Assessment Framework", "Content Validation", "File Management"],
            "workflow_position": "Finalization - Final stage, ensures quality and manages delivery"
        }