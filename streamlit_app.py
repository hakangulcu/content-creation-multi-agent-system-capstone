"""
Streamlit Web Interface for Content Creation Multi-Agent System

This provides a user-friendly web interface for the content creation system,
abstracting away technical complexity and providing real-time feedback.
"""

import streamlit as st
import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import os
import sys

# Add the current directory to Python path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ContentCreationWorkflow
from types_shared import ContentRequest, ContentType
from security_utils import ValidationError
from security_utils import validate_content_request
from resilience_utils import get_system_health, get_performance_stats


# Streamlit page configuration
st.set_page_config(
    page_title="Content Creation AI System",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize Streamlit session state."""
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'is_generating' not in st.session_state:
        st.session_state.is_generating = False


@st.cache_resource
def initialize_workflow():
    """Initialize and cache the workflow instance."""
    try:
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        workflow = ContentCreationWorkflow(
            model_name=model_name, 
            base_url=base_url
        )
        return workflow, None
    except Exception as e:
        return None, str(e)


def display_header():
    """Display the application header."""
    st.title("Content Creation AI System")
    st.markdown("""
    **Powered by Multi-Agent AI Pipeline**
    
    Transform your ideas into high-quality, SEO-optimized content using our sophisticated 
    6-agent system running entirely offline with local Ollama models.
    """)
    
    # System status indicator
    col1, col2, col3 = st.columns(3)
    
    with col1:
        workflow, error = initialize_workflow()
        if workflow:
            st.success("System Online")
        else:
            st.error(f"System Offline: {error}")
    
    with col2:
        health = get_system_health()
        breakers = health.get("circuit_breakers", {})
        open_breakers = sum(1 for cb in breakers.values() if cb["state"] == "open")
        if open_breakers == 0:
            st.info("All Services Healthy")
        else:
            st.warning(f"{open_breakers} Service(s) Degraded")
    
    with col3:
        if st.session_state.generation_history:
            success_rate = sum(1 for h in st.session_state.generation_history if h.get("success", False))
            success_rate = (success_rate / len(st.session_state.generation_history)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "N/A")


def display_sidebar():
    """Display the sidebar with configuration and system info."""
    st.sidebar.header("Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    model_name = st.sidebar.text_input(
        "Ollama Model", 
        value=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        help="The Ollama model to use for content generation"
    )
    
    base_url = st.sidebar.text_input(
        "Ollama Base URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="URL of your Ollama server"
    )
    
    # Advanced options
    st.sidebar.subheader("Advanced Options")
    workflow_timeout = st.sidebar.slider(
        "Workflow Timeout (seconds)",
        min_value=60,
        max_value=600,
        value=300,
        help="Maximum time to wait for content generation"
    )
    
    enable_content_filtering = st.sidebar.checkbox(
        "Enable Content Filtering",
        value=True,
        help="Apply content filtering and moderation"
    )
    
    # System monitoring
    st.sidebar.subheader("System Monitor")
    
    if st.sidebar.button("Refresh System Health"):
        st.rerun()
    
    # Performance stats
    stats = get_performance_stats("main.ContentCreationWorkflow.create_content", 60)
    if stats and "error" not in stats:
        st.sidebar.metric("Operations (1h)", stats["total_operations"])
        st.sidebar.metric("Success Rate", f"{stats['success_rate']:.1%}")
        if stats["average_duration"] > 0:
            st.sidebar.metric("Avg Duration", f"{stats['average_duration']:.1f}s")
    
    # Generation history
    if st.session_state.generation_history:
        st.sidebar.subheader("Recent Generations")
        for i, entry in enumerate(reversed(st.session_state.generation_history[-5:])):
            with st.sidebar.expander(f"{entry['topic'][:20]}..." if len(entry['topic']) > 20 else entry['topic']):
                st.write(f"**Type:** {entry['content_type']}")
                st.write(f"**Status:** {'Success' if entry.get('success', False) else 'Failed'}")
                st.write(f"**Time:** {entry['timestamp']}")
                if entry.get('word_count'):
                    st.write(f"**Words:** {entry['word_count']}")


def display_content_form():
    """Display the main content creation form."""
    st.header("Create Content")
    
    with st.form("content_form"):
        # Basic content parameters
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input(
                "Content Topic *",
                placeholder="e.g., 'The Future of Artificial Intelligence in Healthcare'",
                help="What do you want to write about?"
            )
            
            content_type = st.selectbox(
                "Content Type *",
                options=[ct.value for ct in ContentType],
                format_func=lambda x: x.replace('_', ' ').title(),
                help="Choose the type of content to generate"
            )
            
            target_audience = st.text_input(
                "Target Audience *",
                placeholder="e.g., 'Healthcare professionals and technology enthusiasts'",
                help="Who is this content for?"
            )
        
        with col2:
            word_count = st.number_input(
                "Word Count *",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Approximate number of words to generate"
            )
            
            tone = st.selectbox(
                "Tone",
                options=["professional", "casual", "academic", "conversational", "persuasive", "informative"],
                help="The tone and style of the content"
            )
            
            keywords = st.text_area(
                "Keywords (optional)",
                placeholder="Enter keywords separated by commas\ne.g., AI, healthcare, technology, innovation",
                help="SEO keywords to include in the content"
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            special_requirements = st.text_area(
                "Special Requirements",
                placeholder="Any specific requirements, examples to include, or constraints...",
                help="Additional instructions for content generation"
            )
        
        submitted = st.form_submit_button("Generate Content", use_container_width=True)
        
        if submitted:
            # Validate required fields
            if not topic or not target_audience:
                st.error("Please fill in all required fields marked with *")
                return
            
            # Parse keywords
            keyword_list = []
            if keywords:
                keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            
            # Create content request
            try:
                content_request = ContentRequest(
                    topic=topic,
                    content_type=ContentType(content_type),
                    target_audience=target_audience,
                    word_count=word_count,
                    tone=tone,
                    keywords=keyword_list,
                    special_requirements=special_requirements
                )
                
                # Start content generation
                st.session_state.is_generating = True
                generate_content(content_request)
                
            except Exception as e:
                st.error(f"Error creating request: {str(e)}")


def generate_content(content_request: ContentRequest):
    """Generate content using the multi-agent system."""
    workflow, error = initialize_workflow()
    
    if not workflow:
        st.error(f"Cannot generate content: {error}")
        st.session_state.is_generating = False
        return
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Progress callback function
    def progress_callback(progress: float, stage_name: str, current_step: int, total_steps: int):
        progress_bar.progress(progress)
        status_text.text(f"{stage_name}... ({current_step}/{total_steps})")
    
    try:
        # Start generation
        start_time = time.time()
        
        # Execute the workflow with progress tracking
        result = asyncio.run(workflow.create_content(content_request, progress_callback))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Store result
        st.session_state.current_result = result
        
        # Update history
        history_entry = {
            "topic": content_request.topic,
            "content_type": content_request.content_type.value,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": True,
            "duration": duration,
            "word_count": result["draft"].word_count if result.get("draft") else 0
        }
        st.session_state.generation_history.append(history_entry)
        
        # Clear progress
        progress_bar.progress(1.0)
        status_text.text("Content generation completed!")
        
        # Success message
        st.success(f"Content generated successfully in {duration:.1f} seconds!")
        
    except ValidationError as e:
        st.error(f"Validation Error: {str(e)}")
        st.info("Please review your input and try again.")
        
    except asyncio.TimeoutError:
        st.error("Content generation timed out. Please try again with a shorter word count or simpler topic.")
        
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        
        # Log failure
        history_entry = {
            "topic": content_request.topic,
            "content_type": content_request.content_type.value,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": False,
            "error": str(e)
        }
        st.session_state.generation_history.append(history_entry)
    
    finally:
        st.session_state.is_generating = False


def display_results():
    """Display the generated content and analysis."""
    if not st.session_state.current_result:
        return
    
    result = st.session_state.current_result
    
    st.header("Generated Content")
    
    # Content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Final Content", "Analytics", "Research Data", "Metadata"])
    
    with tab1:
        if result.get("final_content"):
            # Content preview
            st.subheader("Content Preview")
            st.markdown(result["final_content"])
            
            # Download button
            if st.download_button(
                label="Download Content",
                data=result["final_content"],
                file_name=f"{result['request'].topic.replace(' ', '_')}.md",
                mime="text/markdown"
            ):
                st.success("Content downloaded!")
            
            # Content metrics
            col1, col2, col3, col4 = st.columns(4)
            
            if result.get("draft"):
                draft = result["draft"]
                col1.metric("Word Count", draft.word_count)
                col2.metric("Reading Time", f"{draft.reading_time} min")
            
            if result.get("analysis"):
                analysis = result["analysis"]
                col3.metric("Readability Score", f"{analysis.readability_score:.1f}")
                col4.metric("Grade Level", f"{analysis.grade_level:.1f}")
            
            # SEO Score
            if result.get("metadata", {}).get("seo_score"):
                st.metric("SEO Score", f"{result['metadata']['seo_score']}/100")
    
    with tab2:
        st.subheader("Content Analytics")
        
        if result.get("analysis"):
            analysis = result["analysis"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Readability Score", f"{analysis.readability_score:.1f}")
                st.caption("Higher is better (60-70 is good)")
                
                st.metric("Grade Level", f"{analysis.grade_level:.1f}")
                st.caption("Target grade level for reading")
            
            with col2:
                if analysis.keyword_density:
                    st.subheader("Top Keywords")
                    for keyword, density in list(analysis.keyword_density.items())[:5]:
                        st.write(f"**{keyword}**: {density}%")
        
        # Suggestions
        if result.get("analysis") and result["analysis"].suggestions:
            st.subheader("Improvement Suggestions")
            for suggestion in result["analysis"].suggestions:
                st.write(f"• {suggestion}")
    
    with tab3:
        st.subheader("Research Data")
        
        if result.get("research_data"):
            research = result["research_data"]
            
            if research.key_facts:
                st.subheader("Key Facts")
                for fact in research.key_facts:
                    st.write(f"• {fact}")
            
            if research.statistics:
                st.subheader("Statistics")
                for stat in research.statistics:
                    st.write(f"Statistic: {stat}")
            
            if research.quotes:
                st.subheader("Relevant Quotes")
                for quote in research.quotes:
                    st.quote(quote)
    
    with tab4:
        st.subheader("Generation Metadata")
        
        if result.get("metadata"):
            metadata = result["metadata"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if "workflow_start_time" in metadata:
                    st.write(f"**Start Time:** {metadata['workflow_start_time']}")
                if "workflow_end_time" in metadata:
                    st.write(f"**End Time:** {metadata['workflow_end_time']}")
                if "total_duration" in metadata:
                    st.write(f"**Duration:** {metadata['total_duration']:.1f}s")
            
            with col2:
                st.write(f"**Revisions:** {result.get('revision_count', 0)}")
                if metadata.get("seo_score"):
                    st.write(f"**SEO Score:** {metadata['seo_score']}")
                if metadata.get("output_file"):
                    st.write(f"**Saved to:** {metadata['output_file']}")
            
            # Security info
            if "security_validation" in metadata:
                security = metadata["security_validation"]
                st.subheader("Security Validation")
                st.write(f"**Status:** {'Passed' if security['validated'] else 'Failed'}")
                st.write(f"**Threat Level:** {security['threat_level']}")
                if security["warnings"]:
                    st.write("**Warnings:**")
                    for warning in security["warnings"]:
                        st.write(f"Warning: {warning}")


def display_examples():
    """Display example prompts and use cases."""
    st.header("Examples & Use Cases")
    
    examples = [
        {
            "title": "Blog Post - Technology",
            "topic": "The Impact of AI on Remote Work Productivity",
            "type": "blog_post",
            "audience": "Remote workers and team managers",
            "words": 1200,
            "keywords": "AI, remote work, productivity, automation, team collaboration"
        },
        {
            "title": "Article - Healthcare", 
            "topic": "Telemedicine: Transforming Patient Care in Rural Areas",
            "type": "article",
            "audience": "Healthcare professionals and policy makers",
            "words": 1800,
            "keywords": "telemedicine, rural healthcare, patient access, digital health"
        },
        {
            "title": "Newsletter - Marketing",
            "topic": "5 Email Marketing Trends for 2024",
            "type": "newsletter", 
            "audience": "Digital marketers and business owners",
            "words": 800,
            "keywords": "email marketing, trends, personalization, automation"
        },
        {
            "title": "Social Media - Fitness",
            "topic": "Quick Morning Workout Routine for Busy Professionals",
            "type": "social_media",
            "audience": "Busy professionals interested in fitness",
            "words": 200,
            "keywords": "morning workout, quick exercise, busy professionals, fitness"
        }
    ]
    
    cols = st.columns(2)
    
    for i, example in enumerate(examples):
        with cols[i % 2]:
            with st.expander(f"{example['title']}:"):
                st.write(f"**Topic:** {example['topic']}")
                st.write(f"**Type:** {example['type'].replace('_', ' ').title()}")
                st.write(f"**Audience:** {example['audience']}")
                st.write(f"**Length:** {example['words']} words")
                st.write(f"**Keywords:** {example['keywords']}")
                
                if st.button(f"Use This Example", key=f"example_{i}"):
                    # This would populate the form - in a real app, you'd set session state
                    st.info("Copy the example details to the form above!")


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Check if currently generating
    if st.session_state.is_generating:
        st.info("Content generation in progress...")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Content creation form
        display_content_form()
        
        # Results display
        if st.session_state.current_result:
            display_results()
    
    with col2:
        # Examples and tips
        display_examples()
        
        # Tips section
        st.subheader("Tips for Better Results")
        st.markdown("""
        **Topic Ideas:**
        - Be specific and focused
        - Include your target keywords
        - Consider your audience's interests
        
        **Audience Targeting:**
        - Define demographics clearly
        - Consider expertise level
        - Think about their goals
        
        **SEO Optimization:**
        - Use relevant keywords naturally
        - Include long-tail keywords
        - Consider search intent
        """)


if __name__ == "__main__":
    main()