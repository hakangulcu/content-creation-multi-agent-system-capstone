#!/usr/bin/env python3
"""
Simple test script to verify agent functionality
"""

import asyncio
import os
from main import ContentCreationWorkflow, ContentRequest, ContentType

async def test_agents():
    """Test the multi-agent system with a simple request"""
    
    print("Testing Multi-Agent Content Creation System")
    print("=" * 50)
    
    # Initialize workflow
    try:
        workflow = ContentCreationWorkflow(
            model_name="llama3.1:8b",
            base_url="http://localhost:11434"
        )
        print("Workflow initialized successfully")
    except Exception as e:
        print(f"Failed to initialize workflow: {e}")
        return
    
    # Create a simple test request
    request = ContentRequest(
        topic="Benefits of Exercise",
        content_type=ContentType.ARTICLE,
        target_audience="General public",
        word_count=500,  # Shorter for testing
        tone="friendly and informative",
        keywords=["exercise", "health", "fitness"],
        special_requirements="Keep it simple and engaging"
    )
    
    print(f"Testing with topic: {request.topic}")
    print(f"Target length: {request.word_count} words")
    print(f"Audience: {request.target_audience}")
    print("-" * 50)
    
    try:
        # Test individual agent info
        print("\nAgent Information:")
        print(f"1. Research Agent: {workflow.research_agent.__class__.__name__}")
        print(f"2. Planning Agent: {workflow.planning_agent.__class__.__name__}")
        print(f"3. Writer Agent: {workflow.writer_agent.__class__.__name__}")
        print(f"4. Editor Agent: {workflow.editor_agent.__class__.__name__}")
        print(f"5. SEO Agent: {workflow.seo_agent.__class__.__name__}")
        print(f"6. QA Agent: {workflow.qa_agent.__class__.__name__}")
        
        # Test agent info methods
        if hasattr(workflow.research_agent, 'get_agent_info'):
            agent_info = workflow.research_agent.get_agent_info()
            print(f"\nResearch Agent Details:")
            print(f"   Role: {agent_info.get('role', 'N/A')}")
            print(f"   Capabilities: {len(agent_info.get('capabilities', []))} capabilities")
        
        print("\nStarting content creation workflow...")
        
        # Execute workflow
        result = await workflow.create_content(request)
        
        print("\nContent Creation Completed!")
        print(f"Final word count: {result['draft'].word_count}")
        print(f"Reading time: {result['draft'].reading_time} minutes")
        print(f"📁 Saved to: {result['metadata'].get('output_file', 'N/A')}")
        print(f"SEO Score: {result['metadata'].get('seo_score', 'N/A')}")
        print(f"Quality Score: {result['metadata'].get('final_quality_score', 'N/A')}")
        
        # Show content preview
        if result.get("final_content"):
            print("\n📖 Content Preview (first 300 chars):")
            print("-" * 40)
            preview = result["final_content"][:300] + "..." if len(result["final_content"]) > 300 else result["final_content"]
            print(preview)
            
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during content creation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agents())