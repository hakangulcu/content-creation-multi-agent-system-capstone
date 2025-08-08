#!/usr/bin/env python3
"""
Content Creation Multi-Agent System Demo
AAIDC Module 2 Project

This script demonstrates various use cases of the multi-agent content creation system.
"""

import os
import asyncio
import time
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our system
from main import ContentCreationWorkflow, ContentRequest, ContentType
from langchain_core.messages import HumanMessage

# Demo configurations
DEMO_REQUESTS = [
    {
        "name": "AI Healthcare Blog Post",
        "request": ContentRequest(
            topic="Artificial Intelligence in Healthcare: Transforming Patient Care",
            content_type=ContentType.BLOG_POST,
            target_audience="Healthcare professionals and technology leaders",
            word_count=1500,
            tone="professional yet accessible",
            keywords=["AI in healthcare", "medical AI", "healthcare technology", "patient care", "diagnostic AI"],
            special_requirements="Include recent statistics, real-world examples, and future trends"
        )
    },
    {
        "name": "Social Media Campaign",
        "request": ContentRequest(
            topic="Sustainable Technology Solutions for Small Businesses",
            content_type=ContentType.SOCIAL_MEDIA,
            target_audience="Small business owners and entrepreneurs",
            word_count=300,
            tone="engaging and inspiring",
            keywords=["sustainable tech", "green business", "eco-friendly", "small business"],
            special_requirements="Focus on actionable tips and ROI benefits"
        )
    },
    {
        "name": "Technical Article",
        "request": ContentRequest(
            topic="Implementing Microservices Architecture: Best Practices and Pitfalls",
            content_type=ContentType.ARTICLE,
            target_audience="Software developers and system architects",
            word_count=2000,
            tone="technical but clear",
            keywords=["microservices", "software architecture", "distributed systems", "DevOps"],
            special_requirements="Include code examples and architectural diagrams descriptions"
        )
    },
    {
        "name": "Marketing Newsletter",
        "request": ContentRequest(
            topic="2025 Digital Marketing Trends Every Business Should Know",
            content_type=ContentType.NEWSLETTER,
            target_audience="Marketing professionals and business owners",
            word_count=800,
            tone="informative and actionable",
            keywords=["digital marketing", "2025 trends", "marketing strategy", "customer engagement"],
            special_requirements="Include statistics and actionable insights for each trend"
        )
    }
]

async def run_single_demo(workflow: ContentCreationWorkflow, demo_config: dict) -> dict:
    """Run a single demo and return results"""
    
    print(f"\nStarting Demo: {demo_config['name']}")
    print(f"Topic: {demo_config['request'].topic}")
    print(f"Type: {demo_config['request'].content_type.value}")
    print(f"Target: {demo_config['request'].word_count} words")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Execute the workflow
        result = await workflow.create_content(demo_config['request'])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Collect results
        demo_result = {
            "name": demo_config['name'],
            "success": True,
            "duration": duration,
            "word_count": result['draft'].word_count if result['draft'] else 0,
            "reading_time": result['draft'].reading_time if result['draft'] else 0,
            "seo_score": result['metadata'].get('seo_score', 'N/A'),
            "output_file": result['metadata'].get('output_file', 'N/A'),
            "error": None
        }
        
        print(f"Demo Completed Successfully!")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Word count: {demo_result['word_count']}")
        print(f"Reading time: {demo_result['reading_time']} minutes")
        print(f"SEO Score: {demo_result['seo_score']}")
        print(f"Saved to: {demo_result['output_file']}")
        
        return demo_result
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        demo_result = {
            "name": demo_config['name'],
            "success": False,
            "duration": duration,
            "error": str(e)
        }
        
        print(f"Demo Failed: {e}")
        print(f"Common solutions:")
        print(f"   • Check if Ollama is running: ollama serve")
        print(f"   • Verify model is installed: ollama pull {os.getenv('OLLAMA_MODEL', 'llama3.1:8b')}")
        print(f"   • Check system resources (RAM/CPU usage)")
        return demo_result

async def main():
    """Main demo function"""
    
    print("Content Creation Multi-Agent System Demo")
    print("AAIDC Module 2 Project - Local Ollama Version")
    print("=" * 70)
    
    # Check Ollama configuration
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"Using Ollama model: {model_name}")
    print(f"Ollama server: {base_url}")
    
    # Initialize workflow
    print("Initializing Content Creation Workflow...")
    try:
        workflow = ContentCreationWorkflow(model_name=model_name, base_url=base_url)
        print("Workflow initialized successfully")
        
        # Test Ollama connection
        print("Testing Ollama connection...")
        test_messages = [HumanMessage(content="Hello, respond with just 'OK' if you can hear me.")]
        response = await workflow.llm.ainvoke(test_messages)
        print(f"Ollama connection test successful: {response.content[:50]}...")
        
    except Exception as e:
        print(f"Failed to initialize workflow: {e}")
        print("\nTroubleshooting Steps:")
        print("1. Make sure Ollama is running:")
        print("   ollama serve")
        print(f"2. Install the required model:")
        print(f"   ollama pull {model_name}")
        print("3. Test the model:")
        print(f"   ollama run {model_name} \"Hello\"")
        print("4. Check system resources (RAM/CPU)")
        return
    
    # Demo menu
    while True:
        print("\nDemo Options:")
        print("1. Run All Predefined Demos")
        print("2. Interactive Custom Demo") 
        print("3. Performance Benchmark")
        print("4. Show Demo Descriptions")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            # Run all predefined demos
            print(f"\nRunning {len(DEMO_REQUESTS)} Predefined Demos")
            results = []
            
            for demo_config in DEMO_REQUESTS:
                result = await run_single_demo(workflow, demo_config)
                results.append(result)
                
                # Brief pause between demos
                if demo_config != DEMO_REQUESTS[-1]:
                    print("\nPausing 5 seconds before next demo...")
                    await asyncio.sleep(5)
            
            # Summary
            print("\nDemo Summary:")
            print("-" * 40)
            successful = sum(1 for r in results if r['success'])
            total_time = sum(r['duration'] for r in results)
            
            print(f"Successful demos: {successful}/{len(results)}")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Average time: {total_time/len(results):.1f} seconds")
            
            for result in results:
                status = "[OK]" if result['success'] else "[FAIL]"
                print(f"{status} {result['name']}: {result['duration']:.1f}s")
        
        elif choice == "2":
            print("\nInteractive Demo - Create Custom Content")
            topic = input("Enter topic: ").strip() or "Benefits of AI in Education"
            
            request = ContentRequest(
                topic=topic,
                content_type=ContentType.BLOG_POST,
                target_audience="General audience",
                word_count=1000,
                tone="professional",
                keywords=[topic],
                special_requirements="User-generated content"
            )
            
            demo_config = {"name": "Custom Interactive Demo", "request": request}
            await run_single_demo(workflow, demo_config)
        
        elif choice == "3":
            print("\nPerformance Benchmark")
            # Simple benchmark
            request = ContentRequest(
                topic="Future of Remote Work",
                content_type=ContentType.BLOG_POST,
                target_audience="Business professionals",
                word_count=800,
                tone="professional",
                keywords=["remote work"],
                special_requirements="Performance test"
            )
            
            start = time.time()
            result = await workflow.create_content(request)
            duration = time.time() - start
            
            print(f"Benchmark completed in {duration:.1f} seconds")
            print(f"Words generated: {result['draft'].word_count}")
            print(f"Words per second: {result['draft'].word_count/duration:.1f}")
        
        elif choice == "4":
            print("\nDemo Descriptions:")
            for i, demo in enumerate(DEMO_REQUESTS, 1):
                print(f"{i}. {demo['name']}")
                print(f"   Topic: {demo['request'].topic}")
                print(f"   Type: {demo['request'].content_type.value}")
                print(f"   Words: {demo['request'].word_count}")
                print()
        
        elif choice == "5":
            print("\nDemo completed! Thanks for trying the system!")
            break
        
        else:
            print("Invalid option. Please select 1-5.")

if __name__ == "__main__":
    asyncio.run(main())
