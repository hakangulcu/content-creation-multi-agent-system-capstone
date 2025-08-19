#!/usr/bin/env python3
"""
Content Creation Multi-Agent System Demo
AAIDC Module 3 Project

This script demonstrates various use cases of the multi-agent content creation system.
"""

import os
import asyncio
import time
import logging
import sys
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/demo.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

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

async def run_single_demo(workflow: ContentCreationWorkflow, demo_config: dict) -> Dict[str, Any]:
    """Run a single demo and return results with comprehensive error handling"""
    
    demo_name = demo_config.get('name', 'Unknown Demo')
    request = demo_config.get('request')
    
    if not request:
        error_msg = f"Invalid demo configuration: missing request for {demo_name}"
        logger.error(error_msg)
        return {
            "name": demo_name,
            "success": False,
            "duration": 0,
            "error": error_msg
        }
    
    logger.info(f"Starting demo: {demo_name}")
    print(f"\nStarting Demo: {demo_name}")
    print(f"Topic: {request.topic}")
    print(f"Type: {request.content_type.value}")
    print(f"Target: {request.word_count} words")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Validate workflow object
        if not workflow or not hasattr(workflow, 'create_content'):
            raise ValueError("Invalid workflow object provided")
        
        # Execute the workflow with timeout protection
        try:
            result = await asyncio.wait_for(
                workflow.create_content(request),
                timeout=600.0  # 10 minute timeout
            )
        except asyncio.TimeoutError:
            raise Exception(f"Demo timed out after 600 seconds")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Safely extract results with fallbacks
        try:
            word_count = result.get('draft', {}).word_count if result.get('draft') else 0
        except (AttributeError, TypeError):
            word_count = 0
            
        try:
            reading_time = result.get('draft', {}).reading_time if result.get('draft') else 0
        except (AttributeError, TypeError):
            reading_time = 0
        
        # Collect results
        demo_result = {
            "name": demo_name,
            "success": True,
            "duration": duration,
            "word_count": word_count,
            "reading_time": reading_time,
            "seo_score": result.get('metadata', {}).get('seo_score', 'N/A'),
            "output_file": result.get('metadata', {}).get('output_file', 'N/A'),
            "error": None
        }
        
        logger.info(f"Demo completed successfully: {demo_name}, duration: {duration:.1f}s")
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
        
        error_msg = str(e)
        logger.error(f"Demo failed: {demo_name}, error: {error_msg}, duration: {duration:.1f}s")
        
        demo_result = {
            "name": demo_name,
            "success": False,
            "duration": duration,
            "error": error_msg
        }
        
        print(f"Demo Failed: {error_msg}")
        print(f"Common solutions:")
        print(f"   • Check if Ollama is running: ollama serve")
        print(f"   • Verify model is installed: ollama pull {os.getenv('OLLAMA_MODEL', 'llama3.1:8b')}")
        print(f"   • Check system resources (RAM/CPU usage)")
        print(f"   • Verify network connectivity if using remote Ollama")
        print(f"   • Check logs in logs/demo.log for detailed error information")
        return demo_result

async def main():
    """Main demo function with comprehensive error handling"""
    
    logger.info("Starting Content Creation Multi-Agent System Demo")
    print("Content Creation Multi-Agent System Demo")
    print("AAIDC Module 3 Project - Local Ollama Version")
    print("=" * 70)
    
    # Check Ollama configuration
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    logger.info(f"Using Ollama model: {model_name}, server: {base_url}")
    print(f"Using Ollama model: {model_name}")
    print(f"Ollama server: {base_url}")
    
    # Initialize workflow with enhanced error handling
    print("Initializing Content Creation Workflow...")
    workflow = None
    
    try:
        # Validate environment configuration
        if not model_name or not base_url:
            raise ValueError("Missing Ollama configuration. Check OLLAMA_MODEL and OLLAMA_BASE_URL environment variables.")
        
        workflow = ContentCreationWorkflow(model_name=model_name, base_url=base_url)
        logger.info("Workflow initialized successfully")
        print("Workflow initialized successfully")
        
        # Test Ollama connection with timeout
        print("Testing Ollama connection...")
        try:
            test_messages = [HumanMessage(content="Hello, respond with just 'OK' if you can hear me.")]
            response = await asyncio.wait_for(
                workflow.llm.ainvoke(test_messages),
                timeout=30.0  # 30 second timeout for connection test
            )
            
            if response and hasattr(response, 'content'):
                logger.info(f"Ollama connection test successful: {response.content[:50]}")
                print(f"Ollama connection test successful: {response.content[:50]}...")
            else:
                raise Exception("Invalid response from Ollama")
                
        except asyncio.TimeoutError:
            raise Exception("Ollama connection test timed out (30s). Server may be overloaded or unresponsive.")
        except Exception as conn_error:
            raise Exception(f"Ollama connection test failed: {conn_error}")
        
    except Exception as e:
        error_msg = f"Failed to initialize workflow: {e}"
        logger.error(error_msg)
        print(error_msg)
        print("\nTroubleshooting Steps:")
        print("1. Make sure Ollama is running:")
        print("   ollama serve")
        print(f"2. Install the required model:")
        print(f"   ollama pull {model_name}")
        print("3. Test the model:")
        print(f"   ollama run {model_name} \"Hello\"")
        print("4. Check system resources (RAM/CPU)")
        print("5. Verify firewall/network settings if using remote Ollama")
        print("6. Check logs in logs/demo.log for detailed error information")
        return
    
    # Ensure workflow was properly initialized before proceeding
    if not workflow:
        logger.error("Workflow initialization failed")
        print("Critical error: Workflow could not be initialized")
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
            # Run all predefined demos with error recovery
            logger.info(f"Running {len(DEMO_REQUESTS)} predefined demos")
            print(f"\nRunning {len(DEMO_REQUESTS)} Predefined Demos")
            results = []
            
            for i, demo_config in enumerate(DEMO_REQUESTS, 1):
                try:
                    print(f"\n[{i}/{len(DEMO_REQUESTS)}] Processing next demo...")
                    result = await run_single_demo(workflow, demo_config)
                    results.append(result)
                    
                    # Brief pause between demos to prevent resource exhaustion
                    if demo_config != DEMO_REQUESTS[-1]:
                        print("\nPausing 5 seconds before next demo...")
                        await asyncio.sleep(5)
                        
                except Exception as demo_error:
                    logger.error(f"Critical error during demo {i}: {demo_error}")
                    error_result = {
                        "name": demo_config.get('name', f'Demo {i}'),
                        "success": False,
                        "duration": 0,
                        "error": f"Critical demo error: {demo_error}"
                    }
                    results.append(error_result)
                    print(f"Critical error during demo {i}: {demo_error}")
            
            # Summary with error handling
            try:
                print("\nDemo Summary:")
                print("-" * 40)
                successful = sum(1 for r in results if r.get('success', False))
                total_time = sum(r.get('duration', 0) for r in results)
                
                print(f"Successful demos: {successful}/{len(results)}")
                print(f"Total time: {total_time:.1f} seconds")
                if len(results) > 0:
                    print(f"Average time: {total_time/len(results):.1f} seconds")
                
                for result in results:
                    status = "[OK]" if result.get('success', False) else "[FAIL]"
                    duration = result.get('duration', 0)
                    name = result.get('name', 'Unknown')
                    print(f"{status} {name}: {duration:.1f}s")
                    
                    if not result.get('success', False) and 'error' in result:
                        print(f"    Error: {result['error'][:100]}..." if len(result['error']) > 100 else f"    Error: {result['error']}")
                
                logger.info(f"Demo batch completed: {successful}/{len(results)} successful")
                
            except Exception as summary_error:
                logger.error(f"Error generating demo summary: {summary_error}")
                print(f"Error generating summary: {summary_error}")
        
        elif choice == "2":
            print("\nInteractive Demo - Create Custom Content")
            try:
                # Get user input with validation
                topic = input("Enter topic: ").strip()
                if not topic:
                    topic = "Benefits of AI in Education"
                    print(f"Using default topic: {topic}")
                
                # Validate topic length and content
                if len(topic) < 3:
                    print("Topic too short. Using default topic.")
                    topic = "Benefits of AI in Education"
                elif len(topic) > 200:
                    print("Topic too long. Truncating to 200 characters.")
                    topic = topic[:200]
                
                logger.info(f"Interactive demo started with topic: {topic}")
                
                request = ContentRequest(
                    topic=topic,
                    content_type=ContentType.BLOG_POST,
                    target_audience="General audience",
                    word_count=1000,
                    tone="professional",
                    keywords=[topic[:50]],  # Limit keyword length
                    special_requirements="User-generated content"
                )
                
                demo_config = {"name": "Custom Interactive Demo", "request": request}
                await run_single_demo(workflow, demo_config)
                
            except KeyboardInterrupt:
                print("\nInteractive demo cancelled by user.")
                logger.info("Interactive demo cancelled by user")
            except Exception as interactive_error:
                logger.error(f"Interactive demo error: {interactive_error}")
                print(f"Interactive demo failed: {interactive_error}")
        
        elif choice == "3":
            print("\nPerformance Benchmark")
            logger.info("Starting performance benchmark")
            
            try:
                # Simple benchmark with error handling
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
                
                try:
                    result = await asyncio.wait_for(
                        workflow.create_content(request),
                        timeout=300.0  # 5 minute timeout for benchmark
                    )
                except asyncio.TimeoutError:
                    print("Benchmark timed out after 300 seconds")
                    logger.warning("Benchmark timed out")
                    continue
                
                duration = time.time() - start
                
                # Safely extract word count
                try:
                    word_count = result['draft'].word_count if result.get('draft') else 0
                except (AttributeError, TypeError, KeyError):
                    word_count = 0
                
                print(f"Benchmark completed in {duration:.1f} seconds")
                print(f"Words generated: {word_count}")
                
                if duration > 0 and word_count > 0:
                    words_per_second = word_count / duration
                    print(f"Words per second: {words_per_second:.1f}")
                    logger.info(f"Benchmark completed: {duration:.1f}s, {word_count} words, {words_per_second:.1f} w/s")
                else:
                    print("Benchmark completed but no performance metrics available")
                    logger.warning("Benchmark completed with missing metrics")
                
            except Exception as benchmark_error:
                logger.error(f"Benchmark error: {benchmark_error}")
                print(f"Benchmark failed: {benchmark_error}")
        
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
            logger.warning(f"User selected invalid option: {choice}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user. Goodbye!")
        logger.info("Demo interrupted by user (KeyboardInterrupt)")
    except Exception as main_error:
        print(f"Critical error in demo: {main_error}")
        logger.critical(f"Critical demo error: {main_error}")
        sys.exit(1)
