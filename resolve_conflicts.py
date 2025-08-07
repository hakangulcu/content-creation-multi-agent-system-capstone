#!/usr/bin/env python3
"""
Dependency Conflict Resolution Script
Checks if the essential packages work and resolves conflicts
"""

import sys
import subprocess
import importlib

def test_import(package_name, import_name=None):
    """Test if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}: Successfully imported")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: Import failed - {e}")
        return False

def check_essential_packages():
    """Check if essential packages for our system work"""
    print("🔍 Checking essential packages for Content Creation System...")
    print("-" * 60)
    
    essential_packages = [
        ("langchain", "langchain"),
        ("langchain-ollama", "langchain_ollama"),
        ("langgraph", "langgraph"),
        ("ollama", "ollama"),
        ("duckduckgo-search", "duckduckgo_search"),
        ("nltk", "nltk"),
        ("textstat", "textstat"),
        ("requests", "requests"),
        ("python-dotenv", "dotenv"),
    ]
    
    working_packages = []
    failed_packages = []
    
    for package_name, import_name in essential_packages:
        if test_import(package_name, import_name):
            working_packages.append(package_name)
        else:
            failed_packages.append(package_name)
    
    print(f"\n📊 Results:")
    print(f"✅ Working packages: {len(working_packages)}")
    print(f"❌ Failed packages: {len(failed_packages)}")
    
    if failed_packages:
        print(f"\n🔧 Missing packages: {', '.join(failed_packages)}")
        return False
    else:
        print(f"\n🎉 All essential packages are working!")
        return True

def fix_conflicts():
    """Resolve the specific conflicts mentioned"""
    print("\n🔧 Resolving dependency conflicts...")
    
    # The conflicts we need to fix
    fixes = [
        "pip install --upgrade numpy>=1.26.0",
        "pip install --upgrade beautifulsoup4>=4.13.0",
        "pip install --upgrade chromadb>=1.0.9",
    ]
    
    print("Running fixes...")
    for fix in fixes:
        print(f"  {fix}")
        try:
            subprocess.run(fix.split(), check=True, capture_output=True)
            print(f"  ✅ Success")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Warning: {e}")
    
    # Remove conflicting packages if they're not essential
    print("\n🗑️  Removing non-essential conflicting packages...")
    non_essential = ["langchain-chroma", "langchain-openai", "twscrape"]
    
    for package in non_essential:
        try:
            subprocess.run(["pip", "uninstall", package, "-y"], 
                         check=True, capture_output=True)
            print(f"  ✅ Removed {package}")
        except subprocess.CalledProcessError:
            print(f"  ℹ️  {package} not found (already removed)")

def test_system():
    """Test if our content creation system works"""
    print("\n🧪 Testing Content Creation System...")
    
    try:
        # Test basic imports
        from langchain_ollama import ChatOllama
        from langgraph.graph import StateGraph
        import ollama
        
        print("✅ Core imports successful")
        
        # Test Ollama connection (basic)
        try:
            models = ollama.list()
            print(f"✅ Ollama connection successful - {len(models.get('models', []))} models available")
        except Exception as e:
            print(f"⚠️  Ollama connection issue: {e}")
            print("   Make sure 'ollama serve' is running")
        
        # Test ChatOllama initialization
        try:
            llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")
            print("✅ ChatOllama initialization successful")
        except Exception as e:
            print(f"⚠️  ChatOllama initialization issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main resolution process"""
    print("🔧 Content Creation System - Dependency Resolution")
    print("=" * 60)
    
    # Step 1: Check what's working
    essential_ok = check_essential_packages()
    
    if essential_ok:
        print("\n✅ Great! Essential packages are working.")
        print("   The dependency conflicts are with non-essential packages.")
        
        # Step 2: Test the system
        if test_system():
            print("\n🎉 SUCCESS: Your Content Creation System is ready to use!")
            print("\nNext steps:")
            print("1. Make sure Ollama is running: ollama serve")
            print("2. Pull a model: ollama pull llama3.1:8b")
            print("3. Run the demo: python demo.py")
            return
    
    print("\n🔧 Attempting to fix conflicts...")
    fix_conflicts()
    
    print("\n🔄 Re-checking after fixes...")
    if check_essential_packages() and test_system():
        print("\n🎉 SUCCESS: System is now working!")
    else:
        print("\n⚠️  Some issues remain. See suggestions below.")
        print_suggestions()

def print_suggestions():
    """Print suggestions for manual resolution"""
    print("\n💡 Manual Resolution Suggestions:")
    print("-" * 40)
    print("1. Clean installation approach:")
    print("   rm -rf venv")
    print("   python -m venv venv")
    print("   source venv/bin/activate")
    print("   pip install langchain langchain-ollama ollama langgraph")
    print("   pip install duckduckgo-search nltk textstat requests python-dotenv")
    print()
    print("2. Ignore conflicts and test:")
    print("   python demo.py")
    print("   (The system might work despite warnings)")
    print()
    print("3. Use Docker:")
    print("   docker run -it python:3.11 bash")
    print("   pip install langchain langchain-ollama ollama langgraph")

if __name__ == "__main__":
    main()