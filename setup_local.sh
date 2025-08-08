#!/bin/bash

# Content Creation Multi-Agent System - Local Setup Script
# AAIDC Module 3 Project - Ollama Implementation
# This script automates the complete setup process for local development

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.8"
RECOMMENDED_RAM_GB=8
RECOMMENDED_STORAGE_GB=10
DEFAULT_MODEL="llama3.1:8b"
FALLBACK_MODEL="phi3:mini"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Functions
print_header() {
    echo -e "${CYAN}"
    echo "Content Creation Multi-Agent System - Local Setup"
    echo "AAIDC Module 3 Project - Ollama Implementation"
    echo "=============================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}Step $1: $2${NC}"
}

print_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_info() {
    echo -e "${PURPLE}[INFO] $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_system_requirements() {
    print_step "1" "Checking System Requirements"
    
    # Check operating system
    OS="$(uname -s)"
    case "${OS}" in
        Linux*)     MACHINE=Linux;;
        Darwin*)    MACHINE=Mac;;
        CYGWIN*)    MACHINE=Cygwin;;
        MINGW*)     MACHINE=MinGw;;
        MSYS*)      MACHINE=Git;;
        *)          MACHINE="UNKNOWN:${OS}"
    esac
    print_info "Detected OS: $MACHINE"
    
    # Check available RAM
    if [[ "$MACHINE" == "Linux" ]]; then
        TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
    elif [[ "$MACHINE" == "Mac" ]]; then
        TOTAL_RAM_BYTES=$(sysctl -n hw.memsize)
        TOTAL_RAM_GB=$((TOTAL_RAM_BYTES / 1024 / 1024 / 1024))
    else
        TOTAL_RAM_GB=0
        print_warning "Cannot determine RAM on this system"
    fi
    
    if [[ $TOTAL_RAM_GB -ge $RECOMMENDED_RAM_GB ]]; then
        print_success "RAM: ${TOTAL_RAM_GB}GB (sufficient for llama3.1:8b)"
    elif [[ $TOTAL_RAM_GB -ge 4 ]]; then
        print_warning "RAM: ${TOTAL_RAM_GB}GB (sufficient for phi3:mini only)"
        DEFAULT_MODEL="phi3:mini"
    else
        print_warning "RAM: ${TOTAL_RAM_GB}GB (may be insufficient)"
    fi
    
    # Check available disk space
    if [[ "$MACHINE" == "Linux" ]] || [[ "$MACHINE" == "Mac" ]]; then
        AVAILABLE_SPACE_KB=$(df . | tail -1 | awk '{print $4}')
        AVAILABLE_SPACE_GB=$((AVAILABLE_SPACE_KB / 1024 / 1024))
        
        if [[ $AVAILABLE_SPACE_GB -ge $RECOMMENDED_STORAGE_GB ]]; then
            print_success "Disk Space: ${AVAILABLE_SPACE_GB}GB available"
        else
            print_warning "Disk Space: Only ${AVAILABLE_SPACE_GB}GB available (${RECOMMENDED_STORAGE_GB}GB recommended)"
        fi
    fi
    
    # Check CPU cores
    if [[ "$MACHINE" == "Linux" ]]; then
        CPU_CORES=$(nproc)
    elif [[ "$MACHINE" == "Mac" ]]; then
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        CPU_CORES="unknown"
    fi
    print_info "CPU Cores: $CPU_CORES"
    
    echo ""
}

# Check and install Python
check_python() {
    print_step "2" "Checking Python Installation"
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 ]] && [[ $PYTHON_MINOR -ge 8 ]]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python $PYTHON_VERSION is too old (need 3.8+)"
            exit 1
        fi
    elif command_exists python; then
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
        if [[ $PYTHON_VERSION == 3.* ]]; then
            PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
            PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
            
            if [[ $PYTHON_MAJOR -eq 3 ]] && [[ $PYTHON_MINOR -ge 8 ]]; then
                print_success "Python $PYTHON_VERSION found"
                PYTHON_CMD="python"
            else
                print_error "Python $PYTHON_VERSION is too old (need 3.8+)"
                exit 1
            fi
        else
            print_error "Python 3.8+ required, found Python $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.8 or higher"
        print_info "Install from: https://python.org/downloads/"
        exit 1
    fi
    
    # Check pip
    if ! command_exists pip3 && ! command_exists pip; then
        print_error "pip not found. Please install pip"
        exit 1
    fi
    
    print_success "Python environment ready"
    echo ""
}

# Install Ollama
install_ollama() {
    print_step "3" "Installing Ollama"
    
    if command_exists ollama; then
        OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
        print_success "Ollama already installed: $OLLAMA_VERSION"
    else
        print_info "Installing Ollama..."
        
        if [[ "$MACHINE" == "Linux" ]] || [[ "$MACHINE" == "Mac" ]]; then
            # Install using official script
            if command_exists curl; then
                curl -fsSL https://ollama.com/install.sh | sh
            elif command_exists wget; then
                wget -qO- https://ollama.com/install.sh | sh
            else
                print_error "Neither curl nor wget found. Please install one of them"
                exit 1
            fi
        else
            print_error "Please install Ollama manually from https://ollama.com/download"
            print_info "After installation, restart this script"
            exit 1
        fi
        
        print_success "Ollama installed successfully"
    fi
    
    echo ""
}

# Start Ollama server
start_ollama() {
    print_step "4" "Starting Ollama Server"
    
    # Check if Ollama is already running
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_success "Ollama server is already running"
    else
        print_info "Starting Ollama server..."
        
        # Start Ollama in background
        ollama serve > /dev/null 2>&1 &
        OLLAMA_PID=$!
        
        # Wait for server to start
        print_info "Waiting for Ollama server to start..."
        for i in {1..30}; do
            if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                print_success "Ollama server started successfully"
                break
            fi
            sleep 2
            echo -n "."
        done
        
        if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_error "Failed to start Ollama server"
            exit 1
        fi
    fi
    
    echo ""
}

# Download Ollama models
download_models() {
    print_step "5" "Downloading Ollama Models"
    
    # Check available models
    AVAILABLE_MODELS=$(ollama list 2>/dev/null || echo "")
    
    # Download default model
    if echo "$AVAILABLE_MODELS" | grep -q "$DEFAULT_MODEL"; then
        print_success "$DEFAULT_MODEL already downloaded"
    else
        print_info "Downloading $DEFAULT_MODEL (this may take a while)..."
        
        if ollama pull "$DEFAULT_MODEL"; then
            print_success "$DEFAULT_MODEL downloaded successfully"
        else
            print_warning "Failed to download $DEFAULT_MODEL, trying fallback model..."
            DEFAULT_MODEL="$FALLBACK_MODEL"
            
            if ollama pull "$FALLBACK_MODEL"; then
                print_success "$FALLBACK_MODEL downloaded successfully"
            else
                print_error "Failed to download any model"
                exit 1
            fi
        fi
    fi
    
    # Download fallback model if not already present
    if [[ "$DEFAULT_MODEL" != "$FALLBACK_MODEL" ]] && ! echo "$AVAILABLE_MODELS" | grep -q "$FALLBACK_MODEL"; then
        print_info "Downloading fallback model $FALLBACK_MODEL..."
        if ollama pull "$FALLBACK_MODEL"; then
            print_success "$FALLBACK_MODEL downloaded as fallback"
        else
            print_warning "Failed to download fallback model (not critical)"
        fi
    fi
    
    # Test model
    print_info "Testing model..."
    if echo "Hello! Respond with just 'OK' if you can hear me." | ollama run "$DEFAULT_MODEL" --stream=false >/dev/null 2>&1; then
        print_success "Model test successful"
    else
        print_warning "Model test failed (may still work)"
    fi
    
    echo ""
}

# Setup Python virtual environment
setup_python_env() {
    print_step "6" "Setting Up Python Environment"
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    if [[ "$MACHINE" == "Linux" ]] || [[ "$MACHINE" == "Mac" ]]; then
        source venv/bin/activate
    else
        # Windows (Git Bash, WSL, etc.)
        source venv/Scripts/activate 2>/dev/null || source venv/bin/activate
    fi
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    print_success "Python environment ready"
    echo ""
}

# Install Python dependencies
install_dependencies() {
    print_step "7" "Installing Python Dependencies"
    
    if [[ -f "requirements.txt" ]]; then
        print_info "Installing from requirements.txt..."
        pip install -r requirements.txt
    else
        print_info "Installing core dependencies..."
        
        # Core dependencies
        pip install langchain>=0.2.0
        pip install langchain-ollama>=0.1.0
        pip install langchain-community>=0.2.0
        pip install langgraph>=0.0.55
        pip install langchain-core>=0.2.20
        pip install ollama>=0.2.1
        
        # Tools and utilities
        pip install duckduckgo-search==5.3.0
        pip install requests==2.31.0
        pip install nltk==3.8.1
        pip install textstat==0.7.3
        pip install python-dotenv==1.0.0
        
        # Testing (optional)
        pip install pytest pytest-asyncio
    fi
    
    print_success "Dependencies installed"
    echo ""
}

# Setup NLTK data
setup_nltk() {
    print_step "8" "Setting Up NLTK Data"
    
    print_info "Downloading NLTK data..."
    $PYTHON_CMD -c "
import nltk
import ssl

# Handle SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

print('NLTK data downloaded successfully')
"
    
    print_success "NLTK data ready"
    echo ""
}

# Create environment configuration
create_config() {
    print_step "9" "Creating Configuration"
    
    # Create .env file
    cat > .env << EOF
# Ollama Configuration
OLLAMA_MODEL=$DEFAULT_MODEL
OLLAMA_BASE_URL=http://localhost:11434

# Advanced Ollama Settings
OLLAMA_TEMPERATURE=0.7
OLLAMA_TOP_P=0.9
OLLAMA_TOP_K=40
OLLAMA_NUM_PREDICT=4096

# Logging
LOG_LEVEL=INFO

# Performance Settings
MAX_CONCURRENT_REQUESTS=2
REQUEST_TIMEOUT=300
EOF
    
    print_success "Configuration file created (.env)"
    
    # Create outputs directory
    mkdir -p outputs
    print_success "Output directory created"
    
    echo ""
}

# Test the system
test_system() {
    print_step "10" "Testing System"
    
    print_info "Running system tests..."
    
    # Test 1: Basic imports
    print_info "Testing Python imports..."
    if $PYTHON_CMD -c "
try:
    from main import ContentCreationWorkflow, ContentRequest, ContentType
    from main import web_search_tool, content_analysis_tool
    print('[OK] All imports successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"; then
        print_success "Import test passed"
    else
        print_error "Import test failed"
        return 1
    fi
    
    # Test 2: Ollama connection
    print_info "Testing Ollama connection..."
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_success "Ollama connection test passed"
    else
        print_error "Ollama connection test failed"
        return 1
    fi
    
    # Test 3: Model functionality
    print_info "Testing model functionality..."
    if echo "Test" | ollama run "$DEFAULT_MODEL" --stream=false >/dev/null 2>&1; then
        print_success "Model test passed"
    else
        print_warning "Model test failed (may still work in application)"
    fi
    
    # Test 4: Web search
    print_info "Testing web search..."
    if $PYTHON_CMD -c "
try:
    from main import web_search_tool
    result = web_search_tool.invoke({'query': 'test', 'max_results': 1})
    print('[OK] Web search test passed')
except Exception as e:
    print(f'[WARNING] Web search test failed: {e}')
"; then
        print_success "Web search test passed"
    else
        print_warning "Web search test failed (network issue?)"
    fi
    
    # Test 5: Content analysis
    print_info "Testing content analysis..."
    if $PYTHON_CMD -c "
try:
    from main import content_analysis_tool
    result = content_analysis_tool.invoke({'content': 'Test content for analysis.'})
    print('[OK] Content analysis test passed')
except Exception as e:
    print(f'❌ Content analysis test failed: {e}')
    exit(1)
"; then
        print_success "Content analysis test passed"
    else
        print_error "Content analysis test failed"
        return 1
    fi
    
    print_success "All tests completed"
    echo ""
    
    return 0
}

# Create launcher scripts
create_launchers() {
    print_step "11" "Creating Launcher Scripts"
    
    # Create start script
    cat > start_system.sh << 'EOF'
#!/bin/bash
# Start Content Creation System

echo "Starting Content Creation Multi-Agent System..."

# Activate virtual environment
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
elif [[ -f "venv/Scripts/activate" ]]; then
    source venv/Scripts/activate
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "📡 Starting Ollama server..."
    ollama serve > /dev/null 2>&1 &
    
    # Wait for Ollama to start
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "[OK] Ollama server started"
            break
        fi
        sleep 1
    done
fi

echo "System ready! You can now run:"
echo "  python main.py          # Basic demo"
echo "  python demo.py          # Interactive demo"
echo "  python test_agents.py   # Run tests"

exec "$@"
EOF

    chmod +x start_system.sh
    print_success "Launcher script created (start_system.sh)"
    
    # Create demo script
    cat > run_demo.sh << 'EOF'
#!/bin/bash
# Quick demo launcher

./start_system.sh python demo.py
EOF

    chmod +x run_demo.sh
    print_success "Demo launcher created (run_demo.sh)"
    
    echo ""
}

# Print final instructions
print_final_instructions() {
    echo -e "${GREEN}"
    echo "🎉 Setup Complete!"
    echo "=================="
    echo -e "${NC}"
    
    echo -e "${CYAN}Your Content Creation Multi-Agent System is ready!${NC}"
    echo ""
    
    echo -e "${YELLOW}What was installed:${NC}"
    echo "  [OK] Ollama server with $DEFAULT_MODEL model"
    echo "  [OK] Python virtual environment with all dependencies"
    echo "  [OK] NLTK data for content analysis"
    echo "  [OK] Configuration files and output directories"
    echo "  [OK] Launcher scripts for easy startup"
    echo ""
    
    echo -e "${YELLOW}Quick Start:${NC}"
    echo "  # Option 1: Use launcher scripts"
    echo "  ./start_system.sh       # Start system"
    echo "  ./run_demo.sh           # Run interactive demo"
    echo ""
    echo "  # Option 2: Manual startup"
    if [[ "$MACHINE" == "Linux" ]] || [[ "$MACHINE" == "Mac" ]]; then
        echo "  source venv/bin/activate"
    else
        echo "  source venv/Scripts/activate  # or venv/bin/activate"
    fi
    echo "  python demo.py          # Interactive demo"
    echo "  python main.py          # Basic demo"
    echo "  python test_agents.py   # Run tests"
    echo ""
    
    echo -e "${YELLOW}System Information:${NC}"
    echo "  Model: $DEFAULT_MODEL"
    echo "  RAM: ${TOTAL_RAM_GB}GB"
    echo "  Ollama: http://localhost:11434"
    echo "  Output: ./outputs/"
    echo ""
    
    echo -e "${YELLOW}Tips:${NC}"
    echo "  • Run 'ollama list' to see downloaded models"
    echo "  • Check 'outputs/' directory for generated content"
    echo "  • Use Ctrl+C to stop the demo"
    echo "  • Edit .env file to change model or settings"
    echo ""
    
    if [[ $TOTAL_RAM_GB -lt $RECOMMENDED_RAM_GB ]]; then
        echo -e "${YELLOW}[WARNING] RAM Warning:${NC}"
        echo "  Your system has ${TOTAL_RAM_GB}GB RAM (${RECOMMENDED_RAM_GB}GB recommended)"
        echo "  Consider using 'phi3:mini' model for better performance"
        echo "  Edit .env file: OLLAMA_MODEL=phi3:mini"
        echo ""
    fi
    
    echo -e "${CYAN}Ready to create amazing content with zero API costs!${NC}"
    echo ""
}

# Cleanup function
cleanup() {
    if [[ -n "$OLLAMA_PID" ]]; then
        print_info "Cleaning up Ollama process..."
        kill $OLLAMA_PID 2>/dev/null || true
    fi
}

# Main execution
main() {
    trap cleanup EXIT
    
    print_header
    
    # Check if we're in the right directory
    if [[ ! -f "main.py" ]] && [[ ! -f "setup_project.py" ]]; then
        print_error "Please run this script from the project directory"
        print_info "Expected files: main.py or setup_project.py"
        exit 1
    fi
    
    # If project files don't exist, offer to create them
    if [[ ! -f "main.py" ]] && [[ -f "setup_project.py" ]]; then
        print_info "Project files not found. Running setup_project.py first..."
        $PYTHON_CMD setup_project.py
        print_success "Project files created"
        echo ""
    fi
    
    # Run setup steps
    check_system_requirements
    check_python
    install_ollama
    start_ollama
    download_models
    setup_python_env
    install_dependencies
    setup_nltk
    create_config
    
    if test_system; then
        create_launchers
        print_final_instructions
    else
        print_error "System tests failed. Please check the errors above."
        exit 1
    fi
}

# Run main function
main "$@"