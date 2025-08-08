# Ollama Setup Guide
## Complete Local LLM Setup for Content Creation Multi-Agent System

This comprehensive guide will help you set up Ollama to run the Content Creation Multi-Agent System locally with **zero API costs** and **full privacy**.

## Quick Start (5 Minutes)

### Step 1: Install Ollama

#### Windows Installation
```bash
# Download and run installer
# https://ollama.com/download/windows
# Ollama will start automatically after installation
```

#### macOS Installation  
```bash
# Using curl (recommended)
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download/macos
```

#### Linux Installation
```bash
# Ubuntu/Debian
curl -fsSL https://ollama.com/install.sh | sh

# Arch Linux
sudo pacman -S ollama

# Manual installation
curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama
chmod +x /usr/local/bin/ollama
```

#### Docker Installation (Alternative)
```bash
# Basic Docker setup
docker pull ollama/ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# With GPU support (NVIDIA)
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Start container
docker start ollama
```

### Step 2: Start Ollama Server

```bash
# Start the Ollama server (keep this running)
ollama serve

# Server will start on http://localhost:11434
# You should see: "Ollama is running"
```

**Important Notes:**
- On **Windows**: Ollama typically starts automatically as a service
- On **macOS/Linux**: You may need to run `ollama serve` manually
- Keep this terminal window open while using the system

### Step 3: Download and Test a Model

```bash
# Recommended for most users (good balance of quality and speed)
ollama pull llama3.1:8b

# Test the model
ollama run llama3.1:8b "Hello! Please introduce yourself."

# If successful, you'll see the model respond
```

### Step 4: Verify API Access

```bash
# Test the Ollama API
curl http://localhost:11434/api/tags

# Should return JSON with available models
# Example response:
# {"models":[{"name":"llama3.1:8b","size":4661224448,"digest":"sha256:..."}]}
```

**If all steps work, you're ready to run the Content Creation System!**

## Model Selection Guide

### Performance vs Quality Trade-offs

| Model | Download Size | RAM Required | Speed | Quality | Best For |
|-------|--------------|--------------|-------|---------|----------|
| `phi3:mini` | 2.3GB | 4GB | Very Fast | Good | Quick testing, low-resource systems |
| `mistral:7b` | 4.1GB | 6GB | Fast | Very Good | Balanced usage, good performance |
| `llama3.1:8b` | 4.7GB | 8GB | Fast | Excellent | **Recommended for production** |
| `codellama:7b` | 3.8GB | 6GB | Fast | Very Good | Technical content, code examples |
| `llama3.1:70b` | 39GB | 64GB | Slow | Excellent | Maximum quality, powerful hardware |

### Model Installation Commands

```bash
# Start with the smallest for testing
ollama pull phi3:mini

# Upgrade to balanced performance  
ollama pull mistral:7b

# Production-ready model (recommended)
ollama pull llama3.1:8b

# For technical content
ollama pull codellama:7b

# Maximum quality (requires powerful hardware)
ollama pull llama3.1:70b

# List installed models
ollama list
```

### Model Selection Strategy

1. **First-time users**: Start with `phi3:mini` to test the system
2. **Regular usage**: Upgrade to `llama3.1:8b` for best results
3. **Technical content**: Use `codellama:7b` for code-heavy articles
4. **High-end systems**: Try `llama3.1:70b` for maximum quality

## System Requirements

### Minimum Requirements
- **CPU**: Modern 4-core processor (Intel i5 or AMD Ryzen 5)
- **RAM**: 4GB (for phi3:mini) to 8GB (for llama3.1:8b)
- **Storage**: 5GB for model files + 2GB for dependencies
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Network**: Internet for initial model download (then works offline)

### Recommended Requirements
- **CPU**: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 16GB+ (allows for larger models and better performance)
- **Storage**: SSD with 20GB+ free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but significantly faster)

### Optimal Setup
- **CPU**: High-end desktop processor
- **RAM**: 32GB+ (for llama3.1:70b and concurrent operations)
- **Storage**: NVMe SSD
- **GPU**: NVIDIA RTX 4080/4090 or A100 (for fastest inference)

## Content Creation System Setup

### Step 1: Project Installation

```bash
# Clone the repository  
git clone <your-repository-url>
cd content-creation-multi-agent-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Step 2: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional - defaults work fine)
nano .env  # or your preferred editor
```

**Default .env configuration:**
```env
# Ollama Configuration
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434

# Advanced Settings (optional)
OLLAMA_TEMPERATURE=0.7
OLLAMA_TOP_P=0.9
OLLAMA_TOP_K=40
OLLAMA_NUM_PREDICT=4096
```

### Step 3: Test the Complete System

```bash
# Quick system test
python main.py

# Interactive demo with multiple scenarios
python demo.py

# Run test suite
pytest test_agents.py -v
```

### Step 4: Verify Everything Works

Expected output from `python main.py`:
```
Using Ollama model: llama3.1:8b
Ollama server: http://localhost:11434
[OK] Ollama connection established
Starting Multi-Agent Content Creation System
Topic: Artificial Intelligence in Healthcare
Target: Healthcare professionals and technology leaders
Length: 1500 words
------------------------------------------------------------
[OK] Content Creation Completed Successfully!
Final word count: 1543
Reading time: 8 minutes
📁 Saved to: outputs/Artificial_Intelligence_in_Healthcare_20250714_153045.md
SEO Score: 87
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. "Connection refused" or "Connection error"

**Problem**: Cannot connect to Ollama server
**Solutions**:
```bash
# Check if Ollama is running
ps aux | grep ollama  # Linux/macOS
tasklist | findstr ollama  # Windows

# Start Ollama server
ollama serve

# Check port availability
curl http://localhost:11434/api/tags

# If port 11434 is busy, find what's using it
lsof -i :11434  # Linux/macOS
netstat -ano | findstr :11434  # Windows
```

#### 2. "Model not found" or "Model not available"

**Problem**: Requested model isn't downloaded
**Solutions**:
```bash
# List available models
ollama list

# Pull the required model
ollama pull llama3.1:8b

# Verify model works
ollama run llama3.1:8b "Test message"

# Update .env if using different model
echo "OLLAMA_MODEL=phi3:mini" >> .env
```

#### 3. "Out of memory" or System Freeze

**Problem**: Not enough RAM for the selected model
**Solutions**:
```bash
# Check memory usage
free -h  # Linux
vm_stat  # macOS
taskmgr  # Windows (Performance tab)

# Try smaller model
ollama pull phi3:mini
# Update .env
echo "OLLAMA_MODEL=phi3:mini" > .env

# Close unnecessary applications
# Add swap space (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Very Slow Generation

**Problem**: Model generates text extremely slowly
**Solutions**:
```bash
# Check CPU usage
htop  # Linux/macOS
taskmgr  # Windows

# Try smaller model for speed
ollama pull phi3:mini

# Enable GPU acceleration (if available)
# Install NVIDIA drivers and CUDA toolkit
nvidia-smi  # Check GPU availability

# Adjust model parameters in .env
echo "OLLAMA_NUM_PREDICT=2048" >> .env  # Shorter responses
echo "OLLAMA_TEMPERATURE=0.5" >> .env   # More focused output
```

#### 5. Python Import Errors

**Problem**: Missing dependencies or import failures
**Solutions**:
```bash
# Reinstall requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install specific problematic packages
pip install langchain-ollama ollama langgraph

# Check for conflicting packages
pip list | grep -i langchain

# Clean install (if needed)
pip freeze | xargs pip uninstall -y
pip install -r requirements.txt
```

#### 6. Demo Script Errors

**Problem**: Demo fails with various errors
**Solutions**:
```bash
# Run with debug logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import demo
"

# Test individual components
python -c "
from main import web_search_tool
print(web_search_tool.invoke({'query': 'test', 'max_results': 1}))
"

# Verify NLTK data
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
"
```

## Performance Optimization

### Speed Optimization

1. **Hardware Improvements**
   ```bash
   # Check current performance
   ollama run llama3.1:8b "Generate a 100-word paragraph about AI."
   # Time this command
   
   # GPU acceleration (if available)
   # Verify GPU is detected
   nvidia-smi
   
   # Use SSD for model storage
   # Models are stored in ~/.ollama/models
   ```

2. **Model Parameter Tuning**
   ```env
   # In .env file - balance speed vs quality
   OLLAMA_TEMPERATURE=0.5     # Lower = more focused, faster
   OLLAMA_TOP_P=0.8          # Lower = fewer token choices
   OLLAMA_TOP_K=20           # Lower = faster generation
   OLLAMA_NUM_PREDICT=2048   # Lower = shorter responses
   ```

3. **System Optimization**
   ```bash
   # Increase system priority (Linux/macOS)
   sudo nice -n -10 ollama serve
   
   # Close unnecessary applications
   # Ensure sufficient free RAM
   # Use performance power profile
   ```

### Quality Optimization

1. **Model Selection**
   ```bash
   # For highest quality content
   ollama pull llama3.1:70b  # Requires 64GB RAM
   
   # For technical accuracy
   ollama pull codellama:7b
   
   # For balanced quality/speed
   ollama pull llama3.1:8b  # Recommended
   ```

2. **Parameter Tuning for Quality**
   ```env
   # In .env file - optimize for quality
   OLLAMA_TEMPERATURE=0.7     # Higher = more creative
   OLLAMA_TOP_P=0.95         # Higher = more diverse
   OLLAMA_NUM_PREDICT=4096   # Higher = longer responses
   OLLAMA_REPEAT_PENALTY=1.1 # Reduce repetition
   ```

## Performance Benchmarks

### Expected Performance (llama3.1:8b)

| Hardware Configuration | Content Generation Time | Words/Minute |
|----------------------|------------------------|--------------|
| **Basic** (8GB RAM, 4-core CPU) | 4-6 minutes | 250-375 |
| **Standard** (16GB RAM, 8-core CPU) | 3-4 minutes | 375-500 |
| **High-end** (32GB RAM, GPU) | 1-2 minutes | 750-1500 |

### Real-world Timing Examples

```bash
# Benchmark your system
time python -c "
import asyncio
from main import ContentCreationWorkflow, ContentRequest, ContentType

async def benchmark():
    workflow = ContentCreationWorkflow()
    request = ContentRequest(
        topic='AI in Education',
        content_type=ContentType.BLOG_POST,
        target_audience='Teachers',
        word_count=800,
        tone='informative'
    )
    result = await workflow.create_content(request)
    print(f'Generated {result.draft.word_count} words')

asyncio.run(benchmark())
"
```

## Offline Usage

### Complete Offline Operation

Once models are downloaded, the system works **completely offline**:

```bash
# Download all models while online
ollama pull llama3.1:8b
ollama pull phi3:mini

# Test offline operation
# 1. Disconnect from internet
# 2. Run the content creation system
python demo.py

# Only web search will fail (gracefully handled)
# All other functions work perfectly offline
```

### Offline Content Generation

```python
# Create offline-optimized content request
request = ContentRequest(
    topic="Local AI Development Best Practices",
    content_type=ContentType.BLOG_POST,
    target_audience="Developers",
    word_count=1200,
    tone="technical",
    keywords=["local AI", "offline development"],
    special_requirements="Focus on offline capabilities, no web research needed"
)
```

## Privacy & Security

### Data Privacy Advantages

- **Zero external API calls** - All processing happens locally
- **No data logging** - Your content never leaves your machine
- **No internet required** - Works completely offline after setup
- **Full control** - You own and control all components
- **GDPR compliant** - No personal data transmitted anywhere

### Security Best Practices

```bash
# Secure model storage (Linux/macOS)
chmod 700 ~/.ollama
chmod 600 ~/.ollama/models/*

# Firewall configuration (optional)
# Block Ollama port from external access
sudo ufw deny 11434  # Linux
# Or configure Windows Firewall to block port 11434
```

## Next Steps

### Once Ollama is Running

1. **Basic Test**
   ```bash
   python main.py
   ```

2. **Interactive Demo**
   ```bash
   python demo.py
   ```

3. **Custom Content Creation**
   ```python
   # Create your own content request
   from main import ContentCreationWorkflow, ContentRequest, ContentType
   
   request = ContentRequest(
       topic="Your Topic Here",
       content_type=ContentType.BLOG_POST,
       target_audience="Your Audience",
       word_count=1000,
       tone="your preferred tone"
   )
   ```

4. **Performance Optimization**
   - Try different models
   - Adjust parameters in .env
   - Monitor resource usage

5. **Scale Up**
   - Use larger models for better quality
   - Enable GPU acceleration
   - Increase concurrent processing

## Pro Tips

### Performance Tips

1. **Model Selection**: Start with `phi3:mini`, upgrade to `llama3.1:8b` when comfortable
2. **SSD Storage**: Store models on SSD for 2-3x faster loading
3. **RAM Optimization**: Close browsers and unnecessary apps while running
4. **GPU Acceleration**: Massive speed improvement if you have NVIDIA GPU
5. **Background Running**: Use `tmux` or `screen` to run Ollama server in background

### Quality Tips

1. **Model Warm-up**: Run a quick test prompt before generating content
2. **Parameter Tuning**: Adjust temperature based on content type
3. **Prompt Engineering**: More specific prompts yield better results
4. **Iterative Generation**: Use revision and feedback loops
5. **Model Switching**: Use specialized models for specific content types

### Operational Tips

1. **Monitoring**: Keep an eye on RAM and CPU usage
2. **Backup**: Models are stored in `~/.ollama/models` - back up if needed
3. **Updates**: Regularly update Ollama with `ollama --version`
4. **Cleanup**: Remove unused models with `ollama rm model-name`
5. **Logging**: Enable debug logging for troubleshooting

---

## Success Indicators

You know everything is working correctly when:

[OK] `ollama serve` starts without errors  
[OK] `ollama list` shows your downloaded models  
[OK] `curl http://localhost:11434/api/tags` returns JSON  
[OK] `python main.py` generates content successfully  
[OK] `python demo.py` runs the interactive demo  
[OK] Content files appear in the `outputs/` directory  

**Congratulations! You now have a fully functional, private, cost-free content creation system!**

---

**Last Updated**: August 2025  
**Ollama Version**: 0.2.1+  
**Compatibility**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)