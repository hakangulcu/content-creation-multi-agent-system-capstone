"""Unit tests for content creation tools."""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import json

from main import (
    _web_search_function as web_search_tool,
    _content_analysis_function as content_analysis_tool,
    _seo_optimization_function as seo_optimization_tool,
    _save_content_function as save_content_tool
)


@pytest.mark.unit
class TestWebSearchTool:
    """Test cases for web_search_tool."""
    
    @patch('main.DuckDuckGoSearchRun')
    def test_web_search_success(self, mock_search_class):
        """Test successful web search."""
        # Setup mock
        mock_search = MagicMock()
        mock_search_class.return_value = mock_search
        mock_search.run.return_value = "Result 1: AI advances\nResult 2: Healthcare tech\nResult 3: Medical AI"
        
        # Execute
        results = web_search_tool("AI in healthcare", max_results=3)
        
        # Verify
        assert len(results) == 3
        assert results[0]["title"] == "Result 1"
        assert "AI advances" in results[0]["snippet"]
        mock_search.run.assert_called_once_with("AI in healthcare")
    
    @patch('main.DuckDuckGoSearchRun')
    def test_web_search_empty_results(self, mock_search_class):
        """Test web search with empty results."""
        mock_search = MagicMock()
        mock_search_class.return_value = mock_search
        mock_search.run.return_value = ""
        
        results = web_search_tool("nonexistent topic")
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    @patch('main.DuckDuckGoSearchRun')
    def test_web_search_exception(self, mock_search_class):
        """Test web search with exception."""
        mock_search_class.side_effect = Exception("Network error")
        
        results = web_search_tool("test query")
        
        assert len(results) == 1
        assert results[0]["title"] == "Error"
        assert "Search Unavailable" in results[0]["snippet"]


@pytest.mark.unit
class TestContentAnalysisTool:
    """Test cases for content_analysis_tool."""
    
    @patch('main.nltk')
    @patch('main.flesch_reading_ease')
    @patch('main.flesch_kincaid_grade')
    def test_content_analysis_success(self, mock_grade, mock_ease, mock_nltk):
        """Test successful content analysis."""
        # Setup mocks
        mock_ease.return_value = 65.5
        mock_grade.return_value = 8.2
        mock_nltk.data.find.return_value = True
        
        content = "This is a test article with several sentences. It contains multiple words for analysis."
        
        result = content_analysis_tool(content)
        
        assert result["readability_score"] == 65.5
        assert result["grade_level"] == 8.2
        assert result["word_count"] == 14
        assert result["reading_time"] == 1
        assert "keyword_density" in result
        assert "analysis_timestamp" in result
    
    @patch('main.nltk')
    def test_content_analysis_nltk_download(self, mock_nltk):
        """Test NLTK data download when missing."""
        mock_nltk.data.find.side_effect = LookupError("Not found")
        mock_nltk.download = MagicMock()
        
        with patch('main.flesch_reading_ease', return_value=70.0), \
             patch('main.flesch_kincaid_grade', return_value=7.0):
            
            result = content_analysis_tool("Test content")
            
            mock_nltk.download.assert_called_once_with('punkt')
            assert "readability_score" in result
    
    def test_content_analysis_exception(self):
        """Test content analysis with exception."""
        with patch('main.flesch_reading_ease', side_effect=Exception("Analysis error")):
            result = content_analysis_tool("test")
            
            assert "error" in result
            assert "Analysis temporarily unavailable" in result["error"]


@pytest.mark.unit
class TestSEOOptimizationTool:
    """Test cases for seo_optimization_tool."""
    
    def test_seo_optimization_basic(self):
        """Test basic SEO optimization."""
        content = "# AI in Healthcare\n\nAI is revolutionizing healthcare. AI helps doctors diagnose diseases faster."
        keywords = ["AI", "healthcare", "doctors"]
        
        result = seo_optimization_tool(content, keywords)
        
        assert "keyword_analysis" in result
        assert "suggestions" in result
        assert "word_count" in result
        assert "seo_score" in result
        
        # Check keyword analysis
        assert result["keyword_analysis"]["AI"] >= 1
        assert result["keyword_analysis"]["healthcare"] >= 1
    
    def test_seo_optimization_missing_keywords(self):
        """Test SEO optimization with missing keywords."""
        content = "This article talks about technology and innovation."
        keywords = ["AI", "healthcare", "medicine"]
        
        result = seo_optimization_tool(content, keywords)
        
        suggestions = result["suggestions"]
        assert any("AI" in suggestion for suggestion in suggestions)
        assert any("healthcare" in suggestion for suggestion in suggestions)
        assert any("medicine" in suggestion for suggestion in suggestions)
    
    def test_seo_optimization_overused_keywords(self):
        """Test SEO optimization with overused keywords."""
        content = " ".join(["AI"] * 15 + ["in", "healthcare"] * 5)
        keywords = ["AI"]
        
        result = seo_optimization_tool(content, keywords)
        
        suggestions = result["suggestions"]
        assert any("overused" in suggestion.lower() for suggestion in suggestions)
    
    def test_seo_optimization_no_title(self):
        """Test SEO optimization without title."""
        content = "Content without a proper title or heading."
        keywords = ["test"]
        
        result = seo_optimization_tool(content, keywords)
        
        suggestions = result["suggestions"]
        assert any("title" in suggestion.lower() for suggestion in suggestions)
    
    def test_seo_optimization_content_length(self):
        """Test SEO optimization content length checks."""
        # Short content
        short_content = "Very short."
        result = seo_optimization_tool(short_content, ["test"])
        assert any("short" in suggestion.lower() for suggestion in result["suggestions"])
        
        # Long content
        long_content = " ".join(["word"] * 3001)
        result = seo_optimization_tool(long_content, ["test"])
        assert any("long" in suggestion.lower() for suggestion in result["suggestions"])
    
    def test_seo_optimization_exception(self):
        """Test SEO optimization with exception."""
        with patch('builtins.len', side_effect=Exception("Length error")):
            result = seo_optimization_tool("test", ["keyword"])
            
            assert "error" in result
            assert "SEO analysis temporarily unavailable" in result["error"]


@pytest.mark.unit
class TestSaveContentTool:
    """Test cases for save_content_tool."""
    
    def test_save_content_success(self, temp_output_dir):
        """Test successful content saving."""
        content = "# Test Article\n\nThis is test content."
        filename = "test_article.md"
        
        with patch('main.os.makedirs') as mock_makedirs, \
             patch('main.os.path.join', return_value=f"{temp_output_dir}/{filename}"), \
             patch('builtins.open', mock_open()) as mock_file:
            
            result = save_content_tool(content, filename)
            
            assert result["status"] == "success"
            assert filename in result["filepath"]
            assert "Content saved" in result["message"]
            mock_file.assert_called_once()
            mock_makedirs.assert_called_once_with("outputs", exist_ok=True)
    
    def test_save_content_failure(self):
        """Test content saving failure."""
        content = "Test content"
        filename = "test.md"
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            result = save_content_tool(content, filename)
            
            assert result["status"] == "error"
            assert "Failed to save" in result["message"]
            assert "Access denied" in result["message"]
    
    def test_save_content_directory_creation(self):
        """Test outputs directory creation."""
        content = "Test content"
        filename = "test.md"
        
        with patch('main.os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()):
            
            save_content_tool(content, filename)
            
            mock_makedirs.assert_called_once_with("outputs", exist_ok=True)


@pytest.mark.unit
class TestToolIntegration:
    """Test cases for tool integration scenarios."""
    
    @patch('main.DuckDuckGoSearchRun')
    def test_search_to_analysis_pipeline(self, mock_search_class):
        """Test pipeline from search results to content analysis."""
        # Setup search mock
        mock_search = MagicMock()
        mock_search_class.return_value = mock_search
        mock_search.run.return_value = "Healthcare AI research shows 40% improvement in diagnosis accuracy."
        
        # Get search results
        search_results = web_search_tool("healthcare AI")
        
        # Analyze the search content
        if search_results:
            content_to_analyze = search_results[0]["snippet"]
            with patch('main.flesch_reading_ease', return_value=65.0), \
                 patch('main.flesch_kincaid_grade', return_value=8.0), \
                 patch('main.nltk.data.find', return_value=True):
                
                analysis = content_analysis_tool(content_to_analyze)
        
        assert search_results
        assert analysis["readability_score"] == 65.0
        assert analysis["word_count"] > 0
    
    def test_analysis_to_seo_pipeline(self):
        """Test pipeline from content analysis to SEO optimization."""
        content = "# AI Healthcare Revolution\n\nAI transforms healthcare through machine learning and data analysis."
        keywords = ["AI", "healthcare", "machine learning"]
        
        # First analyze content
        with patch('main.flesch_reading_ease', return_value=70.0), \
             patch('main.flesch_kincaid_grade', return_value=7.5), \
             patch('main.nltk.data.find', return_value=True):
            
            analysis = content_analysis_tool(content)
        
        # Then optimize for SEO
        seo_result = seo_optimization_tool(content, keywords)
        
        assert analysis["readability_score"] == 70.0
        assert seo_result["seo_score"] > 0
        assert "AI" in seo_result["keyword_analysis"]
        assert "healthcare" in seo_result["keyword_analysis"]