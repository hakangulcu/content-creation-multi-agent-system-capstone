"""
Security utilities for input validation, sanitization, and content filtering.

This module provides comprehensive security features for the content creation system:
- Input validation and sanitization
- Content filtering and moderation
- Structured security logging
- Rate limiting and abuse prevention
"""

import re
import html
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import structlog

from types_shared import ContentType, ContentRequest


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SecurityThreat(Enum):
    """Types of security threats detected."""
    MALICIOUS_INPUT = "malicious_input"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERNS = "suspicious_patterns"
    CONTENT_TOO_LARGE = "content_too_large"
    INVALID_FORMAT = "invalid_format"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: Optional[Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    threat_level: str = "none"  # none, low, medium, high
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class SecurityLogger:
    """Structured security logger."""
    
    def __init__(self):
        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        self.logger = structlog.get_logger("security")
    
    def log_security_event(self, event_type: SecurityThreat, message: str, 
                          context: Dict[str, Any] = None, severity: str = "warning"):
        """Log security event with structured data."""
        if context is None:
            context = {}
        
        self.logger.log(
            level=getattr(logging, severity.upper(), logging.WARNING),
            event_type=event_type.value,
            message=message,
            timestamp=datetime.now().isoformat(),
            **context
        )
    
    def log_validation_failure(self, input_type: str, errors: List[str], 
                              input_hash: str = None):
        """Log validation failure."""
        self.log_security_event(
            SecurityThreat.MALICIOUS_INPUT,
            f"Input validation failed for {input_type}",
            {
                "input_type": input_type,
                "errors": errors,
                "input_hash": input_hash,
                "source": "input_validator"
            },
            severity="error"
        )
    
    def log_content_filtering(self, content_type: str, filtered_items: List[str]):
        """Log content filtering actions."""
        self.log_security_event(
            SecurityThreat.INAPPROPRIATE_CONTENT,
            f"Content filtered for {content_type}",
            {
                "content_type": content_type,
                "filtered_items": filtered_items,
                "source": "content_filter"
            }
        )


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    # Malicious patterns to detect
    MALICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # JavaScript injection
        r'javascript:',                # JavaScript URLs
        r'vbscript:',                 # VBScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'<iframe[^>]*>',             # Iframe injection
        r'<object[^>]*>',             # Object injection
        r'<embed[^>]*>',              # Embed injection
        r'<link[^>]*>',               # Link injection (stylesheets)
        r'<meta[^>]*http-equiv',      # Meta refresh injection
        r'sql\s+(select|insert|update|delete|drop|create|alter)',  # SQL injection
        r'union\s+select',            # SQL union attacks
        r'drop\s+table',              # SQL drop attacks
        r'exec\s*\(',                 # Code execution
        r'eval\s*\(',                 # JavaScript eval
        r'system\s*\(',               # System calls
        r'subprocess\s*\.',           # Python subprocess
        r'os\s*\.',                   # OS module calls
        r'__import__\s*\(',           # Python import attacks
        r'file://',                   # File protocol
        r'ftp://',                    # FTP protocol
        r'ldap://',                   # LDAP protocol
    ]
    
    # Inappropriate content patterns
    INAPPROPRIATE_PATTERNS = [
        r'\b(hate|racism|sexism|discrimination)\b',
        r'\b(violence|kill|murder|bomb|weapon)\b',
        r'\b(illegal|drugs|narcotics|trafficking)\b',
        r'\b(fraud|scam|phishing|spam)\b',
    ]
    
    def __init__(self):
        self.security_logger = SecurityLogger()
        self.rate_limiter = RateLimiter()
    
    def _calculate_input_hash(self, input_data: str) -> str:
        """Calculate hash of input for logging without storing sensitive data."""
        return hashlib.sha256(input_data.encode()).hexdigest()[:16]
    
    def _detect_malicious_patterns(self, text: str) -> List[str]:
        """Detect malicious patterns in text."""
        detected = []
        text_lower = text.lower()
        
        for pattern in self.MALICIOUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                detected.append(f"Malicious pattern detected: {pattern[:50]}...")
        
        return detected
    
    def _detect_inappropriate_content(self, text: str) -> List[str]:
        """Detect inappropriate content patterns."""
        detected = []
        text_lower = text.lower()
        
        for pattern in self.INAPPROPRIATE_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                detected.append(f"Inappropriate content: {pattern[:30]}...")
        
        return detected
    
    def _sanitize_html(self, text: str) -> str:
        """Sanitize HTML content."""
        # Escape HTML characters
        sanitized = html.escape(text)
        
        # Remove potentially dangerous HTML tags
        dangerous_tags = [
            'script', 'iframe', 'object', 'embed', 'link', 'meta',
            'style', 'form', 'input', 'button', 'frame', 'frameset'
        ]
        
        for tag in dangerous_tags:
            pattern = rf'<{tag}[^>]*>.*?</{tag}>'
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            pattern = rf'<{tag}[^>]*/?>'
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _validate_length(self, text: str, max_length: int = 50000) -> List[str]:
        """Validate text length."""
        errors = []
        if len(text) > max_length:
            errors.append(f"Text exceeds maximum length of {max_length} characters")
        return errors
    
    def _validate_encoding(self, text: str) -> List[str]:
        """Validate text encoding."""
        errors = []
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            errors.append("Invalid character encoding detected")
        return errors
    
    def validate_content_request(self, request: ContentRequest) -> ValidationResult:
        """Validate ContentRequest input."""
        errors = []
        warnings = []
        threat_level = "none"
        
        # Check rate limiting
        client_id = self._calculate_input_hash(request.topic + str(time.time()))
        if not self.rate_limiter.allow_request(client_id):
            errors.append("Rate limit exceeded")
            threat_level = "medium"
            self.security_logger.log_security_event(
                SecurityThreat.RATE_LIMIT_EXCEEDED,
                "Rate limit exceeded for content request",
                {"client_id": client_id}
            )
        
        # Validate topic
        if not request.topic or len(request.topic.strip()) < 5:
            errors.append("Topic must be at least 5 characters long")
        
        if len(request.topic) > 500:
            errors.append("Topic exceeds maximum length of 500 characters")
            threat_level = max(threat_level, "low")
        
        # Check for malicious patterns in topic
        malicious = self._detect_malicious_patterns(request.topic)
        if malicious:
            errors.extend(malicious)
            threat_level = "high"
        
        # Check for inappropriate content
        inappropriate = self._detect_inappropriate_content(request.topic)
        if inappropriate:
            warnings.extend(inappropriate)
            threat_level = max(threat_level, "medium")
        
        # Validate content type
        if not isinstance(request.content_type, ContentType):
            errors.append("Invalid content type")
        
        # Validate target audience
        if not request.target_audience or len(request.target_audience.strip()) < 3:
            errors.append("Target audience must be at least 3 characters long")
        
        if len(request.target_audience) > 200:
            errors.append("Target audience exceeds maximum length")
        
        # Validate word count
        if request.word_count < 50:
            errors.append("Word count must be at least 50")
        
        if request.word_count > 10000:
            errors.append("Word count exceeds maximum of 10,000")
            threat_level = max(threat_level, "low")
        
        # Validate keywords
        if request.keywords:
            if len(request.keywords) > 20:
                errors.append("Maximum 20 keywords allowed")
            
            for keyword in request.keywords:
                if len(keyword) > 100:
                    errors.append(f"Keyword '{keyword[:20]}...' exceeds maximum length")
                
                keyword_malicious = self._detect_malicious_patterns(keyword)
                if keyword_malicious:
                    errors.extend(keyword_malicious)
                    threat_level = "high"
        
        # Validate special requirements
        if request.special_requirements:
            if len(request.special_requirements) > 1000:
                errors.append("Special requirements exceed maximum length")
            
            req_malicious = self._detect_malicious_patterns(request.special_requirements)
            if req_malicious:
                errors.extend(req_malicious)
                threat_level = "high"
        
        # Log validation results
        if errors:
            input_hash = self._calculate_input_hash(request.topic)
            self.security_logger.log_validation_failure(
                "ContentRequest", errors, input_hash
            )
        
        # Create sanitized version
        sanitized_request = None
        if not errors:
            sanitized_request = ContentRequest(
                topic=self._sanitize_html(request.topic.strip()),
                content_type=request.content_type,
                target_audience=self._sanitize_html(request.target_audience.strip()),
                word_count=min(max(request.word_count, 50), 10000),
                tone=self._sanitize_html(request.tone.strip()) if request.tone else "professional",
                keywords=[self._sanitize_html(kw.strip()) for kw in (request.keywords or [])][:20],
                special_requirements=self._sanitize_html(request.special_requirements.strip()) 
                                   if request.special_requirements else ""
            )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized_request,
            errors=errors,
            warnings=warnings,
            threat_level=threat_level
        )
    
    def validate_text_content(self, content: str, content_type: str = "general") -> ValidationResult:
        """Validate text content."""
        errors = []
        warnings = []
        threat_level = "none"
        
        # Basic validation
        errors.extend(self._validate_length(content))
        errors.extend(self._validate_encoding(content))
        
        # Security checks
        malicious = self._detect_malicious_patterns(content)
        if malicious:
            errors.extend(malicious)
            threat_level = "high"
        
        inappropriate = self._detect_inappropriate_content(content)
        if inappropriate:
            warnings.extend(inappropriate)
            threat_level = max(threat_level, "medium")
        
        # Sanitize content
        sanitized_content = self._sanitize_html(content) if not errors else None
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized_content,
            errors=errors,
            warnings=warnings,
            threat_level=threat_level
        )


class ContentFilter:
    """Content filtering and moderation."""
    
    BLOCKED_KEYWORDS = [
        # Violence and harmful content
        "violence", "kill", "murder", "bomb", "weapon", "gun", "knife", "attack",
        # Illegal activities
        "illegal", "drugs", "narcotics", "trafficking", "fraud", "scam", "phishing",
        # Hate speech
        "hate", "racism", "sexism", "discrimination", "nazi", "terrorist",
        # Adult content
        "explicit", "nsfw", "pornography", "sexual", "adult",
        # Personal information
        "ssn", "social security", "credit card", "password", "bank account"
    ]
    
    def __init__(self):
        self.security_logger = SecurityLogger()
    
    def filter_content(self, content: str) -> Tuple[str, List[str]]:
        """Filter inappropriate content and return cleaned version."""
        filtered_items = []
        filtered_content = content
        
        # Check for blocked keywords
        content_lower = content.lower()
        for keyword in self.BLOCKED_KEYWORDS:
            if keyword in content_lower:
                filtered_items.append(keyword)
                # Replace with placeholder
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                filtered_content = pattern.sub("[FILTERED]", filtered_content)
        
        # Remove potential personal information patterns
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, filtered_content):
            filtered_items.append("email_address")
            filtered_content = re.sub(email_pattern, "[EMAIL_FILTERED]", filtered_content)
        
        # Phone numbers (basic pattern)
        phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b'
        if re.search(phone_pattern, filtered_content):
            filtered_items.append("phone_number")
            filtered_content = re.sub(phone_pattern, "[PHONE_FILTERED]", filtered_content)
        
        # Credit card numbers (basic pattern)
        cc_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        if re.search(cc_pattern, filtered_content):
            filtered_items.append("credit_card")
            filtered_content = re.sub(cc_pattern, "[CC_FILTERED]", filtered_content)
        
        if filtered_items:
            self.security_logger.log_content_filtering("text_content", filtered_items)
        
        return filtered_content, filtered_items


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 10, time_window: int = 300):  # 10 requests per 5 minutes
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, List[float]] = {}
    
    def allow_request(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        current_time = time.time()
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.time_window
            ]
        else:
            self.requests[client_id] = []
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(current_time)
        return True


# Convenience functions for easy integration
def validate_content_request(request: ContentRequest) -> ValidationResult:
    """Convenience function to validate content request."""
    validator = InputValidator()
    return validator.validate_content_request(request)


def validate_and_sanitize_text(text: str, content_type: str = "general") -> ValidationResult:
    """Convenience function to validate and sanitize text."""
    validator = InputValidator()
    return validator.validate_text_content(text, content_type)


def filter_content(content: str) -> Tuple[str, List[str]]:
    """Convenience function to filter content."""
    content_filter = ContentFilter()
    return content_filter.filter_content(content)