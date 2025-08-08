"""
Resilience and monitoring utilities for production-ready operations.

This module provides:
- Retry logic with exponential backoff
- Timeout handling
- Circuit breaker patterns
- Performance monitoring
- Failure tracking and alerting
"""

import asyncio
import time
import logging
import functools
import threading
from typing import Callable, Any, Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import random
import json
import structlog


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff" 
    LINEAR_BACKOFF = "linear_backoff"
    RANDOM_JITTER = "random_jitter"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter_max: float = 0.1
    exceptions: Tuple[Exception, ...] = (Exception,)
    timeout: Optional[float] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    monitor_period: float = 300.0  # 5 minutes


@dataclass 
class OperationMetrics:
    """Metrics for operation monitoring."""
    operation_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = False
    attempt_count: int = 1
    error_message: Optional[str] = None
    error_type: Optional[str] = None


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.metrics: Dict[str, List[OperationMetrics]] = {}
        self.lock = threading.Lock()
        self.logger = structlog.get_logger("performance")
    
    def record_operation(self, operation_name: str, start_time: datetime,
                        end_time: datetime, success: bool, 
                        attempt_count: int = 1, error: Exception = None) -> None:
        """Record operation metrics."""
        duration = (end_time - start_time).total_seconds()
        
        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success=success,
            attempt_count=attempt_count,
            error_message=str(error) if error else None,
            error_type=type(error).__name__ if error else None
        )
        
        with self.lock:
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(metrics)
            
            # Keep only last 1000 metrics per operation
            if len(self.metrics[operation_name]) > 1000:
                self.metrics[operation_name] = self.metrics[operation_name][-1000:]
        
        # Log performance metrics
        self.logger.info(
            "operation_completed",
            operation=operation_name,
            duration=duration,
            success=success,
            attempt_count=attempt_count,
            error_type=metrics.error_type,
            timestamp=end_time.isoformat()
        )
    
    def get_operation_stats(self, operation_name: str, 
                          time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get statistics for an operation within time window."""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self.lock:
            if operation_name not in self.metrics:
                return {"error": "Operation not found"}
            
            recent_metrics = [
                m for m in self.metrics[operation_name]
                if m.start_time >= cutoff_time
            ]
        
        if not recent_metrics:
            return {"error": "No recent metrics found"}
        
        successful = [m for m in recent_metrics if m.success]
        failed = [m for m in recent_metrics if not m.success]
        
        durations = [m.duration for m in recent_metrics if m.duration is not None]
        
        stats = {
            "operation": operation_name,
            "time_window_minutes": time_window_minutes,
            "total_operations": len(recent_metrics),
            "successful_operations": len(successful),
            "failed_operations": len(failed),
            "success_rate": len(successful) / len(recent_metrics) if recent_metrics else 0,
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "average_attempts": sum(m.attempt_count for m in recent_metrics) / len(recent_metrics),
            "error_types": {}
        }
        
        # Count error types
        for metric in failed:
            if metric.error_type:
                if metric.error_type not in stats["error_types"]:
                    stats["error_types"][metric.error_type] = 0
                stats["error_types"][metric.error_type] += 1
        
        return stats


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.lock = threading.Lock()
        self.logger = structlog.get_logger("circuit_breaker")
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        with self.lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    self.logger.info(
                        "circuit_breaker_half_open", 
                        name=self.name,
                        state=self.state.value
                    )
                    return True
                return False
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return True
            
        return False
    
    def record_success(self) -> None:
        """Record successful operation."""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.logger.info(
                        "circuit_breaker_closed",
                        name=self.name,
                        state=self.state.value,
                        success_count=self.success_count
                    )
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record failed operation."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.logger.warning(
                        "circuit_breaker_opened",
                        name=self.name,
                        state=self.state.value,
                        failure_count=self.failure_count
                    )
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(
                    "circuit_breaker_reopened",
                    name=self.name,
                    state=self.state.value
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout


class ResilienceManager:
    """Central manager for resilience features."""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = structlog.get_logger("resilience")
    
    def get_circuit_breaker(self, name: str, 
                           config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(config, name)
        return self.circuit_breakers[name]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        health = {
            "timestamp": datetime.now().isoformat(),
            "circuit_breakers": {},
            "performance_stats": {}
        }
        
        # Circuit breaker statuses
        for name, cb in self.circuit_breakers.items():
            health["circuit_breakers"][name] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count
            }
        
        # Performance statistics
        for operation_name in self.performance_monitor.metrics.keys():
            stats = self.performance_monitor.get_operation_stats(operation_name, 60)
            if "error" not in stats:
                health["performance_stats"][operation_name] = stats
        
        return health


# Global resilience manager instance
resilience_manager = ResilienceManager()


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """Decorator for retry logic with configurable backoff strategy."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__qualname__}"
            circuit_breaker = resilience_manager.get_circuit_breaker(operation_name)
            
            last_exception = None
            start_time = datetime.now()
            
            for attempt in range(1, config.max_attempts + 1):
                # Check circuit breaker
                if not circuit_breaker.can_execute():
                    error_msg = f"Circuit breaker open for {operation_name}"
                    resilience_manager.logger.warning(
                        "circuit_breaker_blocked",
                        operation=operation_name,
                        attempt=attempt
                    )
                    raise Exception(error_msg)
                
                try:
                    # Apply timeout if configured
                    if config.timeout:
                        result = await asyncio.wait_for(
                            func(*args, **kwargs), 
                            timeout=config.timeout
                        )
                    else:
                        result = await func(*args, **kwargs)
                    
                    # Success
                    circuit_breaker.record_success()
                    end_time = datetime.now()
                    resilience_manager.performance_monitor.record_operation(
                        operation_name, start_time, end_time, True, attempt
                    )
                    
                    if attempt > 1:
                        resilience_manager.logger.info(
                            "retry_success",
                            operation=operation_name,
                            attempt=attempt,
                            total_attempts=config.max_attempts
                        )
                    
                    return result
                    
                except config.exceptions as e:
                    last_exception = e
                    circuit_breaker.record_failure()
                    
                    resilience_manager.logger.warning(
                        "operation_failed",
                        operation=operation_name,
                        attempt=attempt,
                        total_attempts=config.max_attempts,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    
                    # Don't wait on last attempt
                    if attempt < config.max_attempts:
                        delay = _calculate_delay(config, attempt)
                        await asyncio.sleep(delay)
                    
                except Exception as e:
                    # Non-retryable exception
                    circuit_breaker.record_failure()
                    end_time = datetime.now()
                    resilience_manager.performance_monitor.record_operation(
                        operation_name, start_time, end_time, False, attempt, e
                    )
                    raise
            
            # All attempts failed
            end_time = datetime.now()
            resilience_manager.performance_monitor.record_operation(
                operation_name, start_time, end_time, False, config.max_attempts, last_exception
            )
            
            raise last_exception or Exception(f"All {config.max_attempts} attempts failed")
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, create a simple retry wrapper
            operation_name = f"{func.__module__}.{func.__qualname__}"
            last_exception = None
            start_time = datetime.now()
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    end_time = datetime.now()
                    resilience_manager.performance_monitor.record_operation(
                        operation_name, start_time, end_time, True, attempt
                    )
                    
                    return result
                    
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts:
                        delay = _calculate_delay(config, attempt)
                        time.sleep(delay)
                
                except Exception as e:
                    end_time = datetime.now()
                    resilience_manager.performance_monitor.record_operation(
                        operation_name, start_time, end_time, False, attempt, e
                    )
                    raise
            
            # All attempts failed
            end_time = datetime.now()
            resilience_manager.performance_monitor.record_operation(
                operation_name, start_time, end_time, False, config.max_attempts, last_exception
            )
            
            raise last_exception or Exception(f"All {config.max_attempts} attempts failed")
        
        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def _calculate_delay(config: RetryConfig, attempt: int) -> float:
    """Calculate delay based on retry strategy."""
    if config.strategy == RetryStrategy.FIXED_DELAY:
        delay = config.base_delay
    
    elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        delay = config.base_delay * (config.backoff_multiplier ** (attempt - 1))
    
    elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = config.base_delay * attempt
    
    elif config.strategy == RetryStrategy.RANDOM_JITTER:
        base_delay = config.base_delay * (config.backoff_multiplier ** (attempt - 1))
        jitter = random.uniform(0, config.jitter_max)
        delay = base_delay + jitter
    
    else:
        delay = config.base_delay
    
    return min(delay, config.max_delay)


def timeout(seconds: float):
    """Decorator for adding timeout to async functions."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                operation_name = f"{func.__module__}.{func.__qualname__}"
                resilience_manager.logger.error(
                    "operation_timeout",
                    operation=operation_name,
                    timeout_seconds=seconds
                )
                raise
        return wrapper
    return decorator


def circuit_breaker(name: Optional[str] = None, 
                   config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker pattern."""
    def decorator(func):
        breaker_name = name or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cb = resilience_manager.get_circuit_breaker(breaker_name, config)
            
            if not cb.can_execute():
                error_msg = f"Circuit breaker open for {breaker_name}"
                resilience_manager.logger.warning("circuit_breaker_blocked", name=breaker_name)
                raise Exception(error_msg)
            
            try:
                result = await func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure()
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cb = resilience_manager.get_circuit_breaker(breaker_name, config)
            
            if not cb.can_execute():
                error_msg = f"Circuit breaker open for {breaker_name}"
                raise Exception(error_msg)
            
            try:
                result = func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure()
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Convenience functions for monitoring
def get_performance_stats(operation_name: str, time_window_minutes: int = 60) -> Dict[str, Any]:
    """Get performance statistics for an operation."""
    return resilience_manager.performance_monitor.get_operation_stats(
        operation_name, time_window_minutes
    )


def get_system_health() -> Dict[str, Any]:
    """Get overall system health."""
    return resilience_manager.get_system_health()


def reset_circuit_breaker(name: str) -> bool:
    """Reset a circuit breaker to closed state."""
    if name in resilience_manager.circuit_breakers:
        cb = resilience_manager.circuit_breakers[name]
        with cb.lock:
            cb.state = CircuitBreakerState.CLOSED
            cb.failure_count = 0
            cb.success_count = 0
            cb.last_failure_time = None
        resilience_manager.logger.info("circuit_breaker_reset", name=name)
        return True
    return False