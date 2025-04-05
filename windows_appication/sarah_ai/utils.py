"""
Sarah AI Utilities

This module provides utility functions and helper classes for working with Sarah.
These utilities support Sarah's core functionality and help manage its resources.

Includes:
- Memory management
- Resource monitoring
- Data conversion
- Configuration handling
"""

import os
import sys
import json
import psutil
from typing import Dict, List, Any, Optional
from datetime import datetime

class SarahMemoryManager:
    """
    Manages Sarah's memory including conversation history and knowledge.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize the memory manager.
        
        Args:
            max_history: Maximum number of conversation entries to store
        """
        self.max_history = max_history
        self.conversation_history = []
        self.short_term_memory = {}
        self.working_memory = {}
    
    def add_conversation(self, role: str, content: str) -> None:
        """
        Add a conversation entry to history.
        
        Args:
            role: The role (user or assistant)
            content: The message content
        """
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().timestamp()
        }
        
        self.conversation_history.append(entry)
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            limit: Optional limit on number of entries to return
            
        Returns:
            list: Conversation history entries
        """
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history
    
    def remember(self, key: str, value: Any) -> None:
        """
        Store a value in short-term memory.
        
        Args:
            key: The memory key
            value: The value to remember
        """
        self.short_term_memory[key] = {
            "value": value,
            "timestamp": datetime.now().timestamp()
        }
    
    def recall(self, key: str) -> Any:
        """
        Recall a value from short-term memory.
        
        Args:
            key: The memory key
            
        Returns:
            Any: The remembered value, or None if not found
        """
        if key in self.short_term_memory:
            return self.short_term_memory[key]["value"]
        return None
    
    def forget(self, key: str) -> bool:
        """
        Remove a value from short-term memory.
        
        Args:
            key: The memory key
            
        Returns:
            bool: True if the key was found and removed, False otherwise
        """
        if key in self.short_term_memory:
            del self.short_term_memory[key]
            return True
        return False
    
    def clear_memory(self, memory_type: str = "all") -> None:
        """
        Clear memory of specified type.
        
        Args:
            memory_type: Type of memory to clear (all, conversation, short_term, working)
        """
        if memory_type in ["all", "conversation"]:
            self.conversation_history = []
        
        if memory_type in ["all", "short_term"]:
            self.short_term_memory = {}
        
        if memory_type in ["all", "working"]:
            self.working_memory = {}


class SarahResourceMonitor:
    """
    Monitors system resources for Sarah to adjust performance.
    Supports auto-refresh functionality for real-time monitoring.
    """
    
    def __init__(self, check_gpu: bool = True, auto_refresh_interval: int = 5):
        """
        Initialize the resource monitor.
        
        Args:
            check_gpu: Whether to check GPU usage if available
            auto_refresh_interval: Interval in seconds for auto-refresh (0 to disable)
        """
        self.check_gpu = check_gpu
        self.auto_refresh_interval = auto_refresh_interval
        self.auto_refresh_enabled = auto_refresh_interval > 0
        self.last_check = None
        self.last_refresh_time = 0
        self.history = []
    
    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.
        
        Returns:
            float: CPU usage percentage (0-100)
        """
        # Call with interval to ensure we get a proper reading
        # First call to cpu_percent returns 0.0, so we use a small interval
        # to get an immediate reading
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # If we somehow still get 0.0, make a second attempt
        if cpu_percent == 0.0:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
        return cpu_percent
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage.
        
        Returns:
            dict: Memory usage details
        """
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "percent": mem.percent
        }
    
    def get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """
        Get GPU usage if available.
        
        Returns:
            dict: GPU usage details, or None if not available
        """
        # Placeholder for GPU monitoring logic
        # In a real implementation, we would use libraries like py3nvml
        return None
    
    def get_all_resources(self) -> Dict[str, Any]:
        """
        Get all resource usage metrics.
        
        Returns:
            dict: Resource usage details
        """
        current_time = datetime.now().timestamp()
        cpu = self.get_cpu_usage()
        memory = self.get_memory_usage()
        gpu = self.get_gpu_usage() if self.check_gpu else None
        
        result = {
            "timestamp": current_time,
            "cpu": cpu,
            "memory": memory
        }
        
        if gpu:
            result["gpu"] = gpu
        
        self.last_check = result
        self.last_refresh_time = current_time
        self.history.append(result)
        
        # Keep only the last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        return result
        
    def should_refresh(self) -> bool:
        """
        Check if it's time to refresh resource metrics based on auto-refresh interval.
        
        Returns:
            bool: True if refresh is needed, False otherwise
        """
        if not self.auto_refresh_enabled:
            return False
            
        current_time = datetime.now().timestamp()
        return (current_time - self.last_refresh_time) >= self.auto_refresh_interval
    
    def recommend_performance_level(self) -> str:
        """
        Recommend a performance level based on available resources.
        
        Returns:
            str: Recommended performance level (low, balanced, high)
        """
        resources = self.get_all_resources()
        
        # Simple heuristic based on memory and CPU
        if resources["memory"]["percent"] > 80 or resources["cpu"] > 80:
            return "low"
        elif resources["memory"]["percent"] > 50 or resources["cpu"] > 50:
            return "balanced"
        else:
            return "high"


class SarahConfig:
    """
    Manages Sarah's configuration settings.
    """
    
    def __init__(self, config_path: str = "data/sarah_config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            dict: Configuration settings
        """
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except:
            # Return default configuration if file not found
            return {
                "name": "Sarah",
                "gender": "Female",
                "tone": "Humorous and sarcastic",
                "performance_level": "balanced",
                "ai_backend": "default",
                "max_history": 100,
                "auto_training": False,
                "enable_resource_monitoring": True
            }
    
    def _save_config(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key
            default: Default value if key not found
            
        Returns:
            Any: The configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key
            value: The value to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.config[key] = value
        return self._save_config()
    
    def update(self, updates: Dict[str, Any]) -> bool:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.config.update(updates)
        return self._save_config()
    
    def reset(self) -> bool:
        """
        Reset configuration to defaults.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.config = {
            "name": "Sarah",
            "gender": "Female",
            "tone": "Humorous and sarcastic",
            "performance_level": "balanced",
            "ai_backend": "default",
            "max_history": 100,
            "auto_training": False,
            "enable_resource_monitoring": True
        }
        return self._save_config()


# Helper functions

def format_time(timestamp: float) -> str:
    """
    Format a timestamp as a human-readable string.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        str: Formatted time string
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def readable_size(size_bytes: float) -> str:
    """
    Convert bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Human-readable size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i = i + 1  # Just increment by 1
    
    return f"{size_bytes:.2f}{size_names[int(i)]}"

def get_sarah_info() -> Dict[str, Any]:
    """
    Get information about Sarah's implementation.
    
    Returns:
        dict: Information about Sarah
    """
    return {
        "name": "Sarah AI",
        "version": "1.0.0",
        "description": "Advanced AI assistant with customizable personality",
        "author": "Sarah AI Development Team",
        "license": "MIT",
        "capabilities": [
            "Natural language conversation",
            "Text generation",
            "Text summarization",
            "Image description generation",
            "Knowledge base management",
            "Sentiment analysis",
            "Keyword extraction",
            "Data analysis assistance"
        ],
        "homepage": "https://sarah-ai.example.com",
        "documentation": "https://docs.sarah-ai.example.com"
    }