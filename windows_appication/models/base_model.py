"""
Base Model Module

This module defines the base class for all AI models used in the application.
"""

import os
import json
from typing import Dict, Any, List, Optional

class BaseModel:
    def __init__(self, performance_level: str = "balanced"):
        """
        Initialize the base model with the specified performance level.
        
        Args:
            performance_level: The performance level to use (low, balanced, high)
        """
        self.performance_level = performance_level
        self.model_size = self._get_model_size(performance_level)
    
    def _get_model_size(self, performance_level: str) -> str:
        """
        Determine the model size based on the performance level.
        
        Args:
            performance_level: The performance level (low, balanced, high)
            
        Returns:
            str: The model size (small, medium, large)
        """
        if performance_level == "high":
            return "large"
        elif performance_level == "balanced":
            return "medium"
        else:
            return "small"
    
    def initialize(self) -> bool:
        """
        Initialize the model.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        # Base implementation - should be overridden by subclasses
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            dict: Information about the model
        """
        # Base implementation - should be overridden by subclasses
        return {
            "performance_level": self.performance_level,
            "model_size": self.model_size
        }
    
    def save(self, path: str) -> bool:
        """
        Save the model to a file.
        
        Args:
            path: The path to save the model to
            
        Returns:
            bool: True if the save was successful, False otherwise
        """
        try:
            # Save basic model info
            with open(path, 'w') as f:
                json.dump(self.get_info(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the model from a file.
        
        Args:
            path: The path to load the model from
            
        Returns:
            bool: True if the load was successful, False otherwise
        """
        try:
            # Load basic model info
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Update model attributes
            if 'performance_level' in data:
                self.performance_level = data['performance_level']
                self.model_size = self._get_model_size(self.performance_level)
                
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False