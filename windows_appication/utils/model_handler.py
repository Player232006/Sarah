"""
Model Handler Module

This module provides a unified interface to all AI models used in the application.
It controls model initialization, usage, and manages performance settings.
"""

import os
import time
import json
import random
from typing import Dict, List, Any, Optional, Union

# Import standard model modules
from models.nlp_model import NLPModel
from models.text_generator import TextGenerator
from models.image_generator import ImageGenerator

# Import additional AI models
try:
    from models.ollama_model import OllamaModel
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama module not available. Ollama features will be disabled.")

try:
    from models.anthropic_model import AnthropicModel
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Anthropic module not available. Claude features will be disabled.")

class ModelHandler:
    def __init__(self, performance_level: str = "balanced", ai_backend: str = "default"):
        """
        Initialize the model handler with the specified performance level and AI backend.
        
        Args:
            performance_level: The performance level to use for the models (low, balanced, high)
            ai_backend: The AI backend to use (default, ollama, anthropic)
        """
        self.performance_level = performance_level
        self.ai_backend = ai_backend
        self.models = {}
        self.knowledge_base = {}
        
        # Initialize models based on performance level and AI backend
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the appropriate models based on the performance level and AI backend."""
        # Default models (always loaded)
        # NLP model for conversation
        self.models["nlp"] = NLPModel(self.performance_level)
        
        # Text generator for stories, articles, etc.
        self.models["text_generator"] = TextGenerator(self.performance_level)
        
        # Image generator
        self.models["image_generator"] = ImageGenerator(self.performance_level)
        
        # Initialize AI backend models
        if self.ai_backend == "ollama" and OLLAMA_AVAILABLE:
            self.models["ollama"] = OllamaModel(self.performance_level)
            print(f"Initialized Ollama backend with performance level: {self.performance_level}")
        
        elif self.ai_backend == "anthropic" and ANTHROPIC_AVAILABLE:
            self.models["anthropic"] = AnthropicModel(self.performance_level)
            print(f"Initialized Anthropic Claude backend with performance level: {self.performance_level}")
        
        else:
            # Default built-in AI backend (fallback)
            print(f"Using default built-in AI backend with performance level: {self.performance_level}")
    
    def generate_text_response(self, user_message: str, ai_name: str = "Sarah", ai_tone: str = "Humorous and sarcastic") -> str:
        """
        Generate a text response to the user's message using the selected AI backend.
        
        Args:
            user_message: The message from the user
            ai_name: The name of the AI assistant
            ai_tone: The tone of the AI assistant
            
        Returns:
            str: The AI assistant's response
        """
        # Use the appropriate AI backend for response generation
        if self.ai_backend == "ollama" and "ollama" in self.models:
            return self.models["ollama"].generate_response(user_message, ai_name, ai_tone)
        
        elif self.ai_backend == "anthropic" and "anthropic" in self.models:
            return self.models["anthropic"].generate_response(user_message, ai_name, ai_tone)
        
        else:
            # Default to built-in NLP model
            return self.models["nlp"].generate_response(user_message, ai_name, ai_tone)
    
    def generate_story(self, prompt: str) -> str:
        """
        Generate a story based on the prompt.
        
        Args:
            prompt: The prompt for the story
            
        Returns:
            str: The generated story
        """
        return self.models["text_generator"].generate_story(prompt)
    
    def generate_image(self, description: str) -> str:
        """
        Generate an image based on the description.
        
        Args:
            description: The description of the image to generate
            
        Returns:
            str: Information about the generated image
        """
        return self.models["image_generator"].generate_image(description)
    
    def summarize_text(self, text: str) -> str:
        """
        Summarize the given text.
        
        Args:
            text: The text to summarize
            
        Returns:
            str: The summarized text
        """
        # Use the text generator model for summarization
        return self.models["text_generator"].summarize(text)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            dict: The sentiment analysis results
        """
        return self.models["nlp"].analyze_sentiment(text)
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from the given text.
        
        Args:
            text: The text to extract keywords from
            
        Returns:
            list: The extracted keywords
        """
        return self.models["nlp"].extract_keywords(text)
    
    def switch_performance_level(self, new_level: str):
        """
        Switch the performance level of the models.
        
        Args:
            new_level: The new performance level (low, balanced, high)
        """
        if new_level not in ["low", "balanced", "high"]:
            raise ValueError("Performance level must be one of: low, balanced, high")
        
        # Only switch if the level is actually changing
        if new_level != self.performance_level:
            self.performance_level = new_level
            
            # Reinitialize models with the new performance level
            self._initialize_models()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models.
        
        Returns:
            dict: Information about the loaded models
        """
        info = {
            "performance_level": self.performance_level,
            "ai_backend": self.ai_backend,
            "models": {}
        }
        
        for model_name, model in self.models.items():
            info["models"][model_name] = {
                "type": model.__class__.__name__,
                "submodels": model.get_info()
            }
        
        return info
        
    def get_available_backends(self) -> List[Dict[str, Any]]:
        """
        Get a list of available AI backends.
        
        Returns:
            list: List of available AI backends with their details
        """
        backends = [
            {
                "id": "default",
                "name": "Built-in AI (Sarah)",
                "description": "The default built-in AI model for general purpose use",
                "available": True,
                "enabled": self.ai_backend == "default"
            }
        ]
        
        # Add Ollama if available
        backends.append({
            "id": "ollama",
            "name": "Ollama",
            "description": "Open source, locally-run LLMs",
            "available": OLLAMA_AVAILABLE,
            "enabled": self.ai_backend == "ollama" and OLLAMA_AVAILABLE
        })
        
        # Add Anthropic Claude if available
        backends.append({
            "id": "anthropic",
            "name": "Anthropic Claude",
            "description": "Advanced AI assistant with natural language capabilities (requires API key)",
            "available": ANTHROPIC_AVAILABLE,
            "enabled": self.ai_backend == "anthropic" and ANTHROPIC_AVAILABLE
        })
        
        return backends
        
    def change_ai_backend(self, new_backend: str) -> bool:
        """
        Change the AI backend used for response generation.
        
        Args:
            new_backend: The new AI backend to use (default, ollama, anthropic)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate the backend
        if new_backend not in ["default", "ollama", "anthropic"]:
            print(f"Invalid AI backend: {new_backend}")
            return False
            
        # Check availability
        if new_backend == "ollama" and not OLLAMA_AVAILABLE:
            print("Ollama backend is not available")
            return False
            
        if new_backend == "anthropic" and not ANTHROPIC_AVAILABLE:
            print("Anthropic Claude backend is not available")
            return False
        
        # Only switch if the backend is actually changing
        if new_backend != self.ai_backend:
            self.ai_backend = new_backend
            
            # Reinitialize models with the new backend
            self._initialize_models()
            
        return True
