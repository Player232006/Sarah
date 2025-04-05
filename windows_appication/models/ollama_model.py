"""
Ollama Model Module

This module provides integration with Ollama models for more human-like responses.
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional

# Import Ollama if available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama module not available. Using mock implementation.")

# Import base model
from models.base_model import BaseModel

class OllamaModel(BaseModel):
    def __init__(self, performance_level: str = "balanced"):
        """
        Initialize the Ollama model with the specified performance level.
        
        Args:
            performance_level: The performance level to use (low, balanced, high)
        """
        super().__init__(performance_level)
        self.model_name = self._get_model_name(performance_level)
        self.server_url = "http://localhost:11434"
        self.is_initialized = False
        self.initialized = False  # Add this attribute for compatibility with model_handler.py
        
        # Initialize the model
        self.initialize()
    
    def _get_model_name(self, performance_level: str) -> str:
        """
        Determine the Ollama model to use based on performance level
        
        Args:
            performance_level: The performance level (low, balanced, high)
            
        Returns:
            str: The name of the Ollama model to use
        """
        if performance_level == "high":
            return "llama3:70b"  # High performance, larger model
        elif performance_level == "balanced":
            return "llama3:8b"   # Balanced performance
        else:
            return "llama3:latest"  # Low performance, smaller model
    
    def initialize(self) -> bool:
        """
        Initialize the Ollama model client.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if not OLLAMA_AVAILABLE:
            print("Ollama module not available. Using mock implementation.")
            return False
        
        try:
            # Check if Ollama server is running
            if not self._check_server_status():
                print("Ollama server is not running. Please start it to use Ollama models.")
                return False
            
            # Ensure the model exists
            if not self._ensure_model_exists():
                print(f"Failed to pull the Ollama model: {self.model_name}")
                return False
            
            self.is_initialized = True
            self.initialized = True  # Update both flags
            print(f"Ollama model initialized: {self.model_name}")
            return True
            
        except Exception as e:
            print(f"Error initializing Ollama model: {str(e)}")
            return False
    
    def _check_server_status(self) -> bool:
        """
        Check if the Ollama server is running.
        
        Returns:
            bool: True if the server is running, False otherwise
        """
        try:
            # In a real implementation, we would check the server status
            # For now, this is a mock implementation
            return True
        except:
            return False
    
    def _ensure_model_exists(self) -> bool:
        """
        Ensure the selected model exists on the Ollama server.
        
        Returns:
            bool: True if the model exists or was pulled, False otherwise
        """
        try:
            # In a real implementation, we would check if the model exists
            # and pull it if it doesn't
            # For now, assume it exists
            return True
        except:
            return False
    
    def generate_response(self, prompt: str, ai_name: str = "Ollama AI", ai_tone: str = "Helpful") -> str:
        """
        Generate a response using the Ollama model.
        
        Args:
            prompt: The prompt to generate a response for
            ai_name: Name of the AI to use in the system prompt
            ai_tone: The tone to use for responses
            
        Returns:
            str: The generated response
        """
        if not self.is_initialized:
            return f"Sorry, Ollama is not properly initialized. Please check that the Ollama server is running."
        
        try:
            # Create a system prompt with the AI name and tone
            system_prompt = f"You are {ai_name}, an AI assistant. Your responses should be {ai_tone}."
            
            if not OLLAMA_AVAILABLE:
                # Mock response for testing
                return f"[MOCK OLLAMA RESPONSE] I am {ai_name} and I'm responding in a {ai_tone} tone. You asked: {prompt}"
            
            # In a real implementation, we would call the Ollama API
            # For now, return a mock response
            return f"[MOCK OLLAMA RESPONSE] I am {ai_name} and I'm responding in a {ai_tone} tone. You asked: {prompt}"
                
        except Exception as e:
            print(f"Error generating response with Ollama: {str(e)}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the Ollama model.
        
        Returns:
            dict: Information about the model
        """
        return {
            "name": "Ollama",
            "model": self.model_name,
            "server_url": self.server_url,
            "initialized": self.is_initialized,
            "ollama_available": OLLAMA_AVAILABLE
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models on the Ollama server.
        
        Returns:
            list: List of available models with their details
        """
        if not self.is_initialized:
            return []
        
        try:
            # In a real implementation, we would call the Ollama API
            # For now, return a mock list
            return [
                {"name": "llama3:latest", "size": "3GB", "description": "Latest Llama 3 model"},
                {"name": "llama3:8b", "size": "5GB", "description": "Medium-sized Llama 3 model"},
                {"name": "llama3:70b", "size": "40GB", "description": "Large Llama 3 model"},
                {"name": "mistral:latest", "size": "4GB", "description": "Latest Mistral model"},
                {"name": "phi3:latest", "size": "2GB", "description": "Latest Phi-3 model"}
            ]
        except:
            return []
    
    def change_model(self, model_name: str) -> bool:
        """
        Change the current Ollama model.
        
        Args:
            model_name: The name of the model to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            return False
        
        try:
            # In a real implementation, we would check if the model exists
            # and pull it if it doesn't
            self.model_name = model_name
            print(f"Changed Ollama model to: {model_name}")
            return True
        except:
            return False
    
    def summarize_text(self, text: str) -> str:
        """
        Summarize text using Ollama model.
        
        Args:
            text: The text to summarize
            
        Returns:
            str: The summarized text
        """
        if not self.is_initialized:
            return "Sorry, Ollama is not properly initialized."
        
        try:
            # Create a summarization prompt
            prompt = f"Please summarize the following text concisely:\n\n{text[:5000]}..."
            
            if not OLLAMA_AVAILABLE:
                # Mock response for testing
                return f"[MOCK OLLAMA SUMMARY] This is a brief summary of the text you provided."
            
            # In a real implementation, we would call the Ollama API
            # For now, return a mock response
            return f"[MOCK OLLAMA SUMMARY] This is a brief summary of the text you provided."
                
        except Exception as e:
            print(f"Error summarizing text with Ollama: {str(e)}")
            return f"Sorry, I encountered an error while summarizing: {str(e)}"
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of text using Ollama model.
        
        Args:
            text: The text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not self.is_initialized:
            return {"error": "Ollama is not properly initialized."}
        
        try:
            # Create a sentiment analysis prompt
            prompt = f"""Analyze the sentiment of the following text:
            {text[:1000]}
            
            Provide a JSON with sentiment analysis.
            """
            
            if not OLLAMA_AVAILABLE:
                # Mock response for testing
                return {
                    "sentiment": "positive",
                    "score": 0.75,
                    "key_phrases": ["mock phrase 1", "mock phrase 2"],
                    "mood": "happy"
                }
            
            # In a real implementation, we would call the Ollama API
            # For now, return a mock response
            return {
                "sentiment": "positive",
                "score": 0.75,
                "key_phrases": ["mock phrase 1", "mock phrase 2"],
                "mood": "happy"
            }
                
        except Exception as e:
            print(f"Error analyzing sentiment with Ollama: {str(e)}")
            return {"sentiment": "unknown", "error": str(e)}