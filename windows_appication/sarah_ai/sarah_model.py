"""
Sarah Model Module

This module provides the core implementation of Sarah's AI personality and capabilities.
It integrates multiple AI backends and provides a unified interface for interacting with
Sarah's advanced features.

Sarah is designed to be:
1. Personable and customizable in tone and responses
2. Capable of handling multiple types of requests
3. Adaptable to different performance levels
4. Extensible through plugins and integrations

Usage:
    from sarah_ai.sarah_model import SarahModel
    
    # Initialize Sarah with default settings
    sarah = SarahModel()
    
    # Generate a response
    response = sarah.generate_response("Hello, how are you today?")
    
    # Change personality settings
    sarah.update_persona(name="Alice", tone="Professional")
    
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional

# Import core model components
from models.base_model import BaseModel
from models.model_handler import ModelHandler
from utils.chat_handler import ChatHandler

class SarahModel:
    """
    Sarah AI Model - Core implementation of Sarah's AI personality and capabilities.
    
    This class provides high-level access to Sarah's features and manages
    the underlying model infrastructure.
    """
    
    def __init__(
        self, 
        name: str = "Sarah", 
        gender: str = "Female",
        tone: str = "Humorous and sarcastic",
        performance_level: str = "balanced",
        ai_backend: str = "default"
    ):
        """
        Initialize Sarah AI with the specified settings.
        
        Args:
            name: Name of the AI assistant
            gender: Gender identity (Female, Male, Non-binary)
            tone: Conversation tone/style
            performance_level: Resource usage level (low, balanced, high)
            ai_backend: AI provider to use (default, ollama, anthropic)
        """
        self.name = name
        self.gender = gender
        self.tone = tone
        self.performance_level = performance_level
        self.ai_backend = ai_backend
        
        # Initialize the model handler
        self.model_handler = ModelHandler(performance_level=performance_level, ai_backend=ai_backend)
        
        # Initialize the chat handler
        self.chat_handler = ChatHandler(self.model_handler)
        
        # Knowledge and memory
        self.knowledge_base = self._load_knowledge()
        self.conversation_history = []
        
        # Ollama specific settings
        self.ollama_settings = {
            "model": "llama3:8b",  # Default model
            "temperature": 0.7,
            "context_length": 2048,
            "top_p": 0.9
        }
        
        self.initialized = True
        
        print(f"{self.name} AI initialized with {self.performance_level} performance and {self.ai_backend} backend")
    
    def _load_knowledge(self) -> Dict[str, Any]:
        """
        Load Sarah's knowledge base from file.
        
        Returns:
            dict: The knowledge base
        """
        try:
            with open('data/knowledge_base.json', 'r') as f:
                return json.load(f)
        except:
            # Return default knowledge base if file not found
            return {
                "facts": [],
                "learned_concepts": {},
                "preferences": {
                    "humor_level": 0.8,
                    "formality_level": 0.3,
                    "verbosity_level": 0.6
                }
            }
    
    def _save_knowledge(self) -> bool:
        """
        Save Sarah's knowledge base to file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open('data/knowledge_base.json', 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving knowledge base: {str(e)}")
            return False
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to the user's input using Sarah's personality.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: Sarah's response
        """
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response
        response = self.model_handler.generate_response(
            user_input,
            ai_name=self.name,
            ai_tone=self.tone
        )
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def update_persona(self, name: Optional[str] = None, gender: Optional[str] = None, tone: Optional[str] = None) -> bool:
        """
        Update Sarah's persona settings.
        
        Args:
            name: New name for the AI (optional)
            gender: New gender setting (optional)
            tone: New conversation tone (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if name:
            self.name = name
        
        if gender:
            if gender in ["Female", "Male", "Non-binary"]:
                self.gender = gender
            else:
                print(f"Invalid gender: {gender}. Using current setting: {self.gender}")
        
        if tone:
            valid_tones = [
                "Humorous and sarcastic",
                "Professional",
                "Friendly",
                "Technical",
                "Empathetic"
            ]
            
            if tone in valid_tones:
                self.tone = tone
            else:
                print(f"Invalid tone: {tone}. Using current setting: {self.tone}")
        
        print(f"Persona updated: {self.name} ({self.gender}) with {self.tone} tone")
        return True
    
    def learn_concept(self, concept: str, information: Any) -> bool:
        """
        Add new information to Sarah's knowledge base.
        
        Args:
            concept: The concept or topic
            information: The information to learn
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.knowledge_base["learned_concepts"][concept] = information
            self._save_knowledge()
            return True
        except Exception as e:
            print(f"Error learning concept: {str(e)}")
            return False
    
    def forget_concept(self, concept: str) -> bool:
        """
        Remove a concept from Sarah's knowledge base.
        
        Args:
            concept: The concept to forget
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if concept in self.knowledge_base["learned_concepts"]:
                del self.knowledge_base["learned_concepts"][concept]
                self._save_knowledge()
                return True
            return False
        except Exception as e:
            print(f"Error forgetting concept: {str(e)}")
            return False
    
    def get_capabilities(self) -> List[str]:
        """
        Get a list of Sarah's capabilities.
        
        Returns:
            list: Sarah's capabilities
        """
        return [
            "Natural language conversation",
            "Text generation (stories, articles)",
            "Text summarization",
            "Image description generation",
            "Knowledge base management",
            "Sentiment analysis",
            "Keyword extraction",
            "Data analysis assistance"
        ]
    
    def summarize_text(self, text: str) -> str:
        """
        Summarize the provided text.
        
        Args:
            text: The text to summarize
            
        Returns:
            str: The summarized text
        """
        return self.model_handler.summarize_text(text)
    
    def generate_story(self, prompt: str) -> str:
        """
        Generate a story based on the provided prompt.
        
        Args:
            prompt: The story prompt
            
        Returns:
            str: The generated story
        """
        return self.model_handler.generate_story(prompt)
    
    def generate_article(self, topic: str, style: str = "general") -> str:
        """
        Generate an article on the specified topic.
        
        Args:
            topic: The article topic
            style: The writing style (general, academic, technical, blog)
            
        Returns:
            str: The generated article
        """
        return self.model_handler.generate_article(topic, style)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the provided text.
        
        Args:
            text: The text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        return self.model_handler.analyze_sentiment(text)
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from the provided text.
        
        Args:
            text: The text to analyze
            
        Returns:
            list: Extracted keywords
        """
        return self.model_handler.extract_keywords(text)
    
    def change_performance(self, new_level: str) -> bool:
        """
        Change Sarah's performance level.
        
        Args:
            new_level: The new performance level (low, balanced, high)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if new_level in ["low", "balanced", "high"]:
            self.performance_level = new_level
            
            # Reinitialize model handler with new performance level
            self.model_handler = ModelHandler(
                performance_level=new_level,
                ai_backend=self.ai_backend
            )
            
            # Update chat handler
            self.chat_handler = ChatHandler(self.model_handler)
            
            print(f"Performance level updated to {new_level}")
            return True
        else:
            print(f"Invalid performance level: {new_level}")
            return False
    
    def pull_ollama_model(self, model_name: str) -> bool:
        """
        Pull an Ollama model from the Ollama library.
        
        Args:
            model_name: The name of the model to pull
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import requests
            
            # Set the server URL (default Ollama port is 11434)
            server_url = "http://localhost:11434"
            
            # Make the request to pull the model
            response = requests.post(
                f"{server_url}/api/pull",
                json={"name": model_name}
            )
            
            # Check if request was successful
            if response.status_code == 200:
                return True
            else:
                print(f"Failed to pull model {model_name}: {response.text}")
                return False
        except Exception as e:
            print(f"Error pulling Ollama model: {str(e)}")
            return False
    
    def set_ollama_model(self, model_name: str) -> bool:
        """
        Set the active Ollama model.
        
        Args:
            model_name: The name of the model to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update the model settings
            self.ollama_settings = {
                "model": model_name,
                **self.ollama_settings
            }
            
            print(f"Ollama model set to {model_name}")
            return True
        except Exception as e:
            print(f"Error setting Ollama model: {str(e)}")
            return False
    
    def update_ollama_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Update Ollama model settings.
        
        Args:
            settings: Dictionary of settings to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update the settings
            self.ollama_settings.update(settings)
            
            print(f"Ollama settings updated: {settings}")
            return True
        except Exception as e:
            print(f"Error updating Ollama settings: {str(e)}")
            return False
    
    def change_backend(self, new_backend: str) -> bool:
        """
        Change Sarah's AI backend.
        
        Args:
            new_backend: The new AI backend (default, ollama, anthropic)
            
        Returns:
            bool: True if successful, False otherwise
        """
        valid_backends = ["default", "ollama", "anthropic"]
        
        if new_backend in valid_backends:
            self.ai_backend = new_backend
            
            # Reinitialize model handler with new backend
            self.model_handler = ModelHandler(
                performance_level=self.performance_level,
                ai_backend=new_backend
            )
            
            # Update chat handler
            self.chat_handler = ChatHandler(self.model_handler)
            
            print(f"AI backend updated to {new_backend}")
            return True
        else:
            print(f"Invalid AI backend: {new_backend}")
            return False
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Get Sarah's current settings.
        
        Returns:
            dict: Current settings
        """
        return {
            "name": self.name,
            "gender": self.gender,
            "tone": self.tone,
            "performance_level": self.performance_level,
            "ai_backend": self.ai_backend,
            "capabilities": self.get_capabilities()
        }