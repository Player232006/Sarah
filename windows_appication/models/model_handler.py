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

# Import model modules
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
            performance_level: The performance level to use (low, balanced, high)
            ai_backend: The AI backend to use (default, ollama, anthropic)
        """
        self.performance_level = performance_level
        self.ai_backend = ai_backend
        
        # Initialize standard models
        self.nlp_model = NLPModel(performance_level)
        self.text_generator = TextGenerator(performance_level)
        self.image_generator = ImageGenerator(performance_level)
        
        # Initialize optional advanced models based on backend selection
        self.ollama_model = None
        self.anthropic_model = None
        
        if ai_backend == "ollama" and OLLAMA_AVAILABLE:
            self.ollama_model = OllamaModel(performance_level)
            # Ollama initialization will be done on first use to avoid startup delays
        
        if ai_backend == "anthropic" and ANTHROPIC_AVAILABLE:
            self.anthropic_model = AnthropicModel(performance_level)
            # Anthropic initialization will be done on first use to avoid startup delays
        
        # Initialize standard models
        self.nlp_model.initialize()
        self.text_generator.initialize()
        self.image_generator.initialize()
        
        # Knowledge base to store training data and user patterns
        self.knowledge_base = {
            "initialized_at": time.time(),
            "performance_level": performance_level,
            "ai_backend": ai_backend,
            "training_examples": 0,
            "user_interactions": 0,
            "common_topics": {},
            "last_trained": None
        }
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Try to load existing knowledge
        self._load_knowledge()
    
    def _load_knowledge(self) -> bool:
        """
        Load knowledge base from file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            knowledge_path = "data/knowledge_base.json"
            if os.path.exists(knowledge_path):
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    loaded_knowledge = json.load(f)
                    self.knowledge_base.update(loaded_knowledge)
                return True
            return False
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            return False
    
    def _save_knowledge(self) -> bool:
        """
        Save knowledge base to file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            knowledge_path = "data/knowledge_base.json"
            with open(knowledge_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving knowledge base: {str(e)}")
            return False
    
    def update_knowledge(self, new_data: Dict[str, Any]) -> bool:
        """
        Update the knowledge base with new data.
        
        Args:
            new_data: New knowledge data to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.knowledge_base.update(new_data)
            
            # Update common topics
            if "topic" in new_data:
                topic = new_data["topic"]
                if topic in self.knowledge_base["common_topics"]:
                    self.knowledge_base["common_topics"][topic] += 1
                else:
                    self.knowledge_base["common_topics"][topic] = 1
            
            # Save knowledge base
            return self._save_knowledge()
        except Exception as e:
            print(f"Error updating knowledge base: {str(e)}")
            return False
    
    def reset_knowledge(self) -> bool:
        """
        Reset the knowledge base to default values.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.knowledge_base = {
                "initialized_at": time.time(),
                "performance_level": self.performance_level,
                "training_examples": 0,
                "user_interactions": 0,
                "common_topics": {},
                "last_trained": None
            }
            return self._save_knowledge()
        except Exception as e:
            print(f"Error resetting knowledge base: {str(e)}")
            return False
    
    def generate_response(self, user_input: str, ai_name: str = "Sarah", ai_tone: str = "Humorous and sarcastic") -> str:
        """
        Generate a response to the user's input using the selected AI backend.
        
        Args:
            user_input: The user's input text
            ai_name: The name of the AI assistant
            ai_tone: The tone of the AI assistant
            
        Returns:
            str: The generated response
        """
        # Update user interactions count
        self.knowledge_base["user_interactions"] += 1
        self._save_knowledge()
        
        # Use the appropriate model based on the selected backend
        if self.ai_backend == "ollama" and self.ollama_model:
            try:
                # Initialize on first use
                if not self.ollama_model.initialized:
                    self.ollama_model.initialize()
                    
                return self.ollama_model.generate_response(user_input, ai_name, ai_tone)
            except Exception as e:
                print(f"Error with Ollama model: {str(e)}")
                print("Falling back to default NLP model")
                return f"Error using Ollama: {str(e)}\n\nFalling back to standard response: " + self.nlp_model.generate_response(user_input, ai_name, ai_tone)
                
        elif self.ai_backend == "anthropic" and self.anthropic_model:
            try:
                # Initialize on first use
                if not self.anthropic_model.initialized:
                    self.anthropic_model.initialize()
                    
                return self.anthropic_model.generate_response(user_input, ai_name, ai_tone)
            except Exception as e:
                print(f"Error with Anthropic model: {str(e)}")
                print("Falling back to default NLP model")
                return f"Error using Claude: {str(e)}\n\nFalling back to standard response: " + self.nlp_model.generate_response(user_input, ai_name, ai_tone)
        
        # Default to the NLP model
        return self.nlp_model.generate_response(user_input, ai_name, ai_tone)
    
    def generate_story(self, prompt: str) -> str:
        """
        Generate a story based on the prompt.
        
        Args:
            prompt: The prompt for the story
            
        Returns:
            str: The generated story
        """
        return self.text_generator.generate_story(prompt)
    
    def generate_article(self, topic: str, style: str = "general") -> str:
        """
        Generate an article on the specified topic.
        
        Args:
            topic: The topic of the article
            style: The style of the article
            
        Returns:
            str: The generated article
        """
        return self.text_generator.generate_article(topic, style)
    
    def summarize_text(self, text: str) -> str:
        """
        Summarize the given text.
        
        Args:
            text: The text to summarize
            
        Returns:
            str: The summarized text
        """
        return self.text_generator.summarize(text)
    
    def generate_image_description(self, description: str) -> str:
        """
        Generate an image description based on the text description.
        
        Args:
            description: The description of the image to generate
            
        Returns:
            str: Information about the generated image
        """
        return self.image_generator.generate_image(description)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            dict: The sentiment analysis results
        """
        return self.nlp_model.analyze_sentiment(text)
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from the given text.
        
        Args:
            text: The text to extract keywords from
            
        Returns:
            list: The extracted keywords
        """
        return self.nlp_model.extract_keywords(text)
    
    def get_models_info(self) -> Dict[str, Any]:
        """
        Get information about all initialized models.
        
        Returns:
            dict: Model information
        """
        return {
            "nlp_model": self.nlp_model.get_info(),
            "text_generator": self.text_generator.get_info(),
            "image_generator": self.image_generator.get_info(),
            "performance_level": self.performance_level,
            "knowledge_base": {
                "training_examples": self.knowledge_base.get("training_examples", 0),
                "user_interactions": self.knowledge_base.get("user_interactions", 0),
                "last_trained": self.knowledge_base.get("last_trained", None),
                "common_topics_count": len(self.knowledge_base.get("common_topics", {}))
            }
        }
    
    def change_performance_level(self, new_level: str) -> bool:
        """
        Change the performance level of all models.
        
        Args:
            new_level: The new performance level (low, balanced, high)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if new_level not in ["low", "balanced", "high"]:
            return False
        
        try:
            self.performance_level = new_level
            
            # Update models
            self.nlp_model = NLPModel(new_level)
            self.text_generator = TextGenerator(new_level)
            self.image_generator = ImageGenerator(new_level)
            
            # Re-initialize models
            self.nlp_model.initialize()
            self.text_generator.initialize()
            self.image_generator.initialize()
            
            # Update backend-specific models if active
            if self.ai_backend == "ollama" and OLLAMA_AVAILABLE:
                self.ollama_model = OllamaModel(new_level)
                
            if self.ai_backend == "anthropic" and ANTHROPIC_AVAILABLE:
                self.anthropic_model = AnthropicModel(new_level)
            
            # Update knowledge base
            self.knowledge_base["performance_level"] = new_level
            self._save_knowledge()
            
            return True
        except Exception as e:
            print(f"Error changing performance level: {str(e)}")
            return False
    
    def test_ollama_connection(self, server_url: str = "http://localhost:11434") -> bool:
        """
        Test connection to an Ollama server.
        
        Args:
            server_url: The URL of the Ollama server
            
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            import requests
            response = requests.get(f"{server_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Error connecting to Ollama server: {str(e)}")
            return False
    
    def get_ollama_models(self, server_url: str = "http://localhost:11434") -> list:
        """
        Get list of available models from an Ollama server.
        
        Args:
            server_url: The URL of the Ollama server
            
        Returns:
            list: List of model information dictionaries
        """
        try:
            import requests
            response = requests.get(f"{server_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            return []
        except Exception as e:
            print(f"Error getting Ollama models: {str(e)}")
            return []
            
    def change_ai_backend(self, new_backend: str) -> bool:
        """
        Change the AI backend used for response generation.
        
        Args:
            new_backend: The new AI backend to use (default, ollama, anthropic)
            
        Returns:
            bool: True if successful, False otherwise
        """
        valid_backends = ["default", "ollama", "anthropic"]
        if new_backend not in valid_backends:
            return False
            
        # Check if requested backend is available
        if new_backend == "ollama" and not OLLAMA_AVAILABLE:
            print("Ollama backend requested but not available")
            return False
            
        if new_backend == "anthropic" and not ANTHROPIC_AVAILABLE:
            print("Anthropic backend requested but not available")
            return False
        
        try:
            self.ai_backend = new_backend
            
            # Initialize the appropriate backend model
            if new_backend == "ollama" and OLLAMA_AVAILABLE:
                self.ollama_model = OllamaModel(self.performance_level)
                # Lazy initialization - will be initialized on first use
                
            if new_backend == "anthropic" and ANTHROPIC_AVAILABLE:
                self.anthropic_model = AnthropicModel(self.performance_level)
                # Lazy initialization - will be initialized on first use
            
            # Update knowledge base
            self.knowledge_base["ai_backend"] = new_backend
            self._save_knowledge()
            
            return True
        except Exception as e:
            print(f"Error changing AI backend: {str(e)}")
            return False
            
    def get_available_backends(self) -> List[Dict[str, Any]]:
        """
        Get a list of available AI backends.
        
        Returns:
            list: List of available AI backends with their info
        """
        backends = [
            {
                "id": "default",
                "name": "Default (Sarah)",
                "description": "The standard rule-based AI assistant",
                "available": True,
                "requires_api_key": False
            }
        ]
        
        # Add Ollama if available
        if OLLAMA_AVAILABLE:
            backends.append({
                "id": "ollama",
                "name": "Ollama",
                "description": "Open-source, locally-run AI models (requires Ollama server)",
                "available": True,
                "requires_api_key": False,
                "server_required": True
            })
        
        # Add Anthropic if available
        if ANTHROPIC_AVAILABLE:
            has_api_key = bool(os.environ.get('ANTHROPIC_API_KEY'))
            backends.append({
                "id": "anthropic",
                "name": "Anthropic Claude",
                "description": "High-quality AI responses with Claude API",
                "available": True,
                "requires_api_key": True,
                "has_api_key": has_api_key
            })
            
        return backends