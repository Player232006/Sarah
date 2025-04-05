"""
Anthropic Claude Model Module

This module provides integration with Anthropic's Claude models for high-quality, human-like AI responses.
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional
import anthropic
from anthropic import Anthropic

# Import base model
from models.base_model import BaseModel

class AnthropicModel(BaseModel):
    def __init__(self, performance_level: str = "balanced"):
        """
        Initialize the Anthropic Claude model with the specified performance level.
        
        Args:
            performance_level: The performance level to use (low, balanced, high)
        """
        super().__init__(performance_level)
        self.client = None
        self.model = self._get_model_name(performance_level)
        self.max_tokens = 4096  # Default max tokens for Claude responses
        self.initialized = False  # Add this attribute for compatibility with model_handler.py
        
        # Initialize the client
        self.initialize()
    
    def _get_model_name(self, performance_level: str) -> str:
        """
        Get the appropriate Claude model based on performance level.
        
        Args:
            performance_level: Performance level (low, balanced, high)
            
        Returns:
            str: The name of the Claude model to use
        """
        # The newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        if performance_level == "high":
            return "claude-3-5-sonnet-20241022"  # Best quality, slower
        elif performance_level == "balanced":
            return "claude-3-opus-20240229"  # Balanced performance
        else:
            return "claude-3-haiku-20240307"  # Faster, simpler responses
    
    def initialize(self) -> bool:
        """
        Initialize the Anthropic API client.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Get API key from environment
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            
            if not api_key:
                print("ANTHROPIC_API_KEY environment variable not set. Claude features will be limited.")
                return False
            
            # Initialize the client
            self.client = Anthropic(api_key=api_key)
            self.initialized = True  # Set initialized flag
            print(f"Anthropic Claude model initialized: {self.model}")
            return True
            
        except Exception as e:
            print(f"Error initializing Anthropic Claude model: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, ai_name: str = "Claude", ai_tone: str = "Helpful") -> str:
        """
        Generate a response using the Anthropic Claude model.
        
        Args:
            prompt: The prompt to generate a response for
            ai_name: Name of the AI to use in the system prompt
            ai_tone: The tone to use for responses
            
        Returns:
            str: The generated response
        """
        if not self.client:
            return f"Sorry, Anthropic Claude API is not properly configured. Please check your API key."
        
        try:
            # Create a system prompt with the AI name and tone
            system_prompt = f"You are {ai_name}, an AI assistant. Your responses should be {ai_tone}."
            
            # Call the API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and return the response text
            if hasattr(response, 'content') and len(response.content) > 0:
                # Get text from the first content block (assuming it's text)
                for content_block in response.content:
                    if content_block.type == 'text':
                        return content_block.text
                
                # Fallback if no text content found
                return "I processed your request, but couldn't generate a text response."
            else:
                return "Sorry, I wasn't able to generate a response."
                
        except Exception as e:
            print(f"Error generating response with Anthropic Claude: {str(e)}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the Anthropic model.
        
        Returns:
            dict: Information about the model
        """
        return {
            "name": "Anthropic Claude",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "initialized": self.client is not None,
            "api_key_available": os.environ.get('ANTHROPIC_API_KEY') is not None
        }
    
    def summarize_text(self, text: str) -> str:
        """
        Summarize text using Anthropic Claude.
        
        Args:
            text: The text to summarize
            
        Returns:
            str: The summarized text
        """
        if not self.client:
            return "Sorry, Anthropic Claude API is not properly configured."
        
        try:
            # Create a summarization prompt
            prompt = f"Please summarize the following text concisely:\n\n{text[:10000]}..."
            
            # Call the API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,  # Shorter for summaries
                system="You are a professional text summarizer. Create concise but comprehensive summaries.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and return the response text
            if hasattr(response, 'content') and len(response.content) > 0:
                for content_block in response.content:
                    if content_block.type == 'text':
                        return content_block.text
                
                return "I processed your text, but couldn't generate a summary."
            else:
                return "Sorry, I wasn't able to generate a summary."
                
        except Exception as e:
            print(f"Error summarizing text with Anthropic Claude: {str(e)}")
            return f"Sorry, I encountered an error while summarizing: {str(e)}"
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of text using Anthropic Claude.
        
        Args:
            text: The text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not self.client:
            return {"error": "Anthropic Claude API is not properly configured."}
        
        try:
            # Create a sentiment analysis prompt
            prompt = f"""Analyze the sentiment of the following text. Provide a JSON response with the following fields:
            1. sentiment: overall sentiment (positive, negative, or neutral)
            2. score: a score from -1.0 (very negative) to 1.0 (very positive)
            3. key_phrases: a list of key sentiment phrases from the text
            4. mood: the emotional tone of the text
            
            Text to analyze:
            {text[:5000]}
            
            Format your response as valid JSON.
            """
            
            # Call the API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system="You are a sentiment analysis tool. Output only valid JSON.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and parse the JSON response
            if hasattr(response, 'content') and len(response.content) > 0:
                for content_block in response.content:
                    if content_block.type == 'text':
                        # Try to extract JSON from the response
                        response_text = content_block.text
                        
                        # Find JSON content (between curly braces)
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}') + 1
                        
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = response_text[start_idx:end_idx]
                            try:
                                return json.loads(json_str)
                            except:
                                # If parsing fails, return the raw text
                                return {"sentiment": "unknown", "error": "Failed to parse JSON response", "raw_response": response_text}
                        
                        return {"sentiment": "unknown", "error": "No JSON found in response", "raw_response": response_text}
                
                return {"sentiment": "unknown", "error": "No text content in response"}
            else:
                return {"sentiment": "unknown", "error": "Empty response from API"}
                
        except Exception as e:
            print(f"Error analyzing sentiment with Anthropic Claude: {str(e)}")
            return {"sentiment": "unknown", "error": str(e)}