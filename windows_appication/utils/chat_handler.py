"""
Chat Handler Module

This module manages chat interactions for the AI assistant.
"""

import time
import json
import random
from typing import Dict, List, Any, Optional

class ChatHandler:
    def __init__(self, model_handler):
        """
        Initialize the chat handler with the specified model handler.
        
        Args:
            model_handler: The model handler to use for generating responses
        """
        self.model_handler = model_handler
        self.conversation_history = []
        self.max_history = 20  # Maximum number of messages to keep in history
    
    def generate_response(self, user_message: str, ai_name: str = "Sarah", ai_tone: str = "Humorous and sarcastic") -> str:
        """
        Generate a response to the user's message.
        
        Args:
            user_message: The message from the user
            ai_name: The name of the AI assistant
            ai_tone: The tone of the AI assistant
            
        Returns:
            str: The AI assistant's response
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Generate response using the model handler
        response = self.model_handler.generate_response(user_message, ai_name, ai_tone)
        
        # Add AI response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Trim history if it exceeds max_history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        return response
    
    def save_history(self, path: str) -> bool:
        """
        Save the conversation history to a file.
        
        Args:
            path: The path to save the history to
            
        Returns:
            bool: True if the save was successful, False otherwise
        """
        try:
            with open(path, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving conversation history: {str(e)}")
            return False
    
    def load_history(self, path: str) -> bool:
        """
        Load conversation history from a file.
        
        Args:
            path: The path to load the history from
            
        Returns:
            bool: True if the load was successful, False otherwise
        """
        try:
            with open(path, 'r') as f:
                self.conversation_history = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading conversation history: {str(e)}")
            return False
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            list: The conversation history
        """
        return self.conversation_history.copy()