"""
NLP Model Module

This module provides natural language processing capabilities for the application.
"""

import random
import time
from typing import Dict, List, Any, Optional

from models.base_model import BaseModel

class NLPModel(BaseModel):
    def __init__(self, performance_level: str = "balanced"):
        """
        Initialize the NLP model with the specified performance level.
        
        Args:
            performance_level: The performance level to use (low, balanced, high)
        """
        super().__init__(performance_level)
        
        # Set model-specific parameters based on performance level
        if performance_level == "high":
            self.max_tokens = 2048
            self.temperature = 0.7
        elif performance_level == "balanced":
            self.max_tokens = 1024
            self.temperature = 0.8
        else:
            self.max_tokens = 512
            self.temperature = 0.9
    
    def generate_response(self, prompt: str, ai_name: str = "Sarah", ai_tone: str = "Humorous and sarcastic") -> str:
        """
        Generate a response to the prompt.
        
        Args:
            prompt: The prompt to generate a response for
            ai_name: The name of the AI assistant
            ai_tone: The tone of the AI assistant
            
        Returns:
            str: The generated response
        """
        # Simple response generation based on the prompt
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            return f"Hi there! I'm {ai_name}. How can I help you today?"
        
        elif "how are you" in prompt.lower():
            return f"I'm doing great, thanks for asking! As {ai_name}, I'm always ready to assist with a {ai_tone} approach."
        
        elif "who are you" in prompt.lower() or "what are you" in prompt.lower():
            return f"I'm {ai_name}, your AI assistant. I'm designed to be {ai_tone} while helping you with various tasks."
        
        elif "help" in prompt.lower():
            return f"I'd be happy to help! I can assist with conversations, generate stories, analyze data, and more. What would you like to do?"
        
        elif "thank" in prompt.lower():
            return f"You're welcome! It's my pleasure to assist. Let me know if you need anything else."
        
        else:
            # Generate a simple response acknowledging the input
            responses = [
                f"I understand you're asking about '{prompt}'. Let me think about that...",
                f"That's an interesting question about '{prompt}'. As {ai_name}, I'd say...",
                f"Hmm, let me process that request about '{prompt}' in my {ai_tone} way...",
                f"I see you're interested in '{prompt}'. Let me provide some insights...",
                f"As {ai_name}, I find your question about '{prompt}' intriguing. Here's my {ai_tone} take on it..."
            ]
            return random.choice(responses)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            dict: The sentiment analysis results
        """
        # Simple keyword-based sentiment analysis
        positive_words = ["good", "great", "excellent", "awesome", "happy", "love", "best", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "worst", "hate", "dislike", "poor", "horrible", "disappointed"]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        # Determine overall sentiment
        if positive_count > negative_count:
            sentiment = "positive"
            score = min(1.0, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = max(-1.0, -0.5 - (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            score = 0.0
        
        # Extract key phrases (simple implementation)
        words = [word.strip(".,!?;:()[]{}\"'") for word in text.lower().split()]
        key_phrases = []
        
        for i in range(len(words)):
            if words[i] in positive_words or words[i] in negative_words:
                phrase = ' '.join(words[max(0, i-2):min(len(words), i+3)])
                key_phrases.append(phrase)
        
        return {
            "sentiment": sentiment,
            "score": score,
            "key_phrases": key_phrases[:5],  # Limit to 5 phrases
            "mood": self._get_mood(score)
        }
    
    def _get_mood(self, score: float) -> str:
        """
        Determine the mood based on the sentiment score.
        
        Args:
            score: The sentiment score (-1.0 to 1.0)
            
        Returns:
            str: The mood description
        """
        if score > 0.7:
            return "very happy"
        elif score > 0.3:
            return "happy"
        elif score > 0.0:
            return "slightly positive"
        elif score > -0.3:
            return "slightly negative"
        elif score > -0.7:
            return "sad"
        else:
            return "very upset"
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from the text.
        
        Args:
            text: The text to extract keywords from
            
        Returns:
            list: The extracted keywords
        """
        # Simple keyword extraction (most frequent words excluding common stopwords)
        stopwords = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "about", "of", "is", "are", "was", "were"]
        words = [word.strip(".,!?;:()[]{}\"'").lower() for word in text.split()]
        
        # Filter out stopwords and short words
        filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:10]]  # Return top 10 keywords
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the NLP model.
        
        Returns:
            dict: Information about the model
        """
        info = super().get_info()
        info.update({
            "type": "Built-in NLP",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "capabilities": ["conversation", "sentiment analysis", "keyword extraction"]
        })
        return info