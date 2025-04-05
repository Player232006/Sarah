"""
Text Generator Module

This module provides text generation capabilities for the application.
"""

import random
import time
from typing import Dict, List, Any, Optional

from models.base_model import BaseModel

class TextGenerator(BaseModel):
    def __init__(self, performance_level: str = "balanced"):
        """
        Initialize the text generator with the specified performance level.
        
        Args:
            performance_level: The performance level to use (low, balanced, high)
        """
        super().__init__(performance_level)
        
        # Set model-specific parameters based on performance level
        if performance_level == "high":
            self.max_tokens = 4096
            self.creativity = 0.8
        elif performance_level == "balanced":
            self.max_tokens = 2048
            self.creativity = 0.7
        else:
            self.max_tokens = 1024
            self.creativity = 0.6
    
    def generate_story(self, prompt: str) -> str:
        """
        Generate a story based on the prompt.
        
        Args:
            prompt: The prompt for the story
            
        Returns:
            str: The generated story
        """
        # Simple story generation based on the prompt
        story_intros = [
            f"Once upon a time, in a world where {prompt}, there lived a curious adventurer.",
            f"In the year 2150, {prompt} had become a reality that changed everything.",
            f"The legend of {prompt} began on a stormy night in the ancient kingdom.",
            f"Nobody believed in {prompt} until that fateful day when everything changed.",
            f"The secret society dedicated to {prompt} had existed for centuries in the shadows."
        ]
        
        story_middles = [
            "The journey would not be easy, but determination drove every step forward.",
            "Surprising allies appeared along the way, each with their own motives and secrets.",
            "Each challenge seemed more impossible than the last, testing resolve and ingenuity.",
            "What started as a simple quest soon revealed layers of complexity and mystery.",
            "Ancient knowledge and modern ingenuity combined to overcome seemingly impossible odds."
        ]
        
        story_endings = [
            "In the end, the true meaning of the adventure wasn't what was found, but what was learned along the way.",
            "Victory came at a price, but the world would never be the same again.",
            "Some questions remained unanswered, setting the stage for future adventures.",
            "The journey's end revealed that this was just the beginning of a much larger story.",
            "What seemed like an ending was actually a new beginning, with endless possibilities ahead."
        ]
        
        # Build the story based on performance level
        if self.performance_level == "high":
            paragraphs = 5
        elif self.performance_level == "balanced":
            paragraphs = 3
        else:
            paragraphs = 2
        
        story = random.choice(story_intros) + "\n\n"
        
        for _ in range(paragraphs):
            story += self._generate_paragraph(prompt) + "\n\n"
        
        story += random.choice(story_middles) + "\n\n"
        
        for _ in range(paragraphs):
            story += self._generate_paragraph(prompt) + "\n\n"
        
        story += random.choice(story_endings)
        
        return story
    
    def _generate_paragraph(self, topic: str) -> str:
        """
        Generate a paragraph related to the topic.
        
        Args:
            topic: The topic to generate a paragraph about
            
        Returns:
            str: The generated paragraph
        """
        # Generate a simple paragraph with 3-5 sentences
        num_sentences = random.randint(3, 5)
        
        sentence_templates = [
            f"The implications of {topic} were far-reaching and unexpected.",
            f"Nobody could have predicted how {topic} would change everything.",
            f"Experts debated the significance of {topic} for years to come.",
            f"The discovery related to {topic} opened new possibilities.",
            f"Many had spent their lives searching for answers about {topic}.",
            f"The ancient texts had mentioned {topic} but in a different context.",
            f"Technology transformed how people approached {topic}.",
            f"The community was divided in their opinions about {topic}.",
            f"Historical records contained subtle references to {topic}.",
            f"The connection between {topic} and the forgotten legends became clearer."
        ]
        
        # Select random sentences to form a paragraph
        sentences = random.sample(sentence_templates, min(num_sentences, len(sentence_templates)))
        return " ".join(sentences)
    
    def summarize(self, text: str) -> str:
        """
        Summarize the given text.
        
        Args:
            text: The text to summarize
            
        Returns:
            str: The summarized text
        """
        # Simple text summarization (extract key sentences)
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        
        if not sentences:
            return "The provided text is empty or contains no valid sentences."
        
        # Determine number of sentences in summary based on performance level
        if self.performance_level == "high":
            num_sentences = min(5, len(sentences))
        elif self.performance_level == "balanced":
            num_sentences = min(3, len(sentences))
        else:
            num_sentences = min(2, len(sentences))
        
        # Simple heuristic: take first sentence and some from the middle/end
        summary_sentences = [sentences[0]]  # Always include the first sentence
        
        if len(sentences) > 1 and num_sentences > 1:
            # Add some sentences from the middle and end
            indices = [
                len(sentences) // 3,
                len(sentences) // 2,
                len(sentences) - 1
            ]
            
            # Add sentences without duplicates until we reach num_sentences
            for idx in indices:
                if len(summary_sentences) < num_sentences and sentences[idx] not in summary_sentences:
                    summary_sentences.append(sentences[idx])
        
        # Create the summary
        summary = ". ".join(summary_sentences)
        if not summary.endswith("."):
            summary += "."
            
        return summary
    
    def generate_article(self, topic: str, style: str = "general") -> str:
        """
        Generate an article on the specified topic in the given style.
        
        Args:
            topic: The topic of the article
            style: The style of the article (general, academic, technical, blog)
            
        Returns:
            str: The generated article
        """
        # Customize content based on style
        if style == "academic":
            intro = f"# {topic.title()}: A Scholarly Analysis\n\n"
            intro += f"This article examines the critical aspects of {topic} through an academic lens, providing insight into its theoretical foundations and practical implications.\n\n"
        elif style == "technical":
            intro = f"# Technical Overview: {topic.title()}\n\n"
            intro += f"This technical document explains the inner workings of {topic}, including its architecture, implementation details, and best practices.\n\n"
        elif style == "blog":
            intro = f"# {topic.title()} - What You Need to Know\n\n"
            intro += f"Hey there! Today we're diving into {topic} - one of the most interesting subjects I've come across lately.\n\n"
        else:  # general
            intro = f"# Understanding {topic.title()}\n\n"
            intro += f"This article provides a comprehensive overview of {topic}, exploring its key aspects and significance.\n\n"
        
        # Generate sections based on performance level
        if self.performance_level == "high":
            num_sections = 4
            paragraphs_per_section = 3
        elif self.performance_level == "balanced":
            num_sections = 3
            paragraphs_per_section = 2
        else:
            num_sections = 2
            paragraphs_per_section = 1
            
        # Section topics
        section_templates = [
            f"## Background of {topic}\n\n",
            f"## Key Features of {topic}\n\n",
            f"## Applications of {topic}\n\n",
            f"## Future Developments in {topic}\n\n",
            f"## Challenges with {topic}\n\n",
            f"## History of {topic}\n\n"
        ]
        
        # Build the article
        article = intro
        
        # Add sections
        for i in range(min(num_sections, len(section_templates))):
            article += section_templates[i]
            
            for _ in range(paragraphs_per_section):
                article += self._generate_paragraph(topic) + "\n\n"
        
        # Add conclusion
        article += f"## Conclusion\n\n"
        article += f"In summary, {topic} represents a fascinating area with numerous implications. "
        article += f"As we have explored in this article, its significance extends across multiple domains. "
        article += f"Understanding {topic} more deeply can provide valuable insights and applications for various fields."
        
        return article
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the text generator.
        
        Returns:
            dict: Information about the text generator
        """
        info = super().get_info()
        info.update({
            "type": "Built-in Text Generator",
            "max_tokens": self.max_tokens,
            "creativity": self.creativity,
            "capabilities": ["story generation", "article generation", "text summarization"]
        })
        return info