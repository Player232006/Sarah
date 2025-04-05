"""
Sarah AI - Advanced AI Assistant Package

This package provides the core implementation of Sarah, an advanced AI assistant
with customizable personality, multiple AI backends, and extensive capabilities.

Main Components:
- SarahModel: Core AI model and personality
- SarahTrainer: Specialized training capabilities
- SarahUtils: Utility functions for working with Sarah

Example Usage:
    from sarah_ai import SarahModel
    
    # Initialize Sarah
    sarah = SarahModel()
    
    # Generate a response
    response = sarah.generate_response("Hello, Sarah!")
"""

# Version information
__version__ = "1.0.0"
__author__ = "Sarah AI Development Team"
__license__ = "MIT"

# Import main components for easier access
from sarah_ai.sarah_model import SarahModel
from sarah_ai.training import SarahTrainer

# Easy access to the main class
Sarah = SarahModel