# Sarah AI Package

Core implementation of the Sarah AI assistant with customizable personality, multiple AI backends, and extensive capabilities.

## Components

- **SarahModel**: Core AI model with customizable personality settings
- **SarahTrainer**: Specialized training capabilities for different topics and skills
- **SarahUtils**: Utility functions for memory management, resource monitoring, and configuration

## Usage Example

```python
from sarah_ai import SarahModel

# Initialize Sarah with default settings
sarah = SarahModel()

# Generate a response
response = sarah.generate_response("Hello, Sarah!")

# Change personality settings
sarah.update_persona(name="Alice", tone="Professional")

# Generate a story
story = sarah.generate_story("A detective in a cyberpunk city")

# Analyze sentiment
sentiment = sarah.analyze_sentiment("This product is amazing!")
```

## SarahModel Features

- Customizable persona (name, gender, tone)
- Multiple AI backends (default, Ollama, Anthropic)
- Adjustable performance levels
- Knowledge base management
- Text generation capabilities
- Sentiment analysis and keyword extraction

## SarahTrainer Features

- Selective training for specific topics or capabilities
- Training history and evaluation metrics
- Integration with local datasets
- Integration with HuggingFace datasets
- Personality-preserving training

## SarahUtils Features

- Memory management for conversations and knowledge
- System resource monitoring (CPU, RAM, GPU)
- Configuration management
- Helper functions for data formatting

## Dependencies

- Python 3.6+
- psutil (for resource monitoring)
- Core model dependencies (based on selected backend)
