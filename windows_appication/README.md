# Sarah AI Assistant

A Streamlit-based AI assistant application with a chat interface, deep learning capabilities, data analysis, and multiple integration options.

## Features

- **Chat Interface**: Converse with Sarah, an AI assistant with a humorous and sarcastic (yet empathetic) tone
- **Multiple Simultaneous Chats**: Create and manage multiple conversations
- **File Upload**: Upload files through the chat interface for analysis
- **Deep Learning Capabilities**: Process natural language, generate text, and create images
- **Data Analysis**: Analyze and visualize data from various formats
- **PDF Processing**: Extract and summarize text from PDF documents
- **Web Server Option**: Host the application on a local web server
- **Customizable Settings**: Adjust app name, AI personality, themes, and performance levels
- **AI Backend Selection**: Choose between different AI backends (default, Ollama, Anthropic)
- **Custom Model Support**: Upload and use your own AI models
- **Real-time Resource Monitoring**: Track CPU, RAM, and GPU usage
- **Selective Model Training**: Train specific capabilities while preserving others
- **Integration Options**: Functionality for Google Calendar, Google Sheets, and Telegram integration

## Sarah AI Package

The `sarah_ai` package provides the core implementation of Sarah with the following components:

- **SarahModel**: Core AI model and personality with customizable settings
- **SarahTrainer**: Specialized training capabilities for different topics and skills
- **SarahUtils**: Utility functions for memory management, resource monitoring, and configuration

### Usage Example

```python
from sarah_ai import SarahModel

# Initialize Sarah with default settings
sarah = SarahModel()

# Generate a response
response = sarah.generate_response("Hello, Sarah!")

# Change personality settings
sarah.update_persona(name="Alice", tone="Professional")
```

## Directory Structure

```
/
├── app.py                 # Main Streamlit application
├── start_application.py   # Entry point for standalone applications
├── create_executable.py   # Script to create standalone executable
├── models/                # AI model implementations
│   ├── base_model.py      # Base model class
│   ├── model_handler.py   # Central model management
│   ├── text_generator.py  # Text generation capabilities
│   ├── image_generator.py # Image generation capabilities
│   ├── anthropic_model.py # Anthropic API integration
│   ├── ollama_model.py    # Ollama integration
│   └── model_trainer.py   # Model training utilities
├── utils/                 # Utility functions
│   ├── chat_handler.py    # Chat management
│   ├── pdf_processor.py   # PDF processing utilities
│   ├── data_analyzer.py   # Data analysis tools
│   ├── integrations.py    # External service integrations
│   ├── server.py          # Web server implementation
│   └── training.py        # Training utilities
├── sarah_ai/              # Sarah AI package
│   ├── __init__.py        # Package initialization
│   ├── sarah_model.py     # Core Sarah model implementation
│   ├── training.py        # Specialized training capabilities
│   └── utils.py           # Sarah-specific utilities
├── data/                  # Data storage
│   ├── knowledge_base.json      # Sarahs knowledge
│   └── default_settings.json    # Default settings
└── .streamlit/            # Streamlit configuration
    └── config.toml        # Streamlit settings
```

## Setup and Installation

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt` or let the application install them automatically
3. Run the application using one of the methods described below

### Running the Application

#### Method 1: Direct Execution (Terminal/Command Line)

```bash
# Using Streamlit directly
streamlit run app.py

# OR using the start application script
python start_application.py
```

#### Method 2: Using Executable Files (Recommended)

First, generate the executable files:

```bash
python create_executable.py
```

This will create:
- `Start App.exe` - Windows executable file (Windows only)
- `Start Application.bat` - Windows batch file
- `start_application.sh` - Linux/macOS shell script

Then, start the application:

- **Windows:** Double-click `Start App.exe` or `Start Application.bat`
- **Linux/macOS:** Run `./start_application.sh` from the terminal

#### What the Executables Do

The executable files automatically:
1. Check for required dependencies and install them if needed
2. Verify the application directory structure and configuration
3. Launch the Streamlit server with optimal settings
4. Open the application in your default web browser

## Configuration

Sarah AI can be configured using the Settings panel within the application, which includes:

- **General settings:** Application name, theme, accent color
- **AI persona settings:** Name, gender, tone
- **Performance settings:** Resource usage levels, real-time monitoring
- **AI Select:** Backend selection (default, Ollama, Anthropic), custom model configuration
- **Server settings:** Enable/disable web server for remote access

## License

MIT
