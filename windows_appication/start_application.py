#!/usr/bin/env python3
"""
Start Application

This script launches the AI Assistant application and performs basic error checking
before starting the Streamlit server.
"""

import os
import sys
import traceback
import importlib
import time
import subprocess
import json
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit', 
        'plotly', 
        'pypdf2',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")
    
    if missing_packages:
        print("\nMissing packages detected. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("All required packages have been installed.")
        except Exception as e:
            print(f"Error installing packages: {str(e)}")
            return False
    
    return True

def check_directory_structure():
    """Check if all required directories and files exist."""
    required_dirs = [
        'data',
        'models',
        'utils'
    ]
    
    required_files = [
        'app.py',
        'data/default_settings.json'
    ]
    
    # Check directories
    for directory in required_dirs:
        if not os.path.isdir(directory):
            print(f"✗ Directory '{directory}' does not exist")
            return False
        print(f"✓ Directory '{directory}' exists")
    
    # Check files
    for file in required_files:
        if not os.path.isfile(file):
            print(f"✗ File '{file}' does not exist")
            return False
        print(f"✓ File '{file}' exists")
    
    return True

def check_settings_file():
    """Check if the settings file is valid JSON."""
    settings_file = 'data/default_settings.json'
    try:
        with open(settings_file, 'r') as f:
            json.load(f)
        print(f"✓ Settings file '{settings_file}' is valid JSON")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ Settings file '{settings_file}' is not valid JSON: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Error reading settings file '{settings_file}': {str(e)}")
        return False

def check_syntax():
    """Check Python syntax in main application files."""
    main_modules = [
        'app.py',
        'utils/chat_handler.py',
        'utils/model_handler.py',
        'utils/data_analyzer.py',
        'utils/pdf_processor.py',
        'models/base_model.py',
        'models/nlp_model.py',
        'models/text_generator.py',
        'models/image_generator.py'
    ]
    
    syntax_errors = []
    
    for module in main_modules:
        try:
            with open(module, 'r') as f:
                compile(f.read(), module, 'exec')
            print(f"✓ {module} has valid syntax")
        except SyntaxError as e:
            syntax_errors.append((module, e))
            print(f"✗ Syntax error in {module}: {str(e)}")
    
    return len(syntax_errors) == 0

def start_application():
    """Start the Streamlit application."""
    try:
        print("\nStarting AI Assistant application...\n")
        # Create .streamlit directory and config.toml if they don't exist
        streamlit_dir = Path('.streamlit')
        streamlit_dir.mkdir(exist_ok=True)
        
        config_file = streamlit_dir / 'config.toml'
        if not config_file.exists():
            with open(config_file, 'w') as f:
                f.write("""[server]
headless = true
address = "0.0.0.0"
port = 5000
""")
            print("✓ Created Streamlit configuration file")
        
        # Start the application
        print("\nAI Assistant is starting. Please wait...\n")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "5000"])
        
        return True
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("AI Assistant Application Launcher")
    print("=" * 60)
    
    # Perform checks
    print("\n1. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n2. Checking directory structure...")
    dirs_ok = check_directory_structure()
    
    print("\n3. Checking settings file...")
    settings_ok = check_settings_file()
    
    print("\n4. Checking Python syntax...")
    syntax_ok = check_syntax()
    
    # Start application if all checks pass
    if deps_ok and dirs_ok and settings_ok and syntax_ok:
        print("\nAll checks passed successfully!")
        start_application()
    else:
        print("\n❌ Some checks failed. Please fix the errors before starting the application.")
        input("Press Enter to exit...")