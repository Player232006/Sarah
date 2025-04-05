#!/usr/bin/env python3
import subprocess
import sys
import os

# Set the current directory to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Start the application
subprocess.run([sys.executable, "start_application.py"])
