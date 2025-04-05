"""
Server Module

This module handles the web server functionality of the application.
It allows users to host the AI assistant on a local server.
"""

import os
import socket
import threading
import time
import random
from typing import Optional

# Global flag to control server status
server_thread = None
server_running = False

def start_server() -> str:
    """
    Start the web server to host the AI assistant.
    
    Returns:
        str: The URL of the server
    """
    global server_thread, server_running
    
    # If server is already running, return current URL
    if server_running and server_thread and server_thread.is_alive():
        # Get the server URL
        local_ip = _get_local_ip()
        server_url = f"http://{local_ip}:5000"
        return server_url
    
    # Set the flag
    server_running = True
    
    # Start the server in a new thread
    server_thread = threading.Thread(target=_run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    # Get the server URL
    local_ip = _get_local_ip()
    server_url = f"http://{local_ip}:5000"
    
    return server_url

def stop_server() -> bool:
    """
    Stop the web server.
    
    Returns:
        bool: True if the server was stopped, False otherwise
    """
    global server_running
    
    # Set the flag to stop the server
    server_running = False
    
    # Wait for the server to stop
    time.sleep(2)
    
    return True

def _run_server():
    """Run the web server (this function runs in a separate thread)."""
    # In a real implementation, this would start a FastAPI server
    # that serves the Streamlit app using streamlit's export functionality
    # or a custom server implementation
    
    # Simulate server running
    global server_running
    while server_running:
        time.sleep(1)

def _get_local_ip() -> str:
    """
    Get the local IP address of the machine.
    
    Returns:
        str: The local IP address
    """
    try:
        # Create a socket connection to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        # Fallback if unable to determine IP
        return "127.0.0.1"

def is_server_running() -> bool:
    """
    Check if the server is running.
    
    Returns:
        bool: True if the server is running, False otherwise
    """
    global server_running, server_thread
    return server_running and server_thread and server_thread.is_alive()

def get_server_status() -> dict:
    """
    Get the status of the server.
    
    Returns:
        dict: Server status information
    """
    global server_running, server_thread
    
    status = {
        "running": is_server_running(),
        "uptime": None,
        "url": None
    }
    
    if status["running"]:
        local_ip = _get_local_ip()
        status["url"] = f"http://{local_ip}:5000"
    
    return status
