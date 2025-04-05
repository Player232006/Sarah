#!/bin/bash
echo "Starting AI Assistant Application..."
python3 start_application.py
if [ $? -ne 0 ]; then
    echo ""
    echo "Error occurred while running the application."
    echo "Please check the error message above."
    read -p "Press Enter to exit..."
    exit 1
fi
