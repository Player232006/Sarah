#!/usr/bin/env python3
"""
Create Executable Script

This script creates a Windows executable (.exe), batch file (.bat), and Linux shell script (.sh)
that will act like an "executable" to start the AI Assistant application.
"""

import os
import platform
import stat
import sys
import subprocess
import shutil
from pathlib import Path

def create_batch_file():
    """Create a Windows batch file to start the application."""
    batch_content = """@echo off
echo Starting AI Assistant Application...
python start_application.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error occurred while running the application.
    echo Please check the error message above.
    pause
    exit /b %ERRORLEVEL%
)
"""
    
    with open("Start Application.bat", "w") as f:
        f.write(batch_content)
    
    print("✓ Created 'Start Application.bat'")

def create_shell_script():
    """Create a Linux/macOS shell script to start the application."""
    shell_content = """#!/bin/bash
echo "Starting AI Assistant Application..."
python3 start_application.py
if [ $? -ne 0 ]; then
    echo ""
    echo "Error occurred while running the application."
    echo "Please check the error message above."
    read -p "Press Enter to exit..."
    exit 1
fi
"""
    
    with open("start_application.sh", "w") as f:
        f.write(shell_content)
    
    # Make the shell script executable
    st = os.stat("start_application.sh")
    os.chmod("start_application.sh", st.st_mode | stat.S_IEXEC)
    
    print("✓ Created 'start_application.sh' (executable)")

def create_windows_executable():
    """Create a Windows executable (.exe) using PyInstaller."""
    try:
        # Check if PyInstaller is installed
        import importlib
        try:
            importlib.import_module('PyInstaller')
            print("✓ PyInstaller is already installed")
        except ImportError:
            print("Installing PyInstaller...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("✓ PyInstaller has been installed")
        
        # Create a small launcher script
        launcher_script = "exe_launcher.py"
        with open(launcher_script, "w") as f:
            f.write("""#!/usr/bin/env python3
import subprocess
import sys
import os

# Set the current directory to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Start the application
subprocess.run([sys.executable, "start_application.py"])
""")
        
        # Create the executable
        print("Creating Windows executable with PyInstaller...")
        cmd = [
            sys.executable, 
            "-m", "PyInstaller",
            "--onefile",
            "--console",
            "--name", "Start App"
        ]
        
        # Add icon if it exists
        if os.path.exists("generated-icon.png"):
            cmd.extend(["--icon", "generated-icon.png"])
            
        # Add launcher script
        cmd.append(launcher_script)
        
        # Run PyInstaller
        subprocess.run(cmd, check=True)
        
        # Copy the executable to the main directory
        dist_path = Path("dist")
        if dist_path.exists():
            for exe_file in dist_path.glob("*.exe"):
                shutil.copy(exe_file, ".")
                print(f"✓ Created '{exe_file.name}'")
                
        # Clean up
        if os.path.exists(launcher_script):
            os.remove(launcher_script)
        if os.path.exists("build"):
            shutil.rmtree("build")
        if os.path.exists("dist"):
            shutil.rmtree("dist")
        if os.path.exists("Start App.spec"):
            os.remove("Start App.spec")
            
        return True
    except Exception as e:
        print(f"Error creating Windows executable: {str(e)}")
        print("You can still use the batch file or shell script to start the application.")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Creating executable files for Sarah AI Assistant")
    print("=" * 60)
    
    # Create platform-appropriate starter files
    system = platform.system()
    
    # Create batch and shell scripts on all platforms
    create_batch_file()
    create_shell_script()
    
    # Create Windows executable on any platform
    print("\nAttempting to create Windows executable (this can work on non-Windows platforms)")
    create_windows_executable()
    
    print("\nExecutable files created successfully!")
    print("\nTo start the application:")
    print("- On Windows: Double-click 'Start App.exe' or 'Start Application.bat'")
    print("- On Linux/macOS: Run './start_application.sh'")
    print("\nThese files will perform checks and start the Sarah AI Assistant.")