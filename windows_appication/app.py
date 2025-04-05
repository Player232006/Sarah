import streamlit as st
import os
import json
import uuid
import time
from datetime import datetime

# Import utility modules
from utils.chat_handler import ChatHandler
from models.model_handler import ModelHandler  # Fixed import path
from utils.data_analyzer import DataAnalyzer
from utils.pdf_processor import PDFProcessor
from utils.integrations import GoogleCalendar, GoogleSheets, TelegramBot
from utils.server import start_server, stop_server
from models.model_trainer import ModelTrainer
from utils.huggingface_trainer import HuggingFaceTrainer
import psutil  # For system monitoring

# Import Sarah AI package
from sarah_ai import SarahModel, SarahTrainer
from sarah_ai.utils import SarahResourceMonitor, SarahConfig, format_time, readable_size

# Set page config
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'chats' not in st.session_state:
    st.session_state.chats = {
        "default": {"messages": [], "title": "New Chat", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    }
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = "default"
if 'settings' not in st.session_state:
    # Load default settings from file
    try:
        with open('data/default_settings.json', 'r') as f:
            st.session_state.settings = json.load(f)
    except:
        # Default settings if file not found
        st.session_state.settings = {
            "app_name": "AI Assistant",
            "ai_name": "Sarah",
            "ai_gender": "Female",
            "ai_tone": "Humorous and sarcastic",
            "theme": "dark",
            "accent_color": "#FF4B4B",
            "performance": "balanced",  # Options: low, balanced, high
            "ai_backend": "default",    # Options: default, ollama, anthropic
            "server_active": False,
            "server_url": ""
        }
if 'model_handler' not in st.session_state:
    st.session_state.model_handler = ModelHandler(
        st.session_state.settings["performance"],
        st.session_state.settings.get("ai_backend", "default")
    )
if 'chat_handler' not in st.session_state:
    st.session_state.chat_handler = ChatHandler(st.session_state.model_handler)
if 'data_analyzer' not in st.session_state:
    st.session_state.data_analyzer = DataAnalyzer()
if 'pdf_processor' not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if 'huggingface_trainer' not in st.session_state:
    st.session_state.huggingface_trainer = HuggingFaceTrainer()
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = ModelTrainer(
        st.session_state.model_handler, 
        st.session_state.huggingface_trainer
    )
if 'server_status' not in st.session_state:
    st.session_state.server_status = "Inactive"
if 'resource_monitor' not in st.session_state:
    st.session_state.resource_monitor = SarahResourceMonitor(check_gpu=True)
if 'sarah_model' not in st.session_state:
    st.session_state.sarah_model = SarahModel(
        name=st.session_state.settings["ai_name"],
        gender=st.session_state.settings["ai_gender"],
        tone=st.session_state.settings["ai_tone"],
        performance_level=st.session_state.settings["performance"],
        ai_backend=st.session_state.settings.get("ai_backend", "default")
    )
if 'sarah_trainer' not in st.session_state:
    st.session_state.sarah_trainer = SarahTrainer(
        performance_level=st.session_state.settings["performance"]
    )

# Function to create a new chat
def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "messages": [], 
        "title": f"New Chat", 
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.current_chat_id = chat_id
    st.rerun()

# Function to switch to a different chat
def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id
    st.rerun()

# Function to delete a chat
def delete_chat(chat_id):
    if chat_id in st.session_state.chats and len(st.session_state.chats) > 1:
        del st.session_state.chats[chat_id]
        # Switch to the first available chat
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
        st.rerun()

# Function to handle file upload
def handle_file_upload(files):
    for file in files:
        try:
            file_type = file.type
            file_size = file.size
            
            # Log file information
            print(f"Processing file: {file.name}, Type: {file_type}, Size: {file_size} bytes")
            
            if 'pdf' in file_type:
                # Process PDF file
                text_content = st.session_state.pdf_processor.extract_text(file)
                summary = st.session_state.model_handler.summarize_text(text_content)
                return f"📄 I've analyzed the PDF '{file.name}' ({file_size:,} bytes).\n\n**Summary**:\n{summary}"
                
            elif 'image' in file_type:
                # Process image file
                return f"🖼️ I've received your image '{file.name}' ({file_size:,} bytes). What would you like me to do with it?"
                
            elif 'text' in file_type or 'csv' in file_type:
                # Process text or CSV file - use chunked processing for large files
                if file_size > 10 * 1024 * 1024:  # If file is larger than 10MB
                    if 'csv' in file_type:
                        return f"📊 The CSV file '{file.name}' is large ({file_size:,} bytes). Processing may take some time. Please use the Data Analysis tab for better handling of large files."
                    else:
                        return f"📝 I've received your large text file '{file.name}' ({file_size:,} bytes). The file is too large to display in chat. Would you like me to analyze specific sections?"
                else:
                    content = file.getvalue().decode('utf-8')
                    if 'csv' in file_type:
                        analysis = st.session_state.data_analyzer.analyze_csv(content)
                        return f"📊 I've analyzed the CSV file '{file.name}'.\n\n**Analysis**:\n{analysis}"
                    else:
                        # Truncate content display for readability
                        displayed_content = content[:500] + "...(truncated)" if len(content) > 500 else content
                        return f"📝 I've received your text file '{file.name}'.\n\nContent: {displayed_content}"
                        
            elif 'excel' in file_type or 'spreadsheet' in file_type or file.name.endswith('.xlsx') or file.name.endswith('.xls'):
                # Handle Excel files
                return f"📊 I've received your Excel file '{file.name}' ({file_size:,} bytes). Please use the Data Analysis tab for spreadsheet analysis."
                
            else:
                return f"I've received your file '{file.name}' ({file_size:,} bytes), but I'm not sure how to process this file type."
                
        except Exception as e:
            return f"⚠️ Error processing file '{file.name}': {str(e)}. Please try again or use a different file format."

# Function to toggle server status
def toggle_server():
    if st.session_state.settings["server_active"]:
        # Stop server
        stop_server()
        st.session_state.settings["server_active"] = False
        st.session_state.server_status = "Inactive"
    else:
        # Start server
        server_url = start_server()
        st.session_state.settings["server_active"] = True
        st.session_state.settings["server_url"] = server_url
        st.session_state.server_status = "Active"
    st.rerun()

# Function to update performance settings
def update_performance(performance_level):
    # Update performance level in settings
    st.session_state.settings["performance"] = performance_level
    
    # Get current AI backend
    current_backend = st.session_state.settings.get("ai_backend", "default")
    
    # Reinitialize model handler with new performance settings
    st.session_state.model_handler = ModelHandler(
        performance_level=performance_level, 
        ai_backend=current_backend
    )
    
    # Update chat handler with new model handler
    st.session_state.chat_handler = ChatHandler(st.session_state.model_handler)
    
    # Show success message
    st.info(f"Performance level updated to {performance_level}. Some changes may require application restart.")
    time.sleep(2)
    st.rerun()

# Function to update AI backend
def update_ai_backend(backend):
    # Update AI backend in settings
    st.session_state.settings["ai_backend"] = backend
    
    # Get current performance level
    current_performance = st.session_state.settings["performance"]
    
    # Reinitialize model handler with new backend
    st.session_state.model_handler = ModelHandler(
        performance_level=current_performance,
        ai_backend=backend
    )
    
    # Update chat handler with new model handler
    st.session_state.chat_handler = ChatHandler(st.session_state.model_handler)
    
    # Show success message
    st.info(f"AI backend updated to {backend}. Some changes may require application restart.")
    time.sleep(2)
    st.rerun()
    
# Function to update resource monitoring display
def update_resource_monitoring():
    # Get current system resources
    resources = st.session_state.resource_monitor.get_all_resources()
    
    # Ensure we have valid CPU usage (non-None and numeric)
    cpu_value = resources.get("cpu", 0.0)
    if cpu_value is None:
        cpu_value = 0.0
    
    # Return formatted resource data
    return {
        "cpu": cpu_value,  # Ensure this is a valid float
        "memory": resources["memory"],
        "gpu": resources.get("gpu", None),
        "timestamp": resources["timestamp"],
        "formatted_time": format_time(resources["timestamp"]),
        "memory_readable": {
            "total": readable_size(resources["memory"]["total"]),
            "available": readable_size(resources["memory"]["available"]),
            "used": readable_size(resources["memory"]["used"]),
            "percent": resources["memory"]["percent"]
        }
    }

# Function to update theme settings
def update_theme(theme, accent_color):
    # Update theme in settings
    st.session_state.settings["theme"] = theme.lower()
    st.session_state.settings["accent_color"] = accent_color
    
    # Update theme in config.toml
    try:
        # Create theme settings dict
        if theme.lower() == "dark":
            bg_color = "#0E1117"
            secondary_bg_color = "#262730"
            text_color = "#FAFAFA"
        elif theme.lower() == "light":
            bg_color = "#FFFFFF"
            secondary_bg_color = "#F0F2F6"
            text_color = "#31333F"
        elif theme.lower() == "blue":
            bg_color = "#071A33"
            secondary_bg_color = "#122D57"
            text_color = "#FFFFFF"
        elif theme.lower() == "green":
            bg_color = "#052614"
            secondary_bg_color = "#0E4A2B"
            text_color = "#FFFFFF"
        elif theme.lower() == "purple":
            bg_color = "#1E0E33"
            secondary_bg_color = "#361A5E"
            text_color = "#FFFFFF"
        else:
            # Default to dark theme
            bg_color = "#0E1117"
            secondary_bg_color = "#262730"
            text_color = "#FAFAFA"
        
        # Write updated config to file
        with open('.streamlit/config.toml', 'w') as f:
            f.write(f"""[server]
headless = true
address = "0.0.0.0"
port = 5000
maxUploadSize = 2000
enableXsrfProtection = false

[theme]
primaryColor = "{accent_color}"
backgroundColor = "{bg_color}"
secondaryBackgroundColor = "{secondary_bg_color}"
textColor = "{text_color}"
font = "sans serif"
""")
        
        # Show success message
        st.success("Theme settings updated. Refresh the page to see changes.")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Failed to update theme settings: {str(e)}")
        return False

# Sidebar layout
with st.sidebar:
    st.title(st.session_state.settings["app_name"])
    
    # Navigation tabs
    st.subheader("Navigation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💬", help="Chat with Sarah"):
            st.session_state.active_tab = "chat"
            st.rerun()
    
    with col2:
        if st.button("🧠", help="Training"):
            st.session_state.active_tab = "training"
            st.rerun()
    
    with col3:
        if st.button("📊", help="Data Analysis"):
            st.session_state.active_tab = "analysis"
            st.rerun()
    
    with col4:
        if st.button("⚙️", help="Settings"):
            st.session_state.active_tab = "settings"
            st.rerun()
    
    # Initialize active_tab if not already set
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "chat"
    
    # Chat management section (only show when in chat tab)
    if st.session_state.active_tab == "chat":
        st.subheader("Your chats")
        
        # Button to create a new chat
        if st.button("➕ New Chat"):
            create_new_chat()
        
        # Display list of chats
        for chat_id, chat_data in st.session_state.chats.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(chat_data["title"], key=f"chat_{chat_id}", use_container_width=True):
                    switch_chat(chat_id)
            with col2:
                if len(st.session_state.chats) > 1 and st.button("🗑️", key=f"delete_{chat_id}"):
                    delete_chat(chat_id)
    
    # Server status indicator
    st.divider()
    st.caption(f"Server: {st.session_state.server_status}")
    if st.session_state.settings["server_active"]:
        st.caption(f"URL: {st.session_state.settings['server_url']}")
    
    # Display AI name and version
    st.caption(f"{st.session_state.settings['ai_name']} v1.0")

# Main content area
if st.session_state.active_tab == "chat":
    st.title(f"Chat with {st.session_state.settings['ai_name']}")
    
    # Display chat messages
    chat_placeholder = st.container()
    with chat_placeholder:
        for message in st.session_state.chats[st.session_state.current_chat_id]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # File uploader
    uploaded_files = st.file_uploader("Upload files to analyze", accept_multiple_files=True, key="file_uploader")
    
    # Process files when uploaded
    if uploaded_files:
        for file in uploaded_files:
            # Check if file was already processed
            file_key = f"processed_{file.name}_{file.size}"
            if file_key not in st.session_state:
                st.session_state[file_key] = True
                response = handle_file_upload([file])
                
                # Add AI message to chat
                st.session_state.chats[st.session_state.current_chat_id]["messages"].append(
                    {"role": "assistant", "content": response}
                )
                st.rerun()
    
    # Input for chat
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add user message to chat history
        st.session_state.chats[st.session_state.current_chat_id]["messages"].append(
            {"role": "user", "content": user_input}
        )
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ai_response = st.session_state.chat_handler.generate_response(
                    user_input, 
                    st.session_state.settings["ai_name"],
                    st.session_state.settings["ai_tone"]
                )
                st.markdown(ai_response)
        
        # Add AI response to chat history
        st.session_state.chats[st.session_state.current_chat_id]["messages"].append(
            {"role": "assistant", "content": ai_response}
        )
        
        # Update chat title if it's a new chat with only one message
        if len(st.session_state.chats[st.session_state.current_chat_id]["messages"]) == 2 and \
           st.session_state.chats[st.session_state.current_chat_id]["title"] == "New Chat":
            # Generate title from the first user message
            new_title = user_input[:20] + "..." if len(user_input) > 20 else user_input
            st.session_state.chats[st.session_state.current_chat_id]["title"] = new_title
            st.rerun()

elif st.session_state.active_tab == "training":
    st.title("Training")
    
    # Training options
    training_option = st.selectbox(
        "Select training option",
        ["HuggingFace Datasets", "Upload training dataset", "Train from text input", "Train from online source"]
    )
    
    if training_option == "HuggingFace Datasets":
        st.subheader("Train with HuggingFace Datasets")
        st.markdown("""
        HuggingFace hosts thousands of datasets for AI training. Using these datasets will make your AI assistant more human-like in its responses.
        """)
        
        # Dataset categories
        dataset_categories = st.session_state.huggingface_trainer.get_dataset_categories()
        selected_category = st.selectbox(
            "Select dataset category",
            list(dataset_categories.keys())
        )
        
        # Datasets in the selected category
        if selected_category:
            datasets = dataset_categories[selected_category]
            selected_dataset = st.selectbox(
                "Select dataset",
                datasets
            )
            
            # Training format
            format_type = st.selectbox(
                "Select training format",
                ["conversation", "completion", "classification"],
                help="How to format the data for training: conversation (dialog), completion (text generation), or classification (sentiment)"
            )
            
            if st.button("Train with HuggingFace Dataset"):
                with st.spinner(f"Training model with {selected_dataset} dataset..."):
                    result = st.session_state.model_trainer.train_from_huggingface(
                        selected_dataset, format_type
                    )
                    st.success(result)
            
            # Display dataset info
            with st.expander("Dataset Information"):
                dataset_info = st.session_state.huggingface_trainer.get_dataset_info(selected_dataset)
                if dataset_info:
                    st.write(f"**Dataset:** {dataset_info.get('name', selected_dataset)}")
                    st.write(f"**Description:** {dataset_info.get('description', 'No description available')}")
                else:
                    st.write("No dataset information available")
    
    elif training_option == "Upload training dataset":
        training_file = st.file_uploader("Upload your training data", type=["txt", "csv", "json"])
        if training_file:
            if st.button("Start Training"):
                with st.spinner("Training model..."):
                    result = st.session_state.model_trainer.train_from_file(training_file)
                    st.success(result)
    
    elif training_option == "Train from text input":
        training_text = st.text_area("Enter your training text:", height=200)
        if training_text:
            if st.button("Start Training"):
                with st.spinner("Training model..."):
                    result = st.session_state.model_trainer.train_from_text(training_text)
                    st.success(result)
    
    elif training_option == "Train from online source":
        url = st.text_input("Enter URL containing training data:", 
                           help="Enter a URL to a text, CSV, JSON file, or a HuggingFace dataset (e.g., huggingface.co/datasets/daily_dialog)")
        if url:
            if st.button("Start Training"):
                with st.spinner("Training model..."):
                    result = st.session_state.model_trainer.train_from_url(url)
                    st.success(result)
    
    # Display model information
    with st.expander("Model Information", expanded=False):
        st.write(f"Current performance setting: {st.session_state.settings['performance']}")
        
        # Training history
        st.subheader("Training History")
        training_history = st.session_state.model_trainer.get_training_history()
        if training_history:
            for i, record in enumerate(training_history[-5:]):  # Show last 5 training records
                timestamp = record.get('timestamp', 'Unknown')
                source = record.get('source', 'Unknown')
                if isinstance(timestamp, (int, float)):
                    from datetime import datetime
                    timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                st.write(f"**Training #{i+1}:** {source} at {timestamp}")
        else:
            st.write("No training history available")
            
        # Model capabilities
        st.subheader("Capabilities")
        st.write("- Natural language processing")
        st.write("- Text generation")
        st.write("- Text summarization")
        st.write("- Image generation")
        st.write("- Data analysis")
        
        # Reset training button
        if st.button("Reset Training"):
            result = st.session_state.model_trainer.reset_training()
            st.success(result)

elif st.session_state.active_tab == "analysis":
    st.title("Data Analysis")
    
    analysis_option = st.selectbox(
        "Select analysis option",
        ["Upload data for analysis", "Visualize data", "Generate report"]
    )
    
    if analysis_option == "Upload data for analysis":
        data_file = st.file_uploader("Upload your data", type=["csv", "xlsx"])
        if data_file:
            if st.button("Analyze"):
                with st.spinner("Analyzing data..."):
                    analysis_result = st.session_state.data_analyzer.analyze_file(data_file)
                    st.write(analysis_result)
    
    elif analysis_option == "Visualize data":
        st.write("Upload data to create visualizations")
        viz_file = st.file_uploader("Upload your data for visualization", type=["csv", "xlsx"])
        if viz_file:
            chart_type = st.selectbox(
                "Select chart type",
                ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Heatmap"]
            )
            if st.button("Generate Visualization"):
                with st.spinner("Creating visualization..."):
                    visualization = st.session_state.data_analyzer.create_visualization(viz_file, chart_type)
                    st.plotly_chart(visualization)
    
    elif analysis_option == "Generate report":
        st.write("Upload data to generate a comprehensive report")
        report_file = st.file_uploader("Upload your data for reporting", type=["csv", "xlsx"])
        if report_file:
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    report = st.session_state.data_analyzer.generate_report(report_file)
                    st.write(report)

elif st.session_state.active_tab == "settings":
    st.title("Settings")
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["General", "AI Persona", "Performance", "AI Select", "Server"])
    
    with tab1:
        st.subheader("Application Settings")
        
        # App name setting
        new_app_name = st.text_input("Application Name", value=st.session_state.settings["app_name"])
        if new_app_name != st.session_state.settings["app_name"]:
            st.session_state.settings["app_name"] = new_app_name
        
        # Theme settings
        st.subheader("Theme Settings")
        theme_option = st.selectbox(
            "Select theme",
            ["Dark", "Light", "Blue", "Green", "Purple"],
            index=0 if st.session_state.settings["theme"] == "dark" else 1
        )
        if theme_option.lower() != st.session_state.settings["theme"]:
            st.session_state.settings["theme"] = theme_option.lower()
        
        # Accent color
        accent_color = st.color_picker("Accent Color", value=st.session_state.settings["accent_color"])
        if accent_color != st.session_state.settings["accent_color"]:
            st.session_state.settings["accent_color"] = accent_color
            
        # Apply theme button
        if st.button("Apply Theme"):
            update_theme(theme_option, accent_color)
    
    with tab2:
        st.subheader("AI Persona Settings")
        
        # AI name setting
        new_ai_name = st.text_input("AI Name", value=st.session_state.settings["ai_name"])
        if new_ai_name != st.session_state.settings["ai_name"]:
            st.session_state.settings["ai_name"] = new_ai_name
        
        # AI gender setting
        ai_gender = st.selectbox(
            "AI Gender",
            ["Female", "Male", "Non-binary"],
            index=0 if st.session_state.settings["ai_gender"] == "Female" else 
                  1 if st.session_state.settings["ai_gender"] == "Male" else 2
        )
        if ai_gender != st.session_state.settings["ai_gender"]:
            st.session_state.settings["ai_gender"] = ai_gender
        
        # AI tone setting
        ai_tone = st.selectbox(
            "AI Tone",
            ["Humorous and sarcastic", "Professional", "Friendly", "Technical", "Empathetic"],
            index=0 if st.session_state.settings["ai_tone"] == "Humorous and sarcastic" else 
                  1 if st.session_state.settings["ai_tone"] == "Professional" else
                  2 if st.session_state.settings["ai_tone"] == "Friendly" else
                  3 if st.session_state.settings["ai_tone"] == "Technical" else 4
        )
        if ai_tone != st.session_state.settings["ai_tone"]:
            st.session_state.settings["ai_tone"] = ai_tone
    
    with tab3:
        st.subheader("Performance Settings")
        
        # System resource monitoring section
        st.divider()
        st.subheader("System Resources")
        
        # Get updated resource metrics if needed - this will auto-refresh based on interval
        if "last_resource_update" not in st.session_state:
            st.session_state.last_resource_update = time.time()
            resource_data = update_resource_monitoring()
        elif st.session_state.resource_monitor.should_refresh():
            resource_data = update_resource_monitoring()
            st.session_state.last_resource_update = time.time()
        else:
            # Use the last check but ensure it has the formatted_time field
            resource_data = st.session_state.resource_monitor.last_check
            
            # If no resource data available, create it now
            if not resource_data:
                resource_data = update_resource_monitoring()
            else:
                # Make sure resource_data has all necessary fields
                if "timestamp" in resource_data and "formatted_time" not in resource_data:
                    resource_data["formatted_time"] = format_time(resource_data["timestamp"])
                
                # Add readable memory sizes if they're not already present
                if "memory" in resource_data and "memory_readable" not in resource_data:
                    resource_data["memory_readable"] = {
                        "total": readable_size(resource_data["memory"]["total"]),
                        "available": readable_size(resource_data["memory"]["available"]),
                        "used": readable_size(resource_data["memory"]["used"]),
                        "percent": resource_data["memory"]["percent"]
                    }
        
        # Auto-refresh settings
        with st.expander("Auto-Refresh Settings"):
            current_interval = st.session_state.resource_monitor.auto_refresh_interval
            new_interval = st.slider(
                "Auto-refresh interval (seconds)", 
                min_value=0, 
                max_value=60, 
                value=current_interval,
                help="Set to 0 to disable auto-refresh"
            )
            
            if new_interval != current_interval:
                st.session_state.resource_monitor.auto_refresh_interval = new_interval
                st.session_state.resource_monitor.auto_refresh_enabled = new_interval > 0
                if new_interval > 0:
                    st.success(f"Auto-refresh enabled, updating every {new_interval} seconds")
                else:
                    st.info("Auto-refresh disabled")
        
        # Create columns for displaying resource usage
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Ensure CPU value is properly formatted with 1 decimal place
            cpu_value = resource_data['cpu']
            if cpu_value is not None:
                # Round to 1 decimal place for display
                formatted_cpu = f"{cpu_value:.1f}%"
                st.metric("CPU Usage", formatted_cpu)
            else:
                st.metric("CPU Usage", "N/A")
                
            with st.expander("CPU Details"):
                st.write(f"Last updated: {resource_data['formatted_time']}")
                # Add CPU core information if available
                try:
                    st.write(f"Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
                    st.write(f"Per-core usage: {psutil.cpu_percent(percpu=True)}")
                except Exception as e:
                    st.write(f"Could not retrieve detailed CPU information: {str(e)}")
                    pass
        
        with col2:
            st.metric("RAM Usage", f"{resource_data['memory_readable']['percent']}%")
            with st.expander("Memory Details"):
                st.write(f"Total: {resource_data['memory_readable']['total']}")
                st.write(f"Used: {resource_data['memory_readable']['used']}")
                st.write(f"Available: {resource_data['memory_readable']['available']}")
                st.write(f"Last updated: {resource_data['formatted_time']}")
        
        with col3:
            # Check if GPU info is available
            if resource_data.get('gpu'):
                gpu_usage = resource_data['gpu'].get('percent', 'N/A')
                st.metric("GPU Memory", f"{gpu_usage}%")
                with st.expander("GPU Details"):
                    st.write(f"Total: {readable_size(resource_data['gpu'].get('total', 0))}")
                    st.write(f"Used: {readable_size(resource_data['gpu'].get('used', 0))}")
            else:
                st.metric("GPU Memory", "N/A")
                with st.expander("GPU Details"):
                    st.write("No GPU detected or GPU monitoring not available")
        
        # Manual refresh button
        if st.button("Refresh Resource Info", key="resource_refresh_button"):
            st.session_state.last_resource_update = time.time()
            st.rerun()
        
        st.divider()
        
        # Performance level slider
        st.subheader("Performance Level")
        st.write("Adjust performance based on your computer's capabilities")
        performance_option = st.select_slider(
            "Performance Level",
            options=["Low-end", "Balanced", "High-end"],
            value="Balanced" if st.session_state.settings["performance"] == "balanced" else
                  "Low-end" if st.session_state.settings["performance"] == "low" else "High-end"
        )
        
        # Map slider value to setting value
        perf_map = {"Low-end": "low", "Balanced": "balanced", "High-end": "high"}
        
        # Only update if changed
        if perf_map[performance_option] != st.session_state.settings["performance"]:
            if st.button("Apply Performance Change"):
                update_performance(perf_map[performance_option])
        
        st.write("**Performance details:**")
        if performance_option == "Low-end":
            st.write("- Uses smaller models with lower resource requirements")
            st.write("- Suitable for computers with limited RAM and CPU power")
            st.write("- Some advanced features may be limited")
        elif performance_option == "Balanced":
            st.write("- Uses medium-sized models with moderate resource requirements")
            st.write("- Good balance between performance and resource usage")
            st.write("- Most features available with reasonable performance")
        else:  # High-end
            st.write("- Uses larger models with higher resource requirements")
            st.write("- Best for computers with ample RAM and CPU/GPU power")
            st.write("- All features available with maximum performance")
        
        st.divider()

    
    with tab4:
        st.subheader("AI Model Selection")
        
        # AI Backend selector
        st.subheader("AI Backend")
        st.write("Choose which AI backend to use for responses")
        
        # Get available backends from model handler
        try:
            available_backends = st.session_state.model_handler.get_available_backends()
            backend_names = {backend["id"]: backend["name"] for backend in available_backends}
            backend_options = list(backend_names.keys())
            
            # Create a selectbox with human-readable names
            current_backend = st.session_state.settings.get("ai_backend", "default")
            backend_index = backend_options.index(current_backend) if current_backend in backend_options else 0
            
            selected_backend = st.selectbox(
                "AI Backend",
                backend_options,
                index=backend_index,
                format_func=lambda x: backend_names.get(x, x.capitalize())
            )
            
            # Display backend information
            for backend in available_backends:
                if backend["id"] == selected_backend:
                    st.write(f"**Description:** {backend.get('description', 'No description available')}")
                    
                    # Show API key status for backends that require one
                    if backend.get("requires_api_key", False):
                        api_key_status = "✅ Available" if backend.get("has_api_key", False) else "❌ Missing"
                        st.write(f"**API Key:** {api_key_status}")
                        
                        if not backend.get("has_api_key", False):
                            st.warning(f"An API key is required for the {backend['name']} backend. Add your API key in the Server tab.")
                    
                    # Show server requirement for backends that need a local server
                    if backend.get("server_required", False):
                        st.write("**Note:** Requires a running Ollama server")
            
            # Only update if changed
            if selected_backend != current_backend:
                if st.button("Apply Backend Change"):
                    update_ai_backend(selected_backend)
                    
        except Exception as e:
            st.error(f"Error loading AI backends: {str(e)}")
            st.write("Using default AI backend")
        
        # Custom model section
        st.divider()
        st.subheader("Custom Models")
        
        # Select model type
        model_type = st.selectbox(
            "Model Type",
            ["Ollama", "Local Hugging Face", "External API"]
        )
        
        if model_type == "Ollama":
            st.write("Manage models for Ollama integration")
            
            # Ollama server info
            ollama_server = st.text_input(
                "Ollama Server URL", 
                value=st.session_state.settings.get("ollama_server_url", "http://localhost:11434"),
                help="The URL where your Ollama server is running"
            )
            
            if ollama_server != st.session_state.settings.get("ollama_server_url", "http://localhost:11434"):
                st.session_state.settings["ollama_server_url"] = ollama_server
            
            # Test connection to server
            if st.button("Test Ollama Connection"):
                try:
                    with st.spinner("Testing connection to Ollama server..."):
                        # Update Sarah's Ollama connection
                        if 'sarah_model' in st.session_state:
                            st.session_state.sarah_model.change_backend("ollama")
                            
                        # Test connection through model handler
                        is_connected = st.session_state.model_handler.test_ollama_connection(ollama_server)
                        if is_connected:
                            st.success("Successfully connected to Ollama server!")
                            # Also get available models
                            models = st.session_state.model_handler.get_ollama_models()
                            if models:
                                st.write(f"Found {len(models)} available models")
                        else:
                            st.error("Failed to connect to Ollama server. Please check the URL and ensure Ollama is running.")
                except Exception as e:
                    st.error(f"Error connecting to Ollama server: {str(e)}")
            
            # Option to pull from Ollama library
            st.write("**Pull a model from Ollama library**")
            ollama_model_name = st.text_input("Ollama Model Name (e.g., llama2, mistral, codellama)", help="Model identifier from Ollama library")
            
            if st.button("Pull Ollama Model") and ollama_model_name:
                try:
                    with st.spinner(f"Pulling {ollama_model_name} model from Ollama library..."):
                        # Use Sarah model handler to pull model
                        result = st.session_state.sarah_model.pull_ollama_model(ollama_model_name)
                        st.success(f"Successfully initiated download of {ollama_model_name}. This may take several minutes depending on model size.")
                except Exception as e:
                    st.error(f"Failed to pull model: {str(e)}")
            
            # Display available Ollama models
            try:
                st.subheader("Available Ollama Models")
                models = st.session_state.model_handler.get_ollama_models()
                
                if models and len(models) > 0:
                    # Create a readable display of models
                    for model in models:
                        model_name = model.get('name', 'Unknown')
                        model_size = model.get('size', 0)
                        model_modified = model.get('modified', 'Unknown')
                        
                        # Display model information
                        st.write(f"**{model_name}**")
                        st.write(f"Size: {readable_size(model_size)} • Modified: {model_modified}")
                        
                        # Add a button to set this as active model
                        if st.button(f"Use {model_name}", key=f"use_{model_name}"):
                            # Set this model as the active Ollama model
                            st.session_state.settings["ollama_active_model"] = model_name
                            
                            # Update Sarah model
                            if 'sarah_model' in st.session_state:
                                st.session_state.sarah_model.set_ollama_model(model_name)
                                
                            st.success(f"Now using {model_name} as the active Ollama model")
                            time.sleep(1)
                            st.rerun()
                else:
                    st.write("No Ollama models available. Use the 'Pull Ollama Model' function above to download models.")
            except Exception as e:
                st.write("Unable to fetch Ollama models. Make sure the Ollama server is running.")
                
            # Advanced Ollama settings
            with st.expander("Ollama Advanced Settings"):
                st.write("Configure the parameters for Ollama model inference")
                
                # Temperature settings
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1,
                                       help="Higher values make output more random, lower values more deterministic")
                
                # Context length
                context_length = st.slider("Context Length", 512, 8192, 2048, 512,
                                          help="Maximum length of input context. Higher values use more memory.")
                
                # Top P
                top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05,
                                 help="Controls diversity via nucleus sampling")
                
                if st.button("Apply Ollama Settings"):
                    # Save settings
                    st.session_state.settings["ollama_temperature"] = temperature
                    st.session_state.settings["ollama_context_length"] = context_length
                    st.session_state.settings["ollama_top_p"] = top_p
                    
                    # Apply to Sarah model
                    if 'sarah_model' in st.session_state:
                        st.session_state.sarah_model.update_ollama_settings({
                            "temperature": temperature,
                            "context_length": context_length,
                            "top_p": top_p
                        })
                        
                    st.success("Ollama settings applied successfully!")
                    time.sleep(1)
                    st.rerun()
            
            # Upload custom model files for Ollama
            st.write("**Upload custom model files for Ollama**")
            model_files = st.file_uploader("Upload model files", accept_multiple_files=True, 
                                          help="Upload custom model files to use with Ollama")
            
            if model_files and st.button("Process Model Files"):
                try:
                    model_names = [file.name for file in model_files]
                    model_file_info = ", ".join(model_names)
                    
                    # Placeholder for model processing logic
                    st.info(f"Processing model files: {model_file_info}")
                    st.warning("Custom model upload requires Ollama API integration. This is a placeholder for demonstration.")
                except Exception as e:
                    st.error(f"Error processing model files: {str(e)}")
        
        elif model_type == "Local Hugging Face":
            st.write("Load and use models from Hugging Face locally")
            
            # Model selection
            hf_model = st.text_input("Hugging Face Model Name (e.g., facebook/opt-350m, gpt2, etc.)")
            
            col1, col2 = st.columns(2)
            with col1:
                quantization = st.selectbox("Quantization", ["None", "8-bit", "4-bit"])
            with col2:
                device = st.selectbox("Device", ["CPU", "GPU", "Auto"])
            
            if st.button("Download Model") and hf_model:
                try:
                    with st.spinner(f"Downloading {hf_model} from Hugging Face..."):
                        # Placeholder for model downloading
                        st.info(f"Started downloading {hf_model}. This may take several minutes.")
                        st.warning("Local Hugging Face model integration is a placeholder for demonstration.")
                except Exception as e:
                    st.error(f"Failed to download model: {str(e)}")
        
        elif model_type == "External API":
            st.write("Configure external API for AI model access")
            
            # API configuration
            api_url = st.text_input("API URL", placeholder="https://api.example.com/v1/completions")
            api_key = st.text_input("API Key", type="password")
            
            with st.expander("Advanced Settings"):
                request_format = st.text_area("Request Format (JSON)", "{\"prompt\": \"%s\", \"max_tokens\": 1000}")
                response_path = st.text_input("Response JSON Path", "choices[0].text")
            
            if st.button("Test API Connection") and api_url:
                if not api_key:
                    st.warning("API key not provided. The connection test may fail if the API requires authentication.")
                
                try:
                    with st.spinner("Testing API connection..."):
                        # Placeholder for API testing
                        st.info("API connection test functionality is a placeholder for demonstration.")
                except Exception as e:
                    st.error(f"API connection failed: {str(e)}")
        
        # Selective training section
        st.divider()
        st.subheader("Selective Training")
        
        st.write("Train specific capabilities while preserving others")
        
        capability_options = [
            "conversation", 
            "summarization", 
            "storytelling", 
            "article_generation", 
            "sentiment_analysis", 
            "keyword_extraction"
        ]
        
        selected_capability = st.selectbox(
            "Select capability to train",
            capability_options
        )
        
        training_method = st.radio(
            "Training method",
            ["Text Input", "File Upload", "HuggingFace Dataset"]
        )
        
        if training_method == "Text Input":
            training_text = st.text_area("Enter training text:", height=150)
            if st.button("Train Selected Capability"):
                if training_text:
                    with st.spinner(f"Selectively training {selected_capability} capability..."):
                        result = st.session_state.sarah_trainer.selective_training(
                            capability=selected_capability,
                            training_data=training_text
                        )
                        st.success(f"Capability '{selected_capability}' trained successfully!")
                else:
                    st.warning("Please enter some training text first.")
                    
        elif training_method == "File Upload":
            training_file = st.file_uploader("Upload training data", type=["txt", "csv", "json"])
            if training_file:
                if st.button("Train Selected Capability"):
                    with st.spinner(f"Selectively training {selected_capability} capability..."):
                        # Convert file to string for processing
                        file_content = training_file.getvalue().decode("utf-8")
                        result = st.session_state.sarah_trainer.selective_training(
                            capability=selected_capability,
                            training_data=file_content
                        )
                        st.success(f"Capability '{selected_capability}' trained successfully!")
                        
        elif training_method == "HuggingFace Dataset":
            dataset_options = st.session_state.sarah_trainer.get_recommended_datasets().get(
                selected_capability, ["No specific datasets available"]
            )
            selected_dataset = st.selectbox("Select dataset", dataset_options)
            
            if st.button("Train with Dataset"):
                with st.spinner(f"Training {selected_capability} with {selected_dataset} dataset..."):
                    result = st.session_state.sarah_trainer.selective_training(
                        capability=selected_capability,
                        dataset=selected_dataset
                    )
                    st.success(f"Capability '{selected_capability}' trained successfully with {selected_dataset}!")
        
        # Unsupervised Learning section
        st.divider()
        st.subheader("Unsupervised Learning")
        
        st.write("""
        Unsupervised learning helps Sarah discover patterns and concepts in data without explicit labels.
        This improves general intelligence and adaptability across multiple domains.
        """)
        
        unsupervised_learning_method = st.radio(
            "Unsupervised learning method",
            ["Text Input", "File Upload", "HuggingFace Dataset"],
            key="unsupervised_method"
        )
        
        iterations = st.slider(
            "Learning iterations", 
            min_value=1, 
            max_value=20, 
            value=5,
            help="More iterations means deeper learning but takes longer"
        )
        
        if unsupervised_learning_method == "Text Input":
            unsupervised_text = st.text_area(
                "Enter text for unsupervised learning:", 
                height=150,
                key="unsupervised_text_input",
                help="Longer and more diverse text allows for better pattern recognition"
            )
            
            if st.button("Start Unsupervised Learning", key="unsupervised_text_button"):
                if unsupervised_text:
                    with st.spinner("Running unsupervised learning algorithm..."):
                        result = st.session_state.sarah_trainer.unsupervised_learning(
                            data_source="text",
                            data_content=unsupervised_text,
                            iterations=iterations
                        )
                        
                        if result["success"]:
                            st.success(result["message"])
                            
                            # Display metrics in an expander
                            with st.expander("Learning Metrics"):
                                metrics = result["metrics"]
                                st.write(f"**Duration:** {metrics['duration']:.2f} seconds")
                                st.write(f"**Iterations:** {metrics['iterations']}")
                                st.write(f"**Data size:** {metrics['data_size']} characters")
                                st.write(f"**Chunks processed:** {metrics['chunks_processed']}")
                                st.write(f"**Concepts identified:** {metrics['concepts_identified']}")
                                st.write(f"**Patterns found:** {metrics['patterns_found']}")
                        else:
                            st.error(result["message"])
                else:
                    st.warning("Please enter some text for unsupervised learning.")
        
        elif unsupervised_learning_method == "File Upload":
            unsupervised_file = st.file_uploader(
                "Upload data for unsupervised learning", 
                type=["txt", "csv", "json"],
                key="unsupervised_file_uploader"
            )
            
            if unsupervised_file:
                if st.button("Start Unsupervised Learning", key="unsupervised_file_button"):
                    with st.spinner("Running unsupervised learning algorithm..."):
                        # Convert file to string for processing
                        file_content = unsupervised_file.getvalue().decode("utf-8")
                        
                        result = st.session_state.sarah_trainer.unsupervised_learning(
                            data_source="text",  # Still using text source since we've converted the file
                            data_content=file_content,
                            iterations=iterations
                        )
                        
                        if result["success"]:
                            st.success(result["message"])
                            
                            # Display metrics in an expander
                            with st.expander("Learning Metrics"):
                                metrics = result["metrics"]
                                st.write(f"**Duration:** {metrics['duration']:.2f} seconds")
                                st.write(f"**Iterations:** {metrics['iterations']}")
                                st.write(f"**Data size:** {metrics['data_size']} characters")
                                st.write(f"**Chunks processed:** {metrics['chunks_processed']}")
                                st.write(f"**Concepts identified:** {metrics['concepts_identified']}")
                                st.write(f"**Patterns found:** {metrics['patterns_found']}")
                        else:
                            st.error(result["message"])
        
        elif unsupervised_learning_method == "HuggingFace Dataset":
            # Get all datasets across different capabilities
            all_datasets = []
            for datasets in st.session_state.sarah_trainer.get_recommended_datasets().values():
                all_datasets.extend(datasets)
            
            # Remove duplicates and sort
            all_datasets = sorted(list(set(all_datasets)))
            
            selected_dataset = st.selectbox(
                "Select dataset for unsupervised learning", 
                all_datasets,
                key="unsupervised_dataset_select"
            )
            
            if st.button("Start Unsupervised Learning with Dataset", key="unsupervised_dataset_button"):
                with st.spinner(f"Running unsupervised learning on {selected_dataset} dataset..."):
                    result = st.session_state.sarah_trainer.unsupervised_learning(
                        data_source="dataset",
                        dataset_name=selected_dataset,
                        iterations=iterations
                    )
                    
                    if result["success"]:
                        st.success(result["message"])
                        
                        # Display metrics if available
                        if result.get("metrics"):
                            with st.expander("Learning Metrics"):
                                metrics = result["metrics"]
                                for key, value in metrics.items():
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.error(result["message"])
    
    with tab5:
        st.subheader("Server Settings")
        
        st.write("Host a web server to access the AI assistant from other devices")
        
        server_status = "Active" if st.session_state.settings["server_active"] else "Inactive"
        st.write(f"Server Status: **{server_status}**")
        
        if st.session_state.settings["server_active"]:
            st.write(f"Server URL: {st.session_state.settings['server_url']}")
            if st.button("Stop Server"):
                toggle_server()
        else:
            if st.button("Start Server"):
                toggle_server()
    
    # Save settings button
    if st.button("Save Settings"):
        # Save settings to file
        with open('data/default_settings.json', 'w') as f:
            json.dump(st.session_state.settings, f)
        st.success("Settings saved successfully!")
