"""
Model Trainer Module

This module handles the training of AI models with various types of data sources.
It provides functionality to train the AI model for better human-like responses.
"""

import os
import time
import json
import random
from typing import Dict, List, Any, Optional, Union

class ModelTrainer:
    def __init__(self, model_handler, huggingface_trainer=None):
        """
        Initialize the model trainer.
        
        Args:
            model_handler: The model handler to train
            huggingface_trainer: Optional HuggingFace trainer instance
        """
        self.model_handler = model_handler
        self.huggingface_trainer = huggingface_trainer
        self.training_history = []
        self.training_data_dir = "data/training"
        
        # Create training data directory if it doesn't exist
        os.makedirs(self.training_data_dir, exist_ok=True)
    
    def train_from_file(self, file) -> str:
        """
        Train the model using a file.
        
        Args:
            file: The training data file
            
        Returns:
            str: Training result message
        """
        try:
            file_type = file.type
            file_name = file.name
            
            # Process different file types
            if 'text' in file_type or file_name.endswith('.txt'):
                content = file.getvalue().decode('utf-8')
                return self.train_from_text(content)
            
            elif 'json' in file_type or file_name.endswith('.json'):
                content = json.loads(file.getvalue().decode('utf-8'))
                return self.train_from_json(content)
            
            elif 'csv' in file_type or file_name.endswith('.csv'):
                # Save the file temporarily
                temp_path = os.path.join(self.training_data_dir, file_name)
                with open(temp_path, 'wb') as f:
                    f.write(file.getvalue())
                
                return self.train_from_csv(temp_path)
            
            else:
                return f"Unsupported file type: {file_type}. Please use TXT, JSON, or CSV files."
        
        except Exception as e:
            return f"Error during training: {str(e)}"
    
    def train_from_text(self, text: str) -> str:
        """
        Train the model using text input.
        
        Args:
            text: The training text
            
        Returns:
            str: Training result message
        """
        try:
            # Simple validation
            if not text or len(text) < 50:
                return "Training text is too short. Please provide more content."
            
            print(f"Training model with text (length: {len(text)} characters)")
            
            # Simulate training process
            start_time = time.time()
            
            # Process the text into training examples (simplified)
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            training_examples = []
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 10:  # Basic filtering
                    training_examples.append({
                        "type": "text",
                        "content": paragraph,
                        "index": i
                    })
            
            # Record training history
            training_record = {
                "source": "text_input",
                "timestamp": time.time(),
                "examples": len(training_examples),
                "content_length": len(text)
            }
            self.training_history.append(training_record)
            
            # Simulate training time based on text length
            sleep_time = min(3, len(text) / 10000)
            time.sleep(sleep_time)
            
            # Update model knowledge
            self.model_handler.update_knowledge({
                "training_examples": len(training_examples),
                "last_trained": time.time()
            })
            
            time_taken = time.time() - start_time
            
            return (f"Training completed in {time_taken:.2f} seconds. "
                    f"Processed {len(training_examples)} examples from {len(paragraphs)} paragraphs. "
                    f"Model knowledge has been updated.")
        
        except Exception as e:
            return f"Error during text training: {str(e)}"
    
    def train_from_json(self, data: Union[Dict, List]) -> str:
        """
        Train the model using JSON data.
        
        Args:
            data: The training data in JSON format
            
        Returns:
            str: Training result message
        """
        try:
            # Validate data structure
            if isinstance(data, dict):
                if "examples" in data and isinstance(data["examples"], list):
                    examples = data["examples"]
                else:
                    examples = [data]  # Use the whole dict as a single example
            elif isinstance(data, list):
                examples = data
            else:
                return "Invalid JSON data format. Expected object or array."
            
            if not examples:
                return "No training examples found in the JSON data."
            
            print(f"Training model with {len(examples)} JSON examples")
            
            # Simulate training process
            start_time = time.time()
            
            # Record training history
            training_record = {
                "source": "json_input",
                "timestamp": time.time(),
                "examples": len(examples)
            }
            self.training_history.append(training_record)
            
            # Simulate training time based on number of examples
            sleep_time = min(3, len(examples) / 10)
            time.sleep(sleep_time)
            
            # Update model knowledge
            self.model_handler.update_knowledge({
                "training_examples": len(examples),
                "last_trained": time.time()
            })
            
            time_taken = time.time() - start_time
            
            return (f"JSON training completed in {time_taken:.2f} seconds. "
                    f"Processed {len(examples)} examples. "
                    f"Model knowledge has been updated.")
        
        except Exception as e:
            return f"Error during JSON training: {str(e)}"
    
    def train_from_csv(self, file_path: str) -> str:
        """
        Train the model using a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            str: Training result message
        """
        try:
            import pandas as pd
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            if df.empty:
                return "The CSV file is empty."
            
            row_count = len(df)
            col_count = len(df.columns)
            print(f"Training model with CSV data: {row_count} rows, {col_count} columns")
            
            # Simulate training process
            start_time = time.time()
            
            # Convert CSV to training examples (simplified)
            examples = []
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            
            if text_columns:
                # Extract text data from columns
                for _, row in df.iterrows():
                    example = {col: str(row[col]) for col in text_columns if pd.notna(row[col])}
                    if example:
                        examples.append(example)
            
            # Record training history
            training_record = {
                "source": "csv_input",
                "timestamp": time.time(),
                "rows": row_count,
                "columns": col_count,
                "examples": len(examples)
            }
            self.training_history.append(training_record)
            
            # Simulate training time based on number of examples
            sleep_time = min(3, len(examples) / 50)
            time.sleep(sleep_time)
            
            # Update model knowledge
            self.model_handler.update_knowledge({
                "training_examples": len(examples),
                "last_trained": time.time()
            })
            
            # Cleanup
            if os.path.exists(file_path):
                os.remove(file_path)
            
            time_taken = time.time() - start_time
            
            return (f"CSV training completed in {time_taken:.2f} seconds. "
                    f"Processed {len(examples)} examples from {row_count} rows. "
                    f"Model knowledge has been updated.")
        
        except Exception as e:
            return f"Error during CSV training: {str(e)}"
    
    def train_from_url(self, url: str) -> str:
        """
        Train the model using data from a URL.
        
        Args:
            url: URL pointing to training data
            
        Returns:
            str: Training result message
        """
        try:
            import requests
            
            # Check if it's a HuggingFace dataset URL
            if 'huggingface.co/datasets' in url:
                if self.huggingface_trainer:
                    dataset_id = url.split('datasets/')[-1].split('/')[0]
                    return self.train_from_huggingface(dataset_id)
                else:
                    return "HuggingFace trainer not available. Please initialize it first."
            
            # Fetch the content from the URL
            print(f"Fetching training data from URL: {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return f"Failed to fetch data from URL: {response.status_code}"
            
            content_type = response.headers.get('Content-Type', '')
            
            # Process based on content type
            if 'json' in content_type:
                data = response.json()
                return self.train_from_json(data)
            
            elif 'text/csv' in content_type or 'application/csv' in content_type:
                # Save the content to a temporary file
                temp_path = os.path.join(self.training_data_dir, 'temp_url_data.csv')
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                
                return self.train_from_csv(temp_path)
            
            elif 'text/' in content_type:
                return self.train_from_text(response.text)
            
            else:
                return f"Unsupported content type: {content_type}"
        
        except Exception as e:
            return f"Error fetching or processing URL data: {str(e)}"
    
    def train_from_huggingface(self, dataset_id: str, format_type: str = "conversation") -> str:
        """
        Train the model using a HuggingFace dataset.
        
        Args:
            dataset_id: The HuggingFace dataset ID
            format_type: The format type for training data
            
        Returns:
            str: Training result message
        """
        if not self.huggingface_trainer:
            return "HuggingFace trainer not available. Please initialize it first."
        
        try:
            print(f"Training model with HuggingFace dataset: {dataset_id}")
            
            # Get dataset info
            dataset_info = self.huggingface_trainer.get_dataset_info(dataset_id)
            if not dataset_info:
                return f"Dataset not found: {dataset_id}"
            
            # Train the model
            results = self.huggingface_trainer.train_model_with_dataset(
                self.model_handler, dataset_id, format_type
            )
            
            if not results.get('success', False):
                return f"Training failed: {results.get('error', 'Unknown error')}"
            
            # Record training history
            training_record = {
                "source": "huggingface",
                "dataset": dataset_id,
                "timestamp": time.time(),
                "examples": results.get('examples_trained', 0),
                "format": format_type
            }
            self.training_history.append(training_record)
            
            time_taken = results.get('time_taken', 0)
            examples_trained = results.get('examples_trained', 0)
            
            return (f"HuggingFace dataset training completed in {time_taken:.2f} seconds. "
                    f"Trained on {examples_trained} examples from '{dataset_info.get('name', dataset_id)}'. "
                    f"Model knowledge has been updated, making responses more human-like.")
        
        except Exception as e:
            return f"Error during HuggingFace dataset training: {str(e)}"
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get the training history.
        
        Returns:
            List of training history records
        """
        return self.training_history
    
    def reset_training(self) -> str:
        """
        Reset the training history and model knowledge.
        
        Returns:
            str: Status message
        """
        self.training_history = []
        self.model_handler.reset_knowledge()
        return "Training history and model knowledge have been reset."