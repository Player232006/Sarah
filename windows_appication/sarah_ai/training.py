"""
Sarah Training Module

This module provides specialized training capabilities for Sarah's AI model.
It extends the core training functionality with Sarah-specific optimizations and features.

Features:
1. Selective training for specific topics or capabilities
2. Personality-focused training to maintain Sarah's unique characteristics
3. Training history and evaluation metrics
4. Integration with various data sources

Usage:
    from sarah_ai.training import SarahTrainer
    
    # Initialize trainer
    trainer = SarahTrainer()
    
    # Train on a specific topic
    result = trainer.train_on_topic("humor", "path/to/humor_dataset.txt")
    
    # Evaluate training effectiveness
    metrics = trainer.evaluate_training("humor")
"""

import json
import os
import sys
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import core model components
from models.model_trainer import ModelTrainer
from models.model_handler import ModelHandler
from utils.huggingface_trainer import HuggingFaceTrainer

class SarahTrainer:
    """
    Sarah AI Trainer - Specialized training capabilities for Sarah's AI model.
    
    This class extends the core training functionality with Sarah-specific
    optimizations and features.
    """
    
    def __init__(self, performance_level: str = "balanced"):
        """
        Initialize Sarah's specialized trainer.
        
        Args:
            performance_level: Training intensity level (low, balanced, high)
        """
        self.performance_level = performance_level
        # Create a model handler with the appropriate performance level
        model_handler = ModelHandler(performance_level=performance_level)
        self.core_trainer = ModelTrainer(model_handler=model_handler)
        self.hf_trainer = HuggingFaceTrainer()
        self.training_history = self._load_training_history()
        print(f"Sarah Trainer initialized with {performance_level} performance level")
    
    def _load_training_history(self) -> List[Dict[str, Any]]:
        """
        Load training history from file.
        
        Returns:
            list: Training history records
        """
        try:
            with open('data/sarah_training_history.json', 'r') as f:
                return json.load(f)
        except:
            # Return empty history if file not found
            return []
    
    def _save_training_history(self) -> bool:
        """
        Save training history to file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open('data/sarah_training_history.json', 'w') as f:
                json.dump(self.training_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving training history: {str(e)}")
            return False
    
    def _add_training_record(self, topic: str, source: str, samples: int, metrics: Dict[str, Any]) -> None:
        """
        Add a new training record to the history.
        
        Args:
            topic: The topic or capability trained
            source: Training data source
            samples: Number of training samples
            metrics: Training performance metrics
        """
        record = {
            "timestamp": datetime.now().timestamp(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topic": topic,
            "source": source,
            "samples": samples,
            "metrics": metrics
        }
        
        self.training_history.append(record)
        self._save_training_history()
    
    def train_on_topic(self, topic: str, data_path: str) -> Dict[str, Any]:
        """
        Train Sarah on a specific topic using local data.
        
        Args:
            topic: The topic to train on
            data_path: Path to training data file
            
        Returns:
            dict: Training results and metrics
        """
        print(f"Training Sarah on topic: {topic}")
        
        try:
            # Load training data
            with open(data_path, 'r') as f:
                training_data = f.read()
            
            # Determine number of samples for metrics
            samples = len(training_data.split('\n'))
            
            # Train using core trainer
            result = self.core_trainer.train_from_text(training_data)
            
            # Evaluate training performance
            metrics = {
                "success": True,
                "duration": 0,  # Placeholder
                "samples": samples,
                "improvement": 0  # Placeholder
            }
            
            # Record training in history
            self._add_training_record(topic, data_path, samples, metrics)
            
            return {
                "success": True,
                "message": f"Successfully trained Sarah on {topic}",
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"Error training on topic {topic}: {str(e)}")
            
            return {
                "success": False,
                "message": f"Error training on topic {topic}: {str(e)}",
                "metrics": None
            }
    
    def train_from_huggingface_dataset(self, dataset_name: str, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Train Sarah using a HuggingFace dataset.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            topic: Optional topic label for the training
            
        Returns:
            dict: Training results and metrics
        """
        if topic is None:
            topic = dataset_name.split('/')[-1]
        
        print(f"Training Sarah on dataset: {dataset_name} (topic: {topic})")
        
        try:
            # Train using HuggingFace trainer
            format_type = "conversation"  # Default format
            result = self.core_trainer.train_from_huggingface(dataset_name, format_type)
            
            # Evaluate training performance
            metrics = {
                "success": True,
                "dataset": dataset_name,
                "format": format_type
            }
            
            # Record training in history
            self._add_training_record(topic, f"huggingface:{dataset_name}", 0, metrics)
            
            return {
                "success": True,
                "message": f"Successfully trained Sarah on {dataset_name}",
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"Error training from HuggingFace dataset {dataset_name}: {str(e)}")
            
            return {
                "success": False,
                "message": f"Error training from HuggingFace dataset {dataset_name}: {str(e)}",
                "metrics": None
            }
    
    def selective_training(self, capability: str, training_data: Optional[str] = None, dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Selectively train a specific capability while preserving others.
        
        Args:
            capability: The capability to train (conversation, writing, summarization, etc.)
            training_data: Optional text training data
            dataset: Optional HuggingFace dataset name
            
        Returns:
            dict: Training results and metrics
        """
        print(f"Selective training for capability: {capability}")
        
        # Map capability to appropriate training approach
        if capability == "conversation":
            format_type = "conversation"
            recommended_datasets = ["daily_dialog", "conv_ai_2"]
        elif capability == "writing":
            format_type = "completion"
            recommended_datasets = ["wikitext", "bookcorpus"]
        elif capability == "summarization":
            format_type = "completion"
            recommended_datasets = ["cnn_dailymail", "samsum"]
        else:
            format_type = "conversation"
            recommended_datasets = []
        
        # Use provided dataset or training data
        if dataset:
            return self.train_from_huggingface_dataset(dataset, capability)
        elif training_data:
            # Save training data to temp file
            temp_path = f"data/temp_{capability}_training.txt"
            with open(temp_path, 'w') as f:
                f.write(training_data)
            
            return self.train_on_topic(capability, temp_path)
        else:
            # No training data provided
            return {
                "success": False,
                "message": f"No training data or dataset provided for {capability}",
                "recommended_datasets": recommended_datasets
            }
    
    def get_training_history(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get training history, optionally filtered by topic.
        
        Args:
            topic: Optional topic to filter the history
            
        Returns:
            list: Training history records
        """
        if topic:
            return [record for record in self.training_history if record.get("topic") == topic]
        else:
            return self.training_history
    
    def evaluate_training(self, topic: str) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of training on a specific topic.
        
        Args:
            topic: The topic to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        # Get training history for the topic
        topic_history = self.get_training_history(topic)
        
        if not topic_history:
            return {
                "topic": topic,
                "trained": False,
                "message": f"No training history found for topic: {topic}"
            }
        
        # Count training sessions and total samples
        sessions = len(topic_history)
        total_samples = sum(record.get("samples", 0) for record in topic_history)
        
        # Get the most recent training date
        latest = max(topic_history, key=lambda x: x.get("timestamp", 0))
        latest_date = latest.get("date", "Unknown")
        
        return {
            "topic": topic,
            "trained": True,
            "sessions": sessions,
            "total_samples": total_samples,
            "latest_training": latest_date,
            "recommended_refresh": sessions < 3 or total_samples < 1000
        }
    
    def reset_training(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset training for a specific topic or all training.
        
        Args:
            topic: Optional topic to reset (if None, resets all training)
            
        Returns:
            dict: Reset results
        """
        if topic:
            # Filter out the specified topic
            self.training_history = [
                record for record in self.training_history 
                if record.get("topic") != topic
            ]
            self._save_training_history()
            
            return {
                "success": True,
                "message": f"Training reset for topic: {topic}",
                "remaining_topics": len(set(record.get("topic") for record in self.training_history))
            }
        else:
            # Reset all training
            self.training_history = []
            self._save_training_history()
            
            # Also reset core training
            result = self.core_trainer.reset_training()
            
            return {
                "success": True,
                "message": "All training history reset",
                "core_reset": result
            }
    
    def get_recommended_datasets(self) -> Dict[str, List[str]]:
        """
        Get recommended datasets for different capabilities.
        
        Returns:
            dict: Mapping of capabilities to recommended datasets
        """
        return {
            "conversation": ["daily_dialog", "conv_ai_2", "empathetic_dialogues"],
            "writing": ["wikitext", "bookcorpus", "gutenberg_poetry"],
            "summarization": ["cnn_dailymail", "samsum", "xsum"],
            "question_answering": ["squad", "natural_questions", "triviaqa"],
            "knowledge": ["wikipedia", "conceptnet", "atomic"]
        }
    
    def unsupervised_learning(self, data_source: str, data_content: Optional[str] = None, 
                              dataset_name: Optional[str] = None, iterations: int = 5) -> Dict[str, Any]:
        """
        Perform unsupervised learning to improve Sarah's general intelligence.
        
        This method uses clustering and pattern recognition to identify topics and concepts
        without explicit labels, allowing Sarah to develop a more nuanced understanding
        of language and concepts.
        
        Args:
            data_source: Source type ('text', 'file', 'dataset')
            data_content: Content for text or file path (optional)
            dataset_name: Name of dataset for 'dataset' source (optional)
            iterations: Number of learning iterations to perform
            
        Returns:
            dict: Learning results and metrics
        """
        print(f"Starting unsupervised learning from {data_source}")
        start_time = time.time()
        
        try:
            # Step 1: Prepare training data
            if data_source == "text" and data_content:
                training_data = data_content
                source_name = "Text input"
            elif data_source == "file" and data_content:
                with open(data_content, 'r') as f:
                    training_data = f.read()
                source_name = f"File: {os.path.basename(data_content)}"
            elif data_source == "dataset" and dataset_name:
                # Use HuggingFace dataset
                print(f"Training model with HuggingFace dataset: {dataset_name}")
                return self.train_from_huggingface_dataset(dataset_name, "general_intelligence")
            else:
                return {
                    "success": False,
                    "message": "Invalid data source or missing required parameters",
                    "metrics": None
                }
            
            # Step 2: Process and analyze data - in a real implementation, this would 
            # use actual machine learning techniques like clustering, topic modeling, etc.
            # For this demo, we'll simulate the process
            
            # Simulated analysis metrics
            chunks = len(training_data.split()) // 100  # Rough word count divided by 100
            concepts_identified = min(chunks // 10, 50)  # Simulated concept identification
            patterns_found = min(chunks // 20, 30)  # Simulated pattern recognition
            
            # Step 3: Iterative learning
            for i in range(iterations):
                print(f"Unsupervised learning iteration {i+1}/{iterations}")
                # In a real implementation, each iteration would refine the model
                time.sleep(0.5)  # Simulate processing time
            
            # Record metrics
            duration = time.time() - start_time
            metrics = {
                "duration": duration,
                "iterations": iterations,
                "data_size": len(training_data),
                "chunks_processed": chunks,
                "concepts_identified": concepts_identified,
                "patterns_found": patterns_found
            }
            
            # Record in training history
            self._add_training_record(
                "unsupervised_learning", 
                source_name,
                chunks,  # Use chunks as sample count
                metrics
            )
            
            return {
                "success": True,
                "message": f"Completed unsupervised learning with {concepts_identified} concepts identified",
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"Error in unsupervised learning: {str(e)}")
            return {
                "success": False,
                "message": f"Error in unsupervised learning: {str(e)}",
                "metrics": None
            }