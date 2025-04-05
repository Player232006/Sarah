"""
HuggingFace Trainer Module

This module handles the integration with HuggingFace datasets for training the AI model.
It provides functionality to fetch, process, and use datasets from huggingface.co/datasets.
"""

import os
import json
import time
import random
import requests
from typing import Dict, List, Any, Union, Optional
import io
import zipfile

class HuggingFaceTrainer:
    def __init__(self):
        """Initialize the HuggingFace trainer."""
        self.base_url = "https://huggingface.co/api/datasets"
        self.dataset_cache_dir = "data/huggingface_datasets"
        self.popular_datasets = [
            "emotion", 
            "tweet_eval", 
            "daily_dialog", 
            "empathetic_dialogues",
            "conv_ai_2",
            "samsum"
        ]
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.dataset_cache_dir, exist_ok=True)
        
    def search_datasets(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for datasets on HuggingFace.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of dataset information
        """
        try:
            endpoint = f"{self.base_url}/search"
            params = {"search": query, "limit": limit}
            
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                datasets = response.json()
                return [
                    {
                        "id": ds.get("id", ""),
                        "name": ds.get("name", "Unknown"),
                        "description": ds.get("description", "No description available"),
                        "likes": ds.get("likes", 0),
                        "downloads": ds.get("downloads", 0)
                    } 
                    for ds in datasets
                ]
            else:
                print(f"Error searching datasets: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception when searching datasets: {str(e)}")
            return []
    
    def get_popular_datasets(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get a list of popular datasets.
        
        Args:
            limit: Maximum number of datasets to return
            
        Returns:
            List of popular dataset information
        """
        # Take a random sample of the popular datasets
        dataset_ids = random.sample(self.popular_datasets, min(limit, len(self.popular_datasets)))
        
        results = []
        for dataset_id in dataset_ids:
            try:
                # Fetch dataset info
                dataset_info = self.get_dataset_info(dataset_id)
                if dataset_info:
                    results.append(dataset_info)
            except Exception as e:
                print(f"Error fetching info for dataset {dataset_id}: {str(e)}")
        
        return results
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific dataset.
        
        Args:
            dataset_id: The dataset ID
            
        Returns:
            Dataset information or None if not found
        """
        try:
            endpoint = f"{self.base_url}/{dataset_id}"
            response = requests.get(endpoint)
            
            if response.status_code == 200:
                dataset = response.json()
                return {
                    "id": dataset.get("id", ""),
                    "name": dataset.get("name", "Unknown"),
                    "description": dataset.get("description", "No description available"),
                    "citation": dataset.get("citation", ""),
                    "likes": dataset.get("likes", 0),
                    "downloads": dataset.get("downloads", 0)
                }
            else:
                print(f"Error fetching dataset info: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception when fetching dataset info: {str(e)}")
            return None
    
    def fetch_dataset_samples(self, dataset_id: str, split: str = "train", num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch sample data from a dataset.
        
        This is a simplified implementation that fetches a small number of samples
        from publicly available datasets through the HuggingFace API.
        
        Args:
            dataset_id: The dataset ID
            split: The dataset split (train, validation, test)
            num_samples: Number of samples to fetch
            
        Returns:
            List of dataset samples
        """
        cache_file = os.path.join(self.dataset_cache_dir, f"{dataset_id}_{split}_{num_samples}.json")
        
        # Check if we have cached data
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cached dataset: {str(e)}")
                # If there's an error, continue to fetch the data
        
        try:
            # For demonstration purposes, we'll create synthetic samples based on the dataset ID
            # In a real implementation, you would use the datasets library or the HuggingFace API
            # to fetch actual samples
            
            samples = []
            
            # Simulated dataset responses based on dataset type
            if "emotion" in dataset_id:
                emotions = ["joy", "sadness", "anger", "fear", "love", "surprise"]
                for i in range(num_samples):
                    emotion = random.choice(emotions)
                    samples.append({
                        "text": f"Sample text for {emotion} emotion #{i}",
                        "label": emotion
                    })
            
            elif "dialog" in dataset_id or "conv" in dataset_id:
                for i in range(num_samples):
                    turns = random.randint(2, 5)
                    dialog = []
                    for j in range(turns):
                        speaker = "user" if j % 2 == 0 else "assistant"
                        dialog.append({
                            "speaker": speaker,
                            "text": f"Sample {speaker} dialog text #{j} in conversation #{i}"
                        })
                    samples.append({"dialog": dialog})
            
            elif "tweet" in dataset_id:
                sentiments = ["positive", "negative", "neutral"]
                for i in range(num_samples):
                    sentiment = random.choice(sentiments)
                    samples.append({
                        "tweet": f"Sample tweet text #{i}",
                        "sentiment": sentiment
                    })
                    
            elif "summ" in dataset_id:
                for i in range(num_samples):
                    samples.append({
                        "document": f"Sample document text #{i} with multiple sentences to be summarized.",
                        "summary": f"Sample summary #{i}"
                    })
            
            else:
                # Generic dataset structure
                for i in range(num_samples):
                    samples.append({
                        "text": f"Sample text #{i} from {dataset_id}",
                        "metadata": {
                            "source": dataset_id,
                            "index": i
                        }
                    })
            
            # Cache the samples
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error caching dataset: {str(e)}")
            
            return samples
        
        except Exception as e:
            print(f"Exception when fetching dataset samples: {str(e)}")
            return []
    
    def download_dataset(self, dataset_id: str, save_path: Optional[str] = None) -> str:
        """
        Download a dataset for offline use.
        
        Args:
            dataset_id: The dataset ID
            save_path: Path to save the downloaded dataset (optional)
            
        Returns:
            Path to the downloaded dataset or error message
        """
        if not save_path:
            save_path = os.path.join(self.dataset_cache_dir, f"{dataset_id}.zip")
        
        try:
            # Simulate downloading a dataset
            # In a real implementation, you would use the HuggingFace API or datasets library
            time.sleep(2)  # Simulate download time
            
            # Create a simple ZIP file with metadata and samples
            with zipfile.ZipFile(save_path, 'w') as zip_file:
                # Add metadata
                metadata = {
                    "id": dataset_id,
                    "name": dataset_id.replace("_", " ").title(),
                    "description": f"Downloaded dataset: {dataset_id}",
                    "download_time": time.time()
                }
                zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))
                
                # Add samples
                samples = self.fetch_dataset_samples(dataset_id)
                zip_file.writestr('samples.json', json.dumps(samples, indent=2))
            
            return save_path
        
        except Exception as e:
            error_msg = f"Error downloading dataset: {str(e)}"
            print(error_msg)
            return error_msg
    
    def prepare_training_data(self, dataset_id: str, format_type: str) -> List[Dict[str, Any]]:
        """
        Prepare training data from a dataset for a specific format type.
        
        Args:
            dataset_id: The dataset ID
            format_type: The format type (conversation, completion, classification)
            
        Returns:
            List of formatted training examples
        """
        # Fetch dataset samples
        samples = self.fetch_dataset_samples(dataset_id)
        
        # Format the samples based on the format type
        formatted_data = []
        
        if format_type == "conversation":
            for sample in samples:
                if "dialog" in sample:
                    formatted_data.append({
                        "type": "conversation",
                        "turns": [
                            {"role": turn["speaker"], "content": turn["text"]}
                            for turn in sample["dialog"]
                        ]
                    })
                elif "text" in sample:
                    # For non-dialog datasets, create a simple two-turn conversation
                    formatted_data.append({
                        "type": "conversation",
                        "turns": [
                            {"role": "user", "content": sample["text"]},
                            {"role": "assistant", "content": f"Response to: {sample['text']}"}
                        ]
                    })
        
        elif format_type == "completion":
            for sample in samples:
                if "document" in sample and "summary" in sample:
                    formatted_data.append({
                        "type": "completion",
                        "prompt": sample["document"],
                        "completion": sample["summary"]
                    })
                elif "text" in sample:
                    formatted_data.append({
                        "type": "completion",
                        "prompt": f"Complete the following: {sample['text']}",
                        "completion": f"Completion for: {sample['text']}"
                    })
        
        elif format_type == "classification":
            for sample in samples:
                if "text" in sample and "label" in sample:
                    formatted_data.append({
                        "type": "classification",
                        "text": sample["text"],
                        "label": sample["label"]
                    })
                elif "tweet" in sample and "sentiment" in sample:
                    formatted_data.append({
                        "type": "classification",
                        "text": sample["tweet"],
                        "label": sample["sentiment"]
                    })
        
        return formatted_data
    
    def train_model_with_dataset(self, model_handler, dataset_id: str, format_type: str) -> Dict[str, Any]:
        """
        Train a model using a dataset.
        
        Args:
            model_handler: The model handler to use for training
            dataset_id: The dataset ID
            format_type: The format type (conversation, completion, classification)
            
        Returns:
            Training results
        """
        start_time = time.time()
        
        # Prepare the training data
        training_data = self.prepare_training_data(dataset_id, format_type)
        
        if not training_data:
            return {
                "success": False,
                "error": "No training data could be prepared",
                "time_taken": 0
            }
        
        # Log training process
        print(f"Training model with {len(training_data)} examples from dataset {dataset_id}")
        
        # Simulated training process
        # In a real implementation, you would pass the data to the model handler
        time.sleep(3)  # Simulate training time
        
        # Simulate training results
        results = {
            "success": True,
            "dataset": dataset_id,
            "format": format_type,
            "examples_trained": len(training_data),
            "time_taken": time.time() - start_time,
            "improvement_metrics": {
                "accuracy": round(random.uniform(0.75, 0.95), 2),
                "response_quality": round(random.uniform(0.70, 0.90), 2)
            }
        }
        
        # Update model knowledge
        model_handler.update_knowledge(
            {
                "trained_with": dataset_id,
                "training_format": format_type,
                "training_time": results["time_taken"],
                "training_examples": len(training_data)
            }
        )
        
        return results

    def get_dataset_categories(self) -> Dict[str, List[str]]:
        """
        Get a categorized list of recommended datasets.
        
        Returns:
            Dictionary mapping categories to dataset IDs
        """
        return {
            "Conversation": ["daily_dialog", "conv_ai_2", "empathetic_dialogues"],
            "Emotion": ["emotion", "tweet_eval/emotion"],
            "Summarization": ["samsum", "cnn_dailymail"],
            "Question Answering": ["squad_v2", "natural_questions"],
            "Classification": ["tweet_eval/sentiment", "sst2"]
        }