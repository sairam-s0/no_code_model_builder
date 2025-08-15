# backend/dataset_utils.py

# This module handles all dataset-related tasks: downloading from sources
# like Kaggle, locating built-in datasets, and preparing custom datasets.

import subprocess
import os
import zipfile
import time

def load_dataset(source, dataset_info, status_callback):
    """
    Loads or downloads a dataset based on the specified source.

    Args:
        source (str): The source of the dataset ("Built-in", "Kaggle", "Custom").
        dataset_info (str): The name, ID, or path of the dataset.
        status_callback (function): A function to send status updates back to the GUI.

    Returns:
        str: The local path to the prepared dataset.
    """
    status_callback(f"[Dataset Utils] Preparing dataset from source: {source}")
    
    base_path = os.path.join(os.getcwd(), "datasets")
    os.makedirs(base_path, exist_ok=True)

    if source == "Built-in":
        # In a real app, you might use libraries like `torchvision.datasets`
        # or `datasets` from Hugging Face to automatically handle this.
        status_callback(f"[Dataset Utils] Using built-in dataset: {dataset_info}")
        time.sleep(1) # Simulate setup
        dataset_path = os.path.join(base_path, dataset_info)
        os.makedirs(dataset_path, exist_ok=True) # Ensure directory exists
        status_callback(f"[Dataset Utils] Built-in dataset ready at: {dataset_path}")
        return dataset_path

    elif source == "Kaggle":
        status_callback(f"[Dataset Utils] Preparing to download from Kaggle: {dataset_info}")
        
        # Kaggle API requires the dataset ID in the format "username/dataset-name"
        if len(dataset_info.split('/')) != 2:
            raise ValueError("Invalid Kaggle dataset ID. Must be in 'username/dataset-name' format.")
            
        dataset_name = dataset_info.split('/')[1]
        dataset_path = os.path.join(base_path, dataset_name)
        zip_path = f"{dataset_path}.zip"
        
        # Command to download the dataset using the Kaggle API
        command = [
            "kaggle", "datasets", "download",
            "-d", dataset_info,
            "-p", base_path,
            "--unzip", # Automatically unzip the file
            "--force"  # Overwrite if it exists
        ]
        
        try:
            status_callback(f"[Dataset Utils] Executing Kaggle command: {' '.join(command)}")
            # Using subprocess.run to execute the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            status_callback("[Dataset Utils] Kaggle download and extraction successful.")
            status_callback(f"[Dataset Utils] Dataset available at: {dataset_path}")
            return dataset_path
        except FileNotFoundError:
            raise FileNotFoundError("Kaggle API not found. Please ensure 'kaggle' is installed and in your system's PATH.")
        except subprocess.CalledProcessError as e:
            # This catches errors from the Kaggle command itself (e.g., dataset not found, auth issues)
            error_message = e.stderr or e.stdout
            raise RuntimeError(f"Kaggle API error: {error_message}")

    elif source == "Custom":
        if not os.path.isdir(dataset_info):
            raise FileNotFoundError(f"Custom dataset path does not exist or is not a directory: {dataset_info}")
        status_callback(f"[Dataset Utils] Using custom dataset at: {dataset_info}")
        return dataset_info

    else:
        raise ValueError(f"Invalid dataset source provided: {source}")

