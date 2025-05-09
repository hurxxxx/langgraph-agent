"""
File Operations Utility Module

This module provides utility functions for file operations, including:
- Downloading files from URLs
- Saving files to disk
- Creating directories
- Handling file paths
"""

import os
import requests
import uuid
import json
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import shutil


def ensure_directory_exists(directory_path: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory

    Returns:
        str: Absolute path to the directory
    """
    # Convert to absolute path
    abs_path = os.path.abspath(directory_path)
    
    # Create directory if it doesn't exist
    os.makedirs(abs_path, exist_ok=True)
    
    return abs_path


def download_file(url: str, save_path: str, timeout: int = 30) -> str:
    """
    Download a file from a URL and save it to disk.

    Args:
        url: URL to download from
        save_path: Path to save the file to
        timeout: Timeout in seconds

    Returns:
        str: Path to the downloaded file
    """
    # Ensure the directory exists
    directory = os.path.dirname(save_path)
    ensure_directory_exists(directory)
    
    # Download the file
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return save_path
    except Exception as e:
        raise Exception(f"Error downloading file from {url}: {str(e)}")


def save_metadata(metadata: Dict[str, Any], file_path: str) -> str:
    """
    Save metadata to a JSON file.

    Args:
        metadata: Metadata to save
        file_path: Path to save the metadata to

    Returns:
        str: Path to the metadata file
    """
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    ensure_directory_exists(directory)
    
    # Save the metadata
    try:
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return file_path
    except Exception as e:
        raise Exception(f"Error saving metadata to {file_path}: {str(e)}")


def load_metadata(file_path: str) -> Dict[str, Any]:
    """
    Load metadata from a JSON file.

    Args:
        file_path: Path to the metadata file

    Returns:
        Dict[str, Any]: Loaded metadata
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error loading metadata from {file_path}: {str(e)}")


def generate_unique_filename(base_dir: str, prefix: str = "", extension: str = "") -> str:
    """
    Generate a unique filename in the specified directory.

    Args:
        base_dir: Base directory
        prefix: Prefix for the filename
        extension: File extension (with or without dot)

    Returns:
        str: Path to the unique filename
    """
    # Ensure the directory exists
    ensure_directory_exists(base_dir)
    
    # Add dot to extension if needed
    if extension and not extension.startswith('.'):
        extension = f".{extension}"
    
    # Generate unique filename
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{prefix}_{timestamp}_{unique_id}{extension}"
    
    return os.path.join(base_dir, filename)


def verify_file_exists(file_path: str) -> bool:
    """
    Verify that a file exists.

    Args:
        file_path: Path to the file

    Returns:
        bool: True if the file exists, False otherwise
    """
    return os.path.isfile(file_path)


def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.

    Args:
        file_path: Path to the file

    Returns:
        int: Size of the file in bytes
    """
    if not verify_file_exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return os.path.getsize(file_path)


def get_file_extension(url: str) -> str:
    """
    Get the file extension from a URL.

    Args:
        url: URL to extract extension from

    Returns:
        str: File extension (with dot)
    """
    # Parse the URL path
    path = requests.utils.urlparse(url).path
    
    # Get the extension
    extension = os.path.splitext(path)[1]
    
    # If no extension, default to .jpg for images
    if not extension:
        extension = ".jpg"
    
    return extension
