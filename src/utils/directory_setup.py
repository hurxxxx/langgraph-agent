"""
Directory Setup Utility

This module provides utility functions for setting up the directory structure
required by the multi-agent system, particularly for document and image generation.
"""

import os
from typing import List, Dict, Any

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

def setup_document_directories() -> Dict[str, List[str]]:
    """
    Set up all directories required for document generation.

    Returns:
        Dict[str, List[str]]: Dictionary of created directories
    """
    # Define base directories
    base_dir = "./generated_documents"

    # Define document types and their directories
    document_types = [
        "reports",
        "blogs",
        "academic",
        "proposals",
        "plans"
    ]

    created_dirs = {
        "base": [ensure_directory_exists(base_dir)],
        "metadata": [ensure_directory_exists(os.path.join(base_dir, "metadata"))],
        "document_types": []
    }

    # Create directories for each document type
    for doc_type in document_types:
        doc_dir = os.path.join(base_dir, doc_type)
        metadata_dir = os.path.join(doc_dir, "metadata")

        created_dirs["document_types"].append(ensure_directory_exists(doc_dir))
        created_dirs["document_types"].append(ensure_directory_exists(metadata_dir))

    return created_dirs

def setup_image_directories() -> Dict[str, List[str]]:
    """
    Set up all directories required for image generation.

    Returns:
        Dict[str, List[str]]: Dictionary of created directories
    """
    # Define base directories
    base_dir = "./generated_images"
    metadata_dir = os.path.join(base_dir, "metadata")

    created_dirs = {
        "base": [ensure_directory_exists(base_dir)],
        "metadata": [ensure_directory_exists(metadata_dir)]
    }

    return created_dirs

def setup_vector_db_directories() -> Dict[str, List[str]]:
    """
    Set up all directories required for vector databases.

    Returns:
        Dict[str, List[str]]: Dictionary of created directories
    """
    # Define base directories
    base_dir = "./vector_db"

    created_dirs = {
        "base": [ensure_directory_exists(base_dir)]
    }

    return created_dirs

def setup_all_directories(verbose: bool = True) -> Dict[str, Dict[str, List[str]]]:
    """
    Set up all directories required by the multi-agent system.

    Args:
        verbose: Whether to print information about created directories

    Returns:
        Dict[str, Dict[str, List[str]]]: Dictionary of all created directories
    """
    all_dirs = {
        "documents": setup_document_directories(),
        "images": setup_image_directories(),
        "vector_db": setup_vector_db_directories()
    }

    if verbose:
        print("Directory setup complete:")
        print(f"- Document directories: {len(all_dirs['documents']['document_types'])} created")
        print(f"- Image directories: {len(all_dirs['images']['base']) + len(all_dirs['images']['metadata'])} created")
        print(f"- Vector DB directories: {len(all_dirs['vector_db']['base'])} created")

    return all_dirs

if __name__ == "__main__":
    # If run directly, set up all directories
    print("Setting up all required directories...")
    dirs = setup_all_directories(verbose=True)
    print("\nDirectory setup complete!")
    print(f"Created {len(dirs['documents']['base']) + len(dirs['documents']['metadata']) + len(dirs['documents']['document_types'])} document directories")
    print(f"Created {len(dirs['images']['base']) + len(dirs['images']['metadata'])} image directories")
    print(f"Created {len(dirs['vector_db']['base'])} vector DB directories")
