#!/usr/bin/env python
"""
Cleanup Generated Files Script

This script deletes all generated files (documents, images, etc.) while preserving
the directory structure required by the multi-agent system.
"""

import os
import sys
import shutil
import argparse
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.utils.directory_setup import setup_all_directories

def count_files(directory: str) -> int:
    """
    Count the number of files in a directory and its subdirectories.

    Args:
        directory: Path to the directory

    Returns:
        int: Number of files
    """
    count = 0
    for root, _, files in os.walk(directory):
        count += len(files)
    return count

def delete_files_in_directory(directory: str, verbose: bool = True) -> int:
    """
    Delete all files in a directory and its subdirectories while preserving the directory structure.

    Args:
        directory: Path to the directory
        verbose: Whether to print information about deleted files

    Returns:
        int: Number of files deleted
    """
    if not os.path.exists(directory):
        if verbose:
            print(f"Directory does not exist: {directory}")
        return 0

    file_count = count_files(directory)
    
    if file_count == 0:
        if verbose:
            print(f"No files to delete in {directory}")
        return 0
    
    # Delete all files but keep directories
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                if verbose and file_count <= 10:  # Only print details for a small number of files
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
    
    if verbose:
        print(f"Deleted {file_count} files from {directory}")
    
    return file_count

def cleanup_generated_files(verbose: bool = True) -> Dict[str, int]:
    """
    Delete all generated files while preserving the directory structure.

    Args:
        verbose: Whether to print information about deleted files

    Returns:
        Dict[str, int]: Dictionary with the number of files deleted from each directory
    """
    # Define directories to clean
    directories = [
        "./generated_documents",
        "./generated_images",
        "./vector_db"
    ]
    
    # Count and delete files
    deleted_counts = {}
    for directory in directories:
        deleted_counts[directory] = delete_files_in_directory(directory, verbose)
    
    # Recreate directory structure
    setup_all_directories(verbose=False)
    
    return deleted_counts

def main():
    """Main function to run the cleanup script."""
    parser = argparse.ArgumentParser(description="Delete all generated files while preserving directory structure.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    print("Cleaning up generated files...")
    deleted_counts = cleanup_generated_files(verbose=verbose)
    
    total_deleted = sum(deleted_counts.values())
    print(f"\nSummary: Deleted {total_deleted} files in total")
    for directory, count in deleted_counts.items():
        print(f"- {directory}: {count} files")
    
    print("\nDirectory structure has been preserved and is ready for use.")

if __name__ == "__main__":
    main()
