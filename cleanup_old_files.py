#!/usr/bin/env python
"""
Cleanup Old Files Script

This script removes old files that are no longer needed after the refactoring.
It preserves the directory structure and essential files.
"""

import os
import sys
import shutil
import argparse
from typing import List, Dict, Any

def get_files_to_remove() -> List[str]:
    """
    Get a list of files to remove.

    Returns:
        List[str]: List of file paths to remove
    """
    files_to_remove = [
        # MCP agents
        "src/agents/mcp_agent.py",
        "src/agents/crew_mcp_agent.py",
        "src/agents/autogen_mcp_agent.py",
        "src/agents/langgraph_mcp_agent.py",
        
        # Old document generation agents
        "src/agents/document_generation.py",
        
        # Old supervisor
        "src/supervisor/parallel_supervisor.py",
        
        # Old documentation
        "docs/learning/langgraph_latest.md",
        "docs/learning/specialized_agents.md",
        "docs/learning/openapi_integration.md",
        "docs/learning/streaming_support.md",
        "docs/learning/serper_integration.md",
        "docs/TASK_LIST.md",
        "docs/DOCUMENT_CATALOG.md",
        
        # Old tests
        "tests/test_image_generation_agent.py",
        "tests/test_image_agent.py",
        "tests/test_react_document_agents.py",
    ]
    
    return files_to_remove

def get_directories_to_remove() -> List[str]:
    """
    Get a list of directories to remove.

    Returns:
        List[str]: List of directory paths to remove
    """
    directories_to_remove = [
        # Old document generation agents
        "src/agents/document_generation",
        
        # Old learning directory
        "docs/learning",
    ]
    
    return directories_to_remove

def remove_files(files: List[str], dry_run: bool = False) -> Dict[str, Any]:
    """
    Remove the specified files.

    Args:
        files: List of file paths to remove
        dry_run: Whether to perform a dry run (don't actually remove files)

    Returns:
        Dict[str, Any]: Results of the operation
    """
    results = {
        "removed": [],
        "not_found": [],
        "errors": []
    }
    
    for file_path in files:
        try:
            if os.path.exists(file_path):
                if not dry_run:
                    os.remove(file_path)
                results["removed"].append(file_path)
            else:
                results["not_found"].append(file_path)
        except Exception as e:
            results["errors"].append({
                "file": file_path,
                "error": str(e)
            })
    
    return results

def remove_directories(directories: List[str], dry_run: bool = False) -> Dict[str, Any]:
    """
    Remove the specified directories.

    Args:
        directories: List of directory paths to remove
        dry_run: Whether to perform a dry run (don't actually remove directories)

    Returns:
        Dict[str, Any]: Results of the operation
    """
    results = {
        "removed": [],
        "not_found": [],
        "errors": []
    }
    
    for directory_path in directories:
        try:
            if os.path.exists(directory_path):
                if not dry_run:
                    shutil.rmtree(directory_path)
                results["removed"].append(directory_path)
            else:
                results["not_found"].append(directory_path)
        except Exception as e:
            results["errors"].append({
                "directory": directory_path,
                "error": str(e)
            })
    
    return results

def rename_files(dry_run: bool = False) -> Dict[str, Any]:
    """
    Rename files.

    Args:
        dry_run: Whether to perform a dry run (don't actually rename files)

    Returns:
        Dict[str, Any]: Results of the operation
    """
    files_to_rename = {
        "README_NEW.md": "README.md",
    }
    
    results = {
        "renamed": [],
        "not_found": [],
        "errors": []
    }
    
    for old_name, new_name in files_to_rename.items():
        try:
            if os.path.exists(old_name):
                if not dry_run:
                    # Backup the original file if it exists
                    if os.path.exists(new_name):
                        backup_name = f"{new_name}.bak"
                        shutil.copy2(new_name, backup_name)
                    
                    # Rename the file
                    os.rename(old_name, new_name)
                
                results["renamed"].append({
                    "old": old_name,
                    "new": new_name
                })
            else:
                results["not_found"].append(old_name)
        except Exception as e:
            results["errors"].append({
                "file": old_name,
                "error": str(e)
            })
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cleanup old files after refactoring")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run (don't actually remove files)")
    args = parser.parse_args()
    
    print(f"{'DRY RUN: ' if args.dry_run else ''}Cleaning up old files...")
    
    # Get files and directories to remove
    files_to_remove = get_files_to_remove()
    directories_to_remove = get_directories_to_remove()
    
    # Remove files
    file_results = remove_files(files_to_remove, args.dry_run)
    print(f"Files removed: {len(file_results['removed'])}")
    print(f"Files not found: {len(file_results['not_found'])}")
    print(f"File errors: {len(file_results['errors'])}")
    
    # Remove directories
    directory_results = remove_directories(directories_to_remove, args.dry_run)
    print(f"Directories removed: {len(directory_results['removed'])}")
    print(f"Directories not found: {len(directory_results['not_found'])}")
    print(f"Directory errors: {len(directory_results['errors'])}")
    
    # Rename files
    rename_results = rename_files(args.dry_run)
    print(f"Files renamed: {len(rename_results['renamed'])}")
    print(f"Files not found for renaming: {len(rename_results['not_found'])}")
    print(f"File renaming errors: {len(rename_results['errors'])}")
    
    # Print details if there are errors
    if file_results["errors"] or directory_results["errors"] or rename_results["errors"]:
        print("\nErrors:")
        for error in file_results["errors"]:
            print(f"  File {error['file']}: {error['error']}")
        for error in directory_results["errors"]:
            print(f"  Directory {error['directory']}: {error['error']}")
        for error in rename_results["errors"]:
            print(f"  Rename {error['file']}: {error['error']}")
    
    print(f"\n{'DRY RUN: ' if args.dry_run else ''}Cleanup complete!")

if __name__ == "__main__":
    main()
