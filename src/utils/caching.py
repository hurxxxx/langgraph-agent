"""
Caching Utility Module

This module provides caching mechanisms to improve agent performance,
including in-memory caching, disk-based caching, and cache invalidation strategies.
"""

import os
import json
import time
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps
from pathlib import Path


class MemoryCache:
    """
    Simple in-memory cache with TTL support.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the memory cache.
        
        Args:
            max_size: Maximum number of items in the cache
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value, or None if not found or expired
        """
        # Check if key exists
        if key not in self.cache:
            return None
        
        # Check if item has expired
        if self.ttl > 0:
            timestamp = self.access_times.get(key, 0)
            if time.time() - timestamp > self.ttl:
                # Item has expired
                self._remove(key)
                return None
        
        # Update access time
        self.access_times[key] = time.time()
        
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Check if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used item
            self._remove_lru()
        
        # Set item
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _remove(self, key: str) -> None:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
        
        if key in self.access_times:
            del self.access_times[key]
    
    def _remove_lru(self) -> None:
        """
        Remove the least recently used item from the cache.
        """
        if not self.access_times:
            return
        
        # Find the least recently used item
        lru_key = min(self.access_times, key=self.access_times.get)
        
        # Remove it
        self._remove(lru_key)
    
    def clear(self) -> None:
        """
        Clear the cache.
        """
        self.cache.clear()
        self.access_times.clear()
    
    def invalidate(self, pattern: str = None) -> None:
        """
        Invalidate cache items matching a pattern.
        
        Args:
            pattern: Pattern to match (if None, clear all)
        """
        if pattern is None:
            self.clear()
            return
        
        # Find keys matching the pattern
        keys_to_remove = [key for key in self.cache if pattern in key]
        
        # Remove them
        for key in keys_to_remove:
            self._remove(key)
    
    def __len__(self) -> int:
        """
        Get the number of items in the cache.
        
        Returns:
            int: Number of items
        """
        return len(self.cache)


class DiskCache:
    """
    Disk-based cache with TTL support.
    """
    
    def __init__(self, cache_dir: str = "./.cache", ttl: int = 3600):
        """
        Initialize the disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.cache_dir = os.path.abspath(cache_dir)
        self.ttl = ttl
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """
        Get the path to a cache file.
        
        Args:
            key: Cache key
            
        Returns:
            str: Path to cache file
        """
        # Hash the key to create a filename
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        
        return os.path.join(self.cache_dir, f"{hashed_key}.cache")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value, or None if not found or expired
        """
        cache_path = self._get_cache_path(key)
        
        # Check if file exists
        if not os.path.exists(cache_path):
            return None
        
        # Check if file has expired
        if self.ttl > 0:
            modified_time = os.path.getmtime(cache_path)
            if time.time() - modified_time > self.ttl:
                # File has expired
                os.remove(cache_path)
                return None
        
        # Load cached value
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            # If there's an error loading the cache, remove it
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)
        
        # Save value to cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Error saving to disk cache: {str(e)}")
    
    def invalidate(self, pattern: str = None) -> None:
        """
        Invalidate cache items matching a pattern.
        
        Args:
            pattern: Pattern to match (if None, clear all)
        """
        if pattern is None:
            # Clear all cache files
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".cache"):
                    os.remove(os.path.join(self.cache_dir, filename))
            return
        
        # Find keys matching the pattern
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".cache"):
                cache_path = os.path.join(self.cache_dir, filename)
                
                # Load the key from the cache
                try:
                    with open(cache_path, "rb") as f:
                        cached_data = pickle.load(f)
                        
                    # Check if the key matches the pattern
                    if hasattr(cached_data, "key") and pattern in cached_data.key:
                        os.remove(cache_path)
                except Exception:
                    # If there's an error loading the cache, remove it
                    os.remove(cache_path)


def cache_result(cache=None, ttl=3600, key_fn=None):
    """
    Decorator to cache function results.
    
    Args:
        cache: Cache instance to use (if None, creates a new MemoryCache)
        ttl: Time-to-live in seconds
        key_fn: Function to generate cache key (if None, uses function name and args)
        
    Returns:
        Callable: Decorated function
    """
    if cache is None:
        cache = MemoryCache(ttl=ttl)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                # Default key is function name + args + kwargs
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result)
            
            return result
        
        return wrapper
    
    return decorator
