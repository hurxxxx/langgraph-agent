"""
Test Caching Utility

This script tests the caching utility module, including:
- Memory cache
- Disk cache
- Cache decorator
"""

import os
import sys
import time
import shutil
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import caching utilities
from src.utils.caching import MemoryCache, DiskCache, cache_result


def test_memory_cache():
    """
    Test the memory cache.
    """
    print("Testing memory cache...")
    
    # Create a memory cache with a short TTL
    cache = MemoryCache(max_size=10, ttl=2)
    
    # Set some values
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    
    # Get values
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    assert cache.get("key4") is None
    
    print("Basic get/set operations: PASSED")
    
    # Test TTL
    print("Testing TTL (waiting 3 seconds)...")
    time.sleep(3)
    
    assert cache.get("key1") is None
    assert cache.get("key2") is None
    assert cache.get("key3") is None
    
    print("TTL expiration: PASSED")
    
    # Test max size
    cache = MemoryCache(max_size=2, ttl=0)
    
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    
    # Access key1 to make it more recently used
    cache.get("key1")
    
    # Add a third key, which should evict key2
    cache.set("key3", "value3")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None
    assert cache.get("key3") == "value3"
    
    print("Max size limit: PASSED")
    
    # Test invalidation
    cache = MemoryCache(ttl=0)
    
    cache.set("prefix_key1", "value1")
    cache.set("prefix_key2", "value2")
    cache.set("other_key", "value3")
    
    cache.invalidate("prefix")
    
    assert cache.get("prefix_key1") is None
    assert cache.get("prefix_key2") is None
    assert cache.get("other_key") == "value3"
    
    print("Pattern invalidation: PASSED")
    
    print("Memory cache tests: ALL PASSED")


def test_disk_cache():
    """
    Test the disk cache.
    """
    print("\nTesting disk cache...")
    
    # Create a temporary cache directory
    cache_dir = "./test_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    # Create a disk cache with a short TTL
    cache = DiskCache(cache_dir=cache_dir, ttl=2)
    
    # Set some values
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    
    # Get values
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    assert cache.get("key4") is None
    
    print("Basic get/set operations: PASSED")
    
    # Test TTL
    print("Testing TTL (waiting 3 seconds)...")
    time.sleep(3)
    
    assert cache.get("key1") is None
    assert cache.get("key2") is None
    assert cache.get("key3") is None
    
    print("TTL expiration: PASSED")
    
    # Test invalidation
    cache = DiskCache(cache_dir=cache_dir, ttl=0)
    
    # Set some values
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    
    # Invalidate all
    cache.invalidate()
    
    assert cache.get("key1") is None
    assert cache.get("key2") is None
    
    print("Full invalidation: PASSED")
    
    # Clean up
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    print("Disk cache tests: ALL PASSED")


def test_cache_decorator():
    """
    Test the cache decorator.
    """
    print("\nTesting cache decorator...")
    
    # Create a counter to track function calls
    call_count = {"count": 0}
    
    # Create a function that increments the counter
    @cache_result(ttl=2)
    def expensive_function(arg1, arg2):
        call_count["count"] += 1
        return f"Result: {arg1} + {arg2} = {arg1 + arg2}"
    
    # Call the function multiple times with the same arguments
    result1 = expensive_function(1, 2)
    result2 = expensive_function(1, 2)
    result3 = expensive_function(1, 2)
    
    # Check that the function was only called once
    assert call_count["count"] == 1
    assert result1 == result2 == result3
    
    print("Basic caching: PASSED")
    
    # Call the function with different arguments
    result4 = expensive_function(3, 4)
    
    # Check that the function was called again
    assert call_count["count"] == 2
    assert result4 != result1
    
    print("Different arguments: PASSED")
    
    # Test TTL
    print("Testing TTL (waiting 3 seconds)...")
    time.sleep(3)
    
    # Call the function again with the same arguments
    result5 = expensive_function(1, 2)
    
    # Check that the function was called again
    assert call_count["count"] == 3
    assert result5 == result1
    
    print("TTL expiration: PASSED")
    
    print("Cache decorator tests: ALL PASSED")


def main():
    """Main function to run the tests."""
    # Load environment variables
    load_dotenv()
    
    # Run the tests
    test_memory_cache()
    test_disk_cache()
    test_cache_decorator()
    
    print("\nAll caching tests passed!")


if __name__ == "__main__":
    main()
