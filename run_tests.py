#!/usr/bin/env python3
"""
Test Runner Script

This script runs all unit tests for the project.
"""

import os
import sys
import unittest
import argparse


def run_all_tests():
    """Run all tests in the tests directory."""
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return the number of failures and errors
    return len(result.failures) + len(result.errors)


def run_specific_test(test_name):
    """Run a specific test module."""
    # Check if the test file exists
    test_file = f'tests/test_{test_name}.py'
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found.")
        return 1
    
    # Import the test module
    module_name = f'tests.test_{test_name}'
    __import__(module_name)
    
    # Get the module
    module = sys.modules[module_name]
    
    # Run the tests in the module
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromModule(module)
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return the number of failures and errors
    return len(result.failures) + len(result.errors)


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run unit tests for the project.')
    parser.add_argument('--test', help='Run a specific test module (without the "test_" prefix)')
    
    args = parser.parse_args()
    
    # Run tests
    if args.test:
        return run_specific_test(args.test)
    else:
        return run_all_tests()


if __name__ == '__main__':
    sys.exit(main())
