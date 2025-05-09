#!/usr/bin/env python3
"""
Run all supervisor tests to verify functionality.

This script runs all the supervisor tests to verify that the supervisor:
1. Analyzes user prompts correctly
2. Creates appropriate execution plans
3. Delegates tasks to the right agents
4. Monitors task execution and adapts as needed
5. Integrates results from multiple agents
6. Delivers coherent final responses

Usage:
    python run_supervisor_tests.py [--verbose] [--specific-test TEST_NAME]

Options:
    --verbose           Show detailed test output
    --specific-test     Run only a specific test (e.g., "test_mcp_supervisor_complex_query")
"""

import os
import sys
import unittest
import argparse
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import test modules
from tests.test_supervisor import TestSupervisor, TestSupervisorConfig
from tests.test_integration import TestSupervisorIntegration
from tests.test_supervisor_e2e import TestSupervisorE2E
from tests.test_supervisor_reactivity import TestSupervisorReactivity
from tests.test_streaming import TestSupervisorStreaming


def run_tests(verbose=False, specific_test=None):
    """
    Run all supervisor tests or a specific test.

    Args:
        verbose: Whether to show detailed test output
        specific_test: Name of a specific test to run
    """
    # Load environment variables
    load_dotenv()

    # Create test suite
    suite = unittest.TestSuite()

    if specific_test:
        # Find and run only the specified test
        for test_class in [TestSupervisor, TestSupervisorConfig, TestSupervisorIntegration, TestSupervisorE2E, TestSupervisorReactivity, TestSupervisorStreaming]:
            for method_name in dir(test_class):
                if method_name.startswith('test_') and (method_name == specific_test or f"{test_class.__name__}.{method_name}" == specific_test):
                    suite.addTest(test_class(method_name))
    else:
        # Add all test classes
        suite.addTest(unittest.makeSuite(TestSupervisorConfig))
        suite.addTest(unittest.makeSuite(TestSupervisor))
        suite.addTest(unittest.makeSuite(TestSupervisorIntegration))
        suite.addTest(unittest.makeSuite(TestSupervisorE2E))
        suite.addTest(unittest.makeSuite(TestSupervisorReactivity))
        suite.addTest(unittest.makeSuite(TestSupervisorStreaming))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    # Return success/failure
    return result.wasSuccessful()


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run supervisor tests")
    parser.add_argument("--verbose", action="store_true", help="Show detailed test output")
    parser.add_argument("--specific-test", help="Run only a specific test")
    args = parser.parse_args()

    success = run_tests(verbose=args.verbose, specific_test=args.specific_test)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
