# UI Testing with Playwright

This document describes the UI testing implementation using Playwright for the multi-agent supervisor system's Streamlit interface.

## Overview

Playwright is used to automate UI testing of the Streamlit interface, allowing for consistent, repeatable testing of the system's functionality through its user interface. The tests simulate real user interactions with the system, testing end-to-end functionality across different scenarios.

## Test Implementation

### Test Files

- **`ui/streamlit/test_ui.py`**: Basic UI tests for the Streamlit interface
- **`ui/streamlit/test_prompt_input.py`**: Tests focused on direct prompt input and response verification
- **`ui/streamlit/run_ui_tests.py`**: Script to run the Streamlit app and execute tests

### Test Scenarios

The UI tests cover the following scenarios:

1. **Basic UI Functionality**
   - Verifying page title and structure
   - Testing sidebar elements and configuration options
   - Checking tab navigation and content

2. **Simple Query Execution**
   - Inputting a simple search query
   - Configuring for a single agent
   - Verifying results and agent outputs

3. **Complex Multi-Agent Queries**
   - Inputting queries that require multiple agents
   - Testing coordination between search and image generation
   - Verifying that all agents are properly utilized

4. **Parallel Supervisor Testing**
   - Testing the parallel execution of agents
   - Verifying that tasks are properly parallelized
   - Checking result integration from parallel tasks

5. **Quality Evaluation**
   - Testing the quality evaluation form
   - Submitting evaluations for test results
   - Verifying evaluation storage and display

## Running the Tests

### Prerequisites

- Python 3.8+
- Streamlit
- Playwright

### Installation

```bash
pip install streamlit playwright pytest pytest-asyncio
playwright install
```

### Running Tests

To run all UI tests:

```bash
cd ui/streamlit
python run_ui_tests.py
```

This script will:
1. Start the Streamlit app in a separate process
2. Wait for the app to be available
3. Run the Playwright tests
4. Capture screenshots during test execution
5. Terminate the Streamlit app when tests complete

To run specific test files:

```bash
# Run basic UI tests
python -m pytest test_ui.py -v

# Run prompt input tests
python -m pytest test_prompt_input.py -v
```

## Test Documentation

### Screenshots

The tests automatically capture screenshots at key points in the test execution:

- Configuration state before test execution
- Processing state during test execution
- Results after test completion
- Agent outputs and details

Screenshots are saved to the `test_screenshots` directory with timestamps and test names for easy identification.

### Test Reports

Test results are reported in the console output, showing:

- Test execution status (pass/fail)
- Execution time
- Error messages for failed tests
- Screenshot locations

## Test Scenarios in Detail

### Simple Search Prompt Test

This test verifies that the system can process a simple search query:

1. Inputs a query about quantum computing
2. Configures the system to use only the search agent
3. Verifies that the search agent is called and returns results
4. Checks that the results are properly displayed in the UI

### Complex Multi-Agent Prompt Test

This test verifies that the system can handle complex queries requiring multiple agents:

1. Inputs a query about quantum computing and requests an image
2. Configures the system to use both search and image generation agents
3. Verifies that both agents are called and return results
4. Checks that the results from both agents are properly integrated and displayed

### Parallel Supervisor Test

This test verifies that the parallel supervisor can execute tasks concurrently:

1. Inputs a query that can be parallelized
2. Configures the system to use the parallel supervisor
3. Verifies that tasks are executed concurrently
4. Checks that the results are properly integrated and displayed

## Troubleshooting

Common issues and solutions:

- **Tests Failing to Connect**: Ensure the Streamlit app is running on port 8501
- **Timeout Errors**: Increase the timeout value in the test constants
- **Element Not Found**: Check if the UI structure has changed and update selectors
- **Screenshot Directory Missing**: Ensure the test_screenshots directory exists

## Future Improvements

Planned improvements for the UI testing:

1. **More Comprehensive Test Coverage**
   - Add tests for all agent combinations
   - Test error handling and recovery
   - Test with various query complexities

2. **Performance Testing**
   - Measure and report response times
   - Test system under load with multiple concurrent requests

3. **Visual Regression Testing**
   - Compare screenshots to detect UI changes
   - Automate visual verification of UI elements

4. **Integration with CI/CD**
   - Run UI tests automatically on code changes
   - Generate and publish test reports
