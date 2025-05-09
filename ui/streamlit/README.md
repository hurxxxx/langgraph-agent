# Streamlit Test Interface for Multi-Agent Supervisor

This directory contains a Streamlit-based test interface for the multi-agent supervisor system. The interface allows for testing the system with various configurations, monitoring execution, and evaluating results.

## Overview

The test interface provides the following features:

- **Test Configuration**: Configure the supervisor, agents, and test parameters
- **Execution Monitoring**: Monitor the execution of tests in real-time
- **Result Inspection**: View detailed results from test executions
- **Quality Evaluation**: Evaluate the quality of test results

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Playwright (for UI testing)

### Installation

1. Install the required packages:

```bash
pip install streamlit playwright
```

2. Install Playwright browsers:

```bash
playwright install
```

### Running the Streamlit App

To run the Streamlit app:

```bash
cd ui/streamlit
streamlit run app.py
```

The app will be available at http://localhost:8501

### Running UI Tests

To run the Playwright UI tests:

```bash
cd ui/streamlit
python -m pytest test_ui.py
```

## Features

### Test Configuration

The sidebar allows configuring various aspects of the test:

- **Test Query**: Enter the query to test
- **Supervisor Configuration**: Configure the supervisor type and parameters
  - Standard supervisor with MCP options
  - Parallel supervisor
- **LLM Configuration**: Select the LLM provider and model
- **Agent Selection**: Choose which agents to include in the test
- **Advanced Options**: Configure streaming and debug mode

### Execution Monitor

The Execution Monitor tab provides real-time monitoring of test execution:

- **Agent Status**: View the status of each agent (waiting, active, completed, error)
- **Execution Flow**: Visualize the flow of information between agents
- **Log Output**: View detailed logs of the execution process

### Results

The Results tab displays detailed information about test results:

- **Test Information**: Query and timestamp
- **Execution Metrics**: Execution time, agent counts, error status
- **Result Content**: View the final response, raw output, and agent outputs

### Quality Evaluation

The Quality Evaluation tab allows evaluating the quality of test results:

- **Test Query and Response**: View the query and response for context
- **Evaluation Form**: Rate the response on accuracy, completeness, and relevance
- **Comments**: Add detailed comments about the response quality

## UI Testing with Playwright

The `test_ui.py` file contains automated tests for the Streamlit interface using Playwright. These tests verify that:

1. The interface loads correctly with all expected elements
2. The tabs function properly
3. Simple queries can be executed successfully
4. Complex queries using multiple agents work correctly
5. The quality evaluation functionality works as expected
6. The parallel supervisor mode functions correctly

## Best Practices

When using the test interface:

1. **Start Simple**: Begin with simple queries and a single agent
2. **Incremental Complexity**: Gradually increase complexity by adding agents and using more complex queries
3. **Monitor Execution**: Use the Execution Monitor to identify issues during test execution
4. **Evaluate Quality**: Always evaluate the quality of responses to track improvements
5. **Document Issues**: Use the comments section to document any issues or observations

## Troubleshooting

Common issues and solutions:

- **Streamlit Not Starting**: Ensure you're in the correct directory and have Streamlit installed
- **Agent Errors**: Check the log output for detailed error messages
- **UI Tests Failing**: Ensure Streamlit is running before executing Playwright tests
- **Slow Execution**: Complex queries with multiple agents may take time to complete
