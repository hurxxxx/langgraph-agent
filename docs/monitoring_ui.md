# Monitoring UI and LangSmith Integration

This document explains how to use the monitoring UI and LangSmith integration for debugging and testing the multi-agent supervisor system.

## Overview

The monitoring UI provides a simple interface for testing the multi-agent supervisor system and visualizing the execution flow. It includes:

- A query input area for submitting queries to the system
- A response area for displaying the final response
- Agent logs for monitoring the execution flow
- Subtask visualization for parallel execution
- Execution statistics for performance monitoring
- LangSmith integration for detailed tracing and debugging

## Running the Monitoring UI

To run the monitoring UI, use the provided script:

```bash
./run_monitor.py
```

This will start the FastAPI server and open the monitoring UI in your default web browser.

Alternatively, you can run the server manually and access the UI at `http://localhost:8000`:

```bash
python src/app.py
```

## Using the Monitoring UI

### Submitting Queries

1. Enter your query in the text area
2. Toggle options:
   - **Enable Streaming**: Enable/disable streaming responses
   - **Use Parallel Supervisor**: Switch between standard and parallel supervisor
3. Click "Submit" to process the query
4. Click "Clear" to reset the UI

### Monitoring Execution

The UI provides several views for monitoring execution:

#### Response Area

Displays the final response from the system. For streaming responses, this will update in real-time as the response is generated.

#### Agent Logs

Shows a chronological log of agent activities, including:
- System messages
- Agent selection
- Agent execution
- Errors

#### Subtasks

For parallel execution, displays the list of subtasks with their status:
- In progress (yellow)
- Completed (green)

Each subtask shows:
- Subtask ID
- Description
- Agent assigned
- Dependencies

#### Raw JSON

Shows the raw JSON response from the API, useful for debugging.

#### Execution Stats

Displays performance metrics:
- Total execution time
- Number of parallel batches
- Completed subtasks

## LangSmith Integration

The system integrates with LangSmith for detailed tracing and debugging of agent execution.

### Setup

1. Make sure you have a LangSmith API key in your `.env` file:
   ```
   LANGSMITH_API_KEY=your-langsmith-api-key
   LANGSMITH_PROJECT=langgraph-agent
   ```

2. The system will automatically trace all agent and supervisor executions to LangSmith.

### Viewing Traces in LangSmith

1. Click the "View in LangSmith" link in the Execution Stats section
2. This will open the LangSmith dashboard in a new tab
3. Navigate to your project to see all traces

### Trace Information

LangSmith traces include:
- Input queries
- Agent selection decisions
- Agent outputs
- Execution times
- Error information

### Debugging with LangSmith

LangSmith provides several tools for debugging:
- Detailed execution traces
- Input/output inspection
- Performance metrics
- Error analysis

## Implementation Details

### Tracing Utilities

The `src/utils/langsmith_utils.py` module provides utilities for LangSmith integration:

- `LangSmithTracer`: Main class for tracing agent execution
- `trace_agent`: Decorator for tracing agent functions
- `trace_supervisor`: Decorator for tracing supervisor functions

### Supervisor Integration

Both the standard and parallel supervisors are integrated with LangSmith tracing:

```python
@tracer.trace_supervisor("StandardSupervisor")
def invoke(self, query, stream=False):
    # Supervisor implementation
```

### Agent Integration

Agents are dynamically wrapped with tracing in the parallel supervisor:

```python
# Wrap the agent with tracing if it's not already wrapped
if not hasattr(agent, "__wrapped__"):
    agent = tracer.trace_agent(agent_name)(agent)
```

## Extending the Monitoring UI

The monitoring UI can be extended with additional features:

1. Add more visualization options for agent interactions
2. Implement more detailed performance metrics
3. Add user feedback mechanisms
4. Integrate with other monitoring tools

## Troubleshooting

### LangSmith Connection Issues

If you encounter issues with LangSmith integration:

1. Verify your API key in the `.env` file
2. Check network connectivity to LangSmith servers
3. Look for error messages in the server logs

### UI Display Issues

If the UI doesn't display correctly:

1. Check browser console for JavaScript errors
2. Verify that the API server is running
3. Check CORS settings if accessing from a different domain

## Future Enhancements

Planned enhancements for the monitoring UI:

1. More detailed visualization of agent interactions
2. Real-time performance metrics
3. User feedback collection
4. Integration with more monitoring tools
5. Enhanced LangSmith integration with custom feedback
