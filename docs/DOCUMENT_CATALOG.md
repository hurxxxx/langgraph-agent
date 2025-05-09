# Document Catalog

This catalog lists all the documentation created for the multi-agent supervisor system, with descriptions of each document's purpose and content.

## Project Documentation

| Document | Path | Description |
|----------|------|-------------|
| Product Requirements Document | `docs/PRD.md` | Outlines the requirements and specifications for the multi-agent supervisor system, including project goals, technical requirements, and success criteria. |
| Task List | `docs/TASK_LIST.md` | Tracks development tasks, including completed, in-progress, and pending tasks. |
| README | `README.md` | Provides an overview of the project, installation instructions, usage examples, and links to detailed documentation. |

## Learning Resources

| Document | Path | Description |
|----------|------|-------------|
| LangGraph Latest Information | `docs/learning/langgraph_latest.md` | Provides information about the latest features and best practices for LangGraph as of May 2025, including functional API, streaming support, and multi-agent architectures. |
| Library Versions | `docs/library_versions.md` | Tracks the latest versions of libraries used in the project and provides best practices for their usage, including version comparisons and recommended updates. |
| Specialized Agents | `docs/learning/specialized_agents.md` | Documents the specialized agents implemented in the system, including search agents, vector storage agents, image generation agents, and quality measurement agents. |
| OpenAPI Integration | `docs/learning/openapi_integration.md` | Explains how to integrate OpenAPI specifications with LangGraph to create standardized APIs for agent interactions. |
| Streaming Support | `docs/learning/streaming_support.md` | Details how to implement streaming responses in a multi-agent system built with LangGraph, including both server and client implementations. |
| Serper API Integration | `docs/learning/serper_integration.md` | Provides information about integrating Serper API with LangGraph for web search capabilities, including API reference and best practices. |
| UI Testing Strategy | `docs/UI_TESTING_STRATEGY.md` | Outlines the strategy for UI testing and continuous improvement, including Streamlit test interfaces, Playwright automated testing, and quality evaluation processes. |
| UI Testing Results | `docs/UI_TESTING_RESULTS.md` | Summarizes the results of UI testing, identifies issues, and provides recommendations for improvements with a detailed plan for addressing each issue. |
| UI Testing with Playwright | `docs/UI_TESTING_WITH_PLAYWRIGHT.md` | Describes the implementation of UI testing using Playwright, including test scenarios, running tests, and capturing screenshots for verification. |


## Code Documentation

| Component | Path | Description |
|-----------|------|-------------|
| Supervisor | `src/supervisor/supervisor.py` | Implements the supervisor agent that orchestrates multiple specialized agents, including handoff tools and task description mechanisms. |
| Parallel Supervisor | `src/supervisor/parallel_supervisor.py` | Implements a supervisor agent that can orchestrate multiple specialized agents in parallel. |
| Search Agent | `src/agents/search_agent.py` | Implements a search agent that can retrieve information from various search providers, including Serper, Tavily, Google, and DuckDuckGo. |
| Vector Storage Agent | `src/agents/vector_storage_agent.py` | Implements a vector storage agent that can store, update, and delete documents in various vector databases. |
| Image Generation Agent | `src/agents/image_generation_agent.py` | Implements an image generation agent that can create images using DALL-E and GPT-4o. |
| Quality Agent | `src/agents/quality_agent.py` | Implements a quality measurement agent that evaluates the quality of responses based on various criteria. |
| LangSmith Utilities | `src/utils/langsmith_utils.py` | Provides utilities for integrating LangSmith tracing and monitoring with the multi-agent system. |
| Main Application | `src/app.py` | Brings together all components of the multi-agent supervisor system and provides an API for interacting with the system. |

## Examples & Tests

| File | Path | Description |
|------|------|-------------|
| Basic Usage | `examples/basic_usage.py` | Demonstrates how to use the multi-agent supervisor system for processing queries with both streaming and non-streaming responses. |
| API Client | `examples/api_client.py` | Shows how to interact with the multi-agent supervisor API from a Python client. |
| Web Client | `examples/web_client.html` | Provides a simple HTML/JavaScript client for interacting with the multi-agent supervisor API. |
| Agent Monitor UI | `src/ui/agent_monitor.html` | Enhanced UI for testing and monitoring agent execution with LangSmith integration. |
| Monitor Runner | `run_monitor.py` | Script to run the application with the monitoring UI in a web browser. |
| Streamlit Test Interface | `ui/streamlit/app.py` | Streamlit-based test interface for configuring, executing, and evaluating tests of the multi-agent supervisor system. |
| UI Tests | `ui/streamlit/test_ui.py` | Playwright-based automated tests for the Streamlit test interface. |
| Prompt Input Tests | `ui/streamlit/test_prompt_input.py` | Playwright tests for direct prompt input in the Streamlit UI, testing various query types and agent combinations. |
| UI Test Runner | `ui/streamlit/run_ui_tests.py` | Script to run the Streamlit app and execute Playwright tests automatically. |


## Configuration

| File | Path | Description |
|------|------|-------------|
| Environment Variables Example | `.env.example` | Provides an example of the environment variables needed to run the system, including API keys for various services. |
| Requirements | `requirements.txt` | Lists all the Python dependencies required to run the system. |
| Monitoring UI | `docs/monitoring_ui.md` | Documentation for the monitoring UI and LangSmith integration for debugging and testing. |
| Streamlit UI README | `ui/streamlit/README.md` | Documentation for the Streamlit test interface, including setup instructions, features, and best practices for UI testing. |
