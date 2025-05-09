#!/usr/bin/env python3
"""
Streamlit Test Interface for Multi-Agent Supervisor System.

This application provides a testing interface for the multi-agent supervisor system,
allowing users to execute tests, monitor execution, and evaluate results.

Usage:
    streamlit run app.py
"""

import os
import sys
import time
import json
import asyncio
from typing import Dict, Any, List, Optional
import streamlit as st
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# Import the supervisor and related components
from src.supervisor.supervisor import Supervisor, SupervisorConfig
from src.supervisor.parallel_supervisor import ParallelSupervisor, ParallelSupervisorConfig
from src.agents.search_agent import SearchAgent, SearchAgentConfig
from src.agents.image_generation_agent import ImageGenerationAgent, ImageGenerationAgentConfig
from src.agents.sql_rag_agent import SQLRAGAgent, SQLRAGAgentConfig
from src.agents.vector_storage_agent import VectorStorageAgent, VectorStorageAgentConfig
from src.agents.document_generation import BaseDocumentAgent as DocumentGenerationAgent
from src.agents.document_generation import BaseDocumentAgentConfig as DocumentGenerationAgentConfig


# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Multi-Agent Supervisor Test Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS
st.markdown("""
<style>
    .agent-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .agent-card.active {
        border-color: #4CAF50;
        background-color: #f0fff0;
    }
    .agent-card.error {
        border-color: #FF5733;
        background-color: #fff0f0;
    }
    .log-container {
        height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        font-family: monospace;
        background-color: #f9f9f9;
    }
    .result-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .metric-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        flex: 1;
        min-width: 150px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "test_results" not in st.session_state:
    st.session_state.test_results = []
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "current_test_id" not in st.session_state:
    st.session_state.current_test_id = None
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {}
if "execution_stats" not in st.session_state:
    st.session_state.execution_stats = {}
if "streaming_chunks" not in st.session_state:
    st.session_state.streaming_chunks = []


def add_log_message(message: str, level: str = "info"):
    """Add a log message to the session state."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.log_messages.append({
        "timestamp": timestamp,
        "message": message,
        "level": level
    })


def update_agent_status(agent_name: str, status: str, has_error: bool = False):
    """Update the status of an agent."""
    st.session_state.agent_status[agent_name] = {
        "status": status,
        "updated_at": time.time(),
        "has_error": has_error
    }


def create_supervisor(config_options: Dict[str, Any]) -> Supervisor:
    """Create a supervisor instance based on configuration options."""
    # Create agent instances
    agents = {}

    if config_options.get("use_search_agent", False):
        agents["search_agent"] = SearchAgent(
            config=SearchAgentConfig(
                provider=config_options.get("search_provider", "serper"),
                llm_provider=config_options.get("llm_provider", "openai"),
                openai_model=config_options.get("openai_model", "gpt-4o")
            )
        )

    if config_options.get("use_image_agent", False):
        agents["image_generation_agent"] = ImageGenerationAgent(
            config=ImageGenerationAgentConfig(
                provider=config_options.get("image_provider", "dalle"),  # Changed from "openai" to "dalle"
                llm_provider=config_options.get("llm_provider", "openai"),
                openai_model=config_options.get("openai_model", "gpt-4o"),
                save_images=True
            )
        )

    if config_options.get("use_sql_agent", False):
        agents["sql_rag_agent"] = SQLRAGAgent(
            config=SQLRAGAgentConfig(
                llm_provider=config_options.get("llm_provider", "openai"),
                openai_model=config_options.get("openai_model", "gpt-4o")
            )
        )

    if config_options.get("use_vector_agent", False):
        agents["vector_storage_agent"] = VectorStorageAgent(
            config=VectorStorageAgentConfig(
                llm_provider=config_options.get("llm_provider", "openai"),
                openai_model=config_options.get("openai_model", "gpt-4o")
            )
        )

    if config_options.get("use_document_agent", False):
        agents["document_generation_agent"] = DocumentGenerationAgent(
            config=DocumentGenerationAgentConfig(
                llm_provider=config_options.get("llm_provider", "openai"),
                openai_model=config_options.get("openai_model", "gpt-4o")
            )
        )

    # Create supervisor config
    if config_options.get("supervisor_type", "standard") == "parallel":
        supervisor_config = ParallelSupervisorConfig(
            llm_provider=config_options.get("llm_provider", "openai"),
            openai_model=config_options.get("openai_model", "gpt-4o"),
            streaming=config_options.get("streaming", True)
            # Removed debug parameter as it's not supported
        )
        supervisor = ParallelSupervisor(config=supervisor_config, agents=agents)
    else:
        supervisor_config = SupervisorConfig(
            llm_provider=config_options.get("llm_provider", "openai"),
            openai_model=config_options.get("openai_model", "gpt-4o"),
            streaming=config_options.get("streaming", True),
            mcp_mode=config_options.get("mcp_mode", "standard"),
            complexity_threshold=config_options.get("complexity_threshold", 0.7)
            # Removed debug parameter as it's not supported
        )
        supervisor = Supervisor(config=supervisor_config, agents=agents)

    return supervisor


async def run_test(query: str, config_options: Dict[str, Any]):
    """Run a test with the given query and configuration options."""
    # Create a unique test ID
    test_id = int(time.time())
    st.session_state.current_test_id = test_id

    # Reset state for new test
    st.session_state.agent_status = {}
    st.session_state.streaming_chunks = []

    # Log test start
    add_log_message(f"Starting test with query: {query}", "info")

    # Create supervisor
    try:
        supervisor = create_supervisor(config_options)
        add_log_message("Supervisor created successfully", "info")
    except Exception as e:
        add_log_message(f"Error creating supervisor: {str(e)}", "error")
        return

    # Initialize execution stats
    st.session_state.execution_stats = {
        "start_time": time.time(),
        "end_time": None,
        "total_agents": len(supervisor.agents),
        "completed_agents": 0,
        "has_errors": False
    }

    # Initialize agent status
    for agent_name in supervisor.agents:
        update_agent_status(agent_name, "waiting")

    # Run the test
    try:
        if config_options.get("streaming", True):
            add_log_message("Running test with streaming enabled", "info")

            # Process streaming results
            async for chunk in supervisor.astream(query):
                # Update streaming chunks
                st.session_state.streaming_chunks.append(chunk)

                # Update agent status if available
                if "current_agent" in chunk:
                    agent_name = chunk["current_agent"]
                    update_agent_status(agent_name, "active")

                # Check for agent outputs
                if "agent_outputs" in chunk:
                    for agent_name, output in chunk["agent_outputs"].items():
                        has_error = output.get("has_error", False) or "error" in output
                        status = "error" if has_error else "completed"
                        update_agent_status(agent_name, status, has_error)

                        if has_error:
                            st.session_state.execution_stats["has_errors"] = True
                            error_msg = output.get("error", "Unknown error")
                            add_log_message(f"Error in {agent_name}: {error_msg}", "error")
                        else:
                            st.session_state.execution_stats["completed_agents"] += 1
                            add_log_message(f"Agent {agent_name} completed successfully", "info")

                # Force a rerun to update the UI
                st.rerun()

            # Get the final result from the last chunk
            result = st.session_state.streaming_chunks[-1] if st.session_state.streaming_chunks else {}
        else:
            add_log_message("Running test without streaming", "info")
            result = supervisor.invoke(query)

            # Update agent status based on result
            if "agent_outputs" in result:
                for agent_name, output in result["agent_outputs"].items():
                    has_error = output.get("has_error", False) or "error" in output
                    status = "error" if has_error else "completed"
                    update_agent_status(agent_name, status, has_error)

                    if has_error:
                        st.session_state.execution_stats["has_errors"] = True
                        error_msg = output.get("error", "Unknown error")
                        add_log_message(f"Error in {agent_name}: {error_msg}", "error")
                    else:
                        st.session_state.execution_stats["completed_agents"] += 1
                        add_log_message(f"Agent {agent_name} completed successfully", "info")

        # Update execution stats
        st.session_state.execution_stats["end_time"] = time.time()

        # Store test result
        st.session_state.test_results.append({
            "id": test_id,
            "query": query,
            "config": config_options,
            "result": result,
            "execution_stats": st.session_state.execution_stats,
            "timestamp": time.time()
        })

        add_log_message("Test completed successfully", "info")
    except Exception as e:
        add_log_message(f"Error running test: {str(e)}", "error")
        st.session_state.execution_stats["has_errors"] = True
        st.session_state.execution_stats["end_time"] = time.time()


# Sidebar - Test Configuration
st.sidebar.title("Test Configuration")

# Query input
query = st.sidebar.text_area("Test Query", height=100,
                            placeholder="Enter your test query here...")

# Supervisor configuration
st.sidebar.subheader("Supervisor Configuration")
supervisor_type = st.sidebar.selectbox(
    "Supervisor Type",
    options=["standard", "parallel"],
    format_func=lambda x: "Standard" if x == "standard" else "Parallel"
)

if supervisor_type == "standard":
    mcp_mode = st.sidebar.selectbox(
        "MCP Mode",
        options=["standard", "mcp", "crew", "autogen", "langgraph"],
        format_func=lambda x: x.capitalize() if x != "mcp" else "MCP"
    )
    complexity_threshold = st.sidebar.slider(
        "Complexity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Threshold for determining when to use MCP mode"
    )
else:
    mcp_mode = "standard"
    complexity_threshold = 0.7

# LLM configuration
st.sidebar.subheader("LLM Configuration")
llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    options=["openai", "anthropic"],
    format_func=lambda x: "OpenAI" if x == "openai" else "Anthropic"
)
openai_model = st.sidebar.selectbox(
    "OpenAI Model",
    options=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    disabled=llm_provider != "openai"
)

# Agent selection
st.sidebar.subheader("Agents")
use_search_agent = st.sidebar.checkbox("Search Agent", value=True)
use_image_agent = st.sidebar.checkbox("Image Generation Agent", value=True)

# Image agent configuration (only show if image agent is selected)
if use_image_agent:
    image_provider = st.sidebar.selectbox(
        "Image Provider",
        options=["dalle", "gpt4o"],
        format_func=lambda x: "DALL-E" if x == "dalle" else "GPT-4o"
    )
else:
    image_provider = "dalle"  # Default value

use_sql_agent = st.sidebar.checkbox("SQL RAG Agent")
use_vector_agent = st.sidebar.checkbox("Vector Storage Agent")
use_document_agent = st.sidebar.checkbox("Document Generation Agent")

# Advanced options
st.sidebar.subheader("Advanced Options")
streaming = st.sidebar.checkbox("Enable Streaming", value=True)
debug_mode = st.sidebar.checkbox("Debug Mode")

# Execute button
if st.sidebar.button("Execute Test", type="primary"):
    # Collect configuration options
    config_options = {
        "supervisor_type": supervisor_type,
        "mcp_mode": mcp_mode,
        "complexity_threshold": complexity_threshold,
        "llm_provider": llm_provider,
        "openai_model": openai_model,
        "use_search_agent": use_search_agent,
        "use_image_agent": use_image_agent,
        "image_provider": image_provider,  # Added image provider
        "use_sql_agent": use_sql_agent,
        "use_vector_agent": use_vector_agent,
        "use_document_agent": use_document_agent,
        "streaming": streaming,
        "debug_mode": debug_mode
    }

    # Run the test
    asyncio.run(run_test(query, config_options))

# Main content
st.title("Multi-Agent Supervisor Test Interface")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Execution Monitor", "Results", "Quality Evaluation"])

with tab1:
    # Execution Monitor
    st.header("Execution Monitor")

    # Agent Status
    st.subheader("Agent Status")

    # Create columns for agent status cards
    if st.session_state.agent_status:
        cols = st.columns(min(5, len(st.session_state.agent_status)))
        for i, (agent_name, status) in enumerate(st.session_state.agent_status.items()):
            col_idx = i % len(cols)
            with cols[col_idx]:
                status_class = "active" if status["status"] == "active" else "error" if status["has_error"] else ""
                st.markdown(f"""
                <div class="agent-card {status_class}">
                    <h4>{agent_name}</h4>
                    <p>Status: {status["status"].capitalize()}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No agents active. Start a test to see agent status.")

    # Execution Flow
    st.subheader("Execution Flow")

    # Create a placeholder for the execution flow visualization
    flow_placeholder = st.empty()

    # Log Output
    st.subheader("Log Output")

    # Create a container for log messages
    log_container = st.container()

    with log_container:
        st.markdown('<div class="log-container">', unsafe_allow_html=True)
        for log in st.session_state.log_messages:
            color = "red" if log["level"] == "error" else "green" if log["level"] == "success" else "blue"
            st.markdown(f'<span style="color: {color}">[{log["timestamp"]}] {log["message"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Results
    st.header("Test Results")

    if st.session_state.test_results:
        # Get the latest test result
        latest_result = st.session_state.test_results[-1]

        # Display test information
        st.subheader("Test Information")
        st.write(f"Query: {latest_result['query']}")
        st.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_result['timestamp']))}")

        # Display execution metrics
        st.subheader("Execution Metrics")

        # Create metrics cards
        metrics_cols = st.columns(4)

        execution_time = latest_result["execution_stats"]["end_time"] - latest_result["execution_stats"]["start_time"]
        with metrics_cols[0]:
            st.metric("Execution Time", f"{execution_time:.2f}s")

        with metrics_cols[1]:
            st.metric("Total Agents", latest_result["execution_stats"]["total_agents"])

        with metrics_cols[2]:
            st.metric("Completed Agents", latest_result["execution_stats"]["completed_agents"])

        with metrics_cols[3]:
            st.metric("Errors", "Yes" if latest_result["execution_stats"]["has_errors"] else "No")

        # Display result content
        st.subheader("Result Content")

        # Create tabs for different result views
        result_tab1, result_tab2, result_tab3 = st.tabs(["Formatted", "Raw Output", "Agent Outputs"])

        with result_tab1:
            # Display the final response
            if "messages" in latest_result["result"]:
                final_message = latest_result["result"]["messages"][-1]["content"]
                st.markdown(final_message)
            else:
                st.warning("No formatted result available")

        with result_tab2:
            # Display the raw result
            st.json(latest_result["result"])

        with result_tab3:
            # Display agent outputs
            if "agent_outputs" in latest_result["result"]:
                for agent_name, output in latest_result["result"]["agent_outputs"].items():
                    with st.expander(f"{agent_name} Output"):
                        st.json(output)
            else:
                st.warning("No agent outputs available")
    else:
        st.info("No test results available. Run a test to see results.")

with tab3:
    # Quality Evaluation
    st.header("Quality Evaluation")

    if st.session_state.test_results:
        # Get the latest test result
        latest_result = st.session_state.test_results[-1]

        # Display the query and response for evaluation
        st.subheader("Test Query")
        st.write(latest_result["query"])

        st.subheader("Response")
        if "messages" in latest_result["result"]:
            final_message = latest_result["result"]["messages"][-1]["content"]
            st.markdown(final_message)
        else:
            st.warning("No response available for evaluation")

        # Quality evaluation form
        st.subheader("Evaluation Form")

        # Create columns for rating categories
        eval_cols = st.columns(3)

        with eval_cols[0]:
            accuracy = st.slider("Accuracy", 1, 5, 3, help="How factually accurate is the response?")

        with eval_cols[1]:
            completeness = st.slider("Completeness", 1, 5, 3, help="How complete is the response?")

        with eval_cols[2]:
            relevance = st.slider("Relevance", 1, 5, 3, help="How relevant is the response to the query?")

        # Additional comments
        comments = st.text_area("Comments", placeholder="Enter any additional comments or observations...")

        # Submit button
        if st.button("Submit Evaluation"):
            # Store the evaluation
            if "evaluations" not in st.session_state:
                st.session_state.evaluations = []

            st.session_state.evaluations.append({
                "test_id": latest_result["id"],
                "accuracy": accuracy,
                "completeness": completeness,
                "relevance": relevance,
                "comments": comments,
                "timestamp": time.time()
            })

            st.success("Evaluation submitted successfully!")
    else:
        st.info("No test results available for evaluation. Run a test first.")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Streamlit app is running!")
