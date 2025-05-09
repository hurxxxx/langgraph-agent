"""
MCP Agent Example

This script demonstrates how to use the Master Control Program (MCP) agent
for breaking down complex tasks into subtasks and delegating them to specialized agents.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import MCP agent and specialized agents
from src.agents.mcp_agent import MCPAgent, MCPAgentConfig
from src.agents.search_agent import SearchAgent, SearchAgentConfig
from src.agents.vector_storage_agent import VectorStorageAgent, VectorStorageAgentConfig
from src.agents.image_generation_agent import ImageGenerationAgent, ImageGenerationAgentConfig
from src.agents.quality_agent import QualityAgent, QualityAgentConfig


def initialize_agents():
    """Initialize all specialized agents."""
    search_agent = SearchAgent(
        config=SearchAgentConfig(
            provider="serper",
            max_results=3
        )
    )

    vector_storage_agent = VectorStorageAgent(
        config=VectorStorageAgentConfig(
            store_type="chroma",
            collection_name="example_collection",
            persist_directory="./example_vector_db"
        )
    )

    image_generation_agent = ImageGenerationAgent(
        config=ImageGenerationAgentConfig(
            provider="dalle",
            dalle_model="dall-e-3",
            image_size="1024x1024"
        )
    )

    quality_agent = QualityAgent(
        config=QualityAgentConfig()
    )

    return {
        "search_agent": search_agent,
        "vector_storage_agent": vector_storage_agent,
        "image_generation_agent": image_generation_agent,
        "quality_agent": quality_agent
    }


def process_complex_task(mcp_agent, task):
    """Process a complex task using the MCP agent."""
    print(f"\n=== Processing Complex Task ===")
    print(f"Task: {task}")

    # Initialize state
    state = {
        "messages": [{"role": "user", "content": task}],
        "agent_outputs": {}
    }

    # Process task with MCP agent
    result = mcp_agent.invoke(state)

    # Print execution plan
    print("\n=== Execution Plan ===")
    print(json.dumps(result.get("execution_plan", {}), indent=2))

    # Print subtask results
    print("\n=== Subtask Results ===")
    print(json.dumps(result.get("subtask_results", {}), indent=2))

    # Print final response
    print("\n=== Final Response ===")
    print(result["messages"][-1]["content"])

    return result


def main():
    """Main function to run the example."""
    # Load environment variables
    load_dotenv()

    # Initialize agents
    agents = initialize_agents()

    # Initialize MCP agent
    mcp_agent = MCPAgent(
        config=MCPAgentConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            max_subtasks=3
        ),
        agents=agents
    )

    # Example complex tasks
    complex_tasks = [
        "Research the latest advancements in quantum computing, create an image visualizing quantum entanglement, and evaluate the quality of the information.",
        "Find information about climate change impacts, store this information in a vector database, and generate an image of a sustainable city.",
        "Research the history of artificial intelligence, its current state, and future prospects. Then create a comprehensive report with a visual representation."
    ]

    # Process each complex task
    for task in complex_tasks:
        process_complex_task(mcp_agent, task)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
