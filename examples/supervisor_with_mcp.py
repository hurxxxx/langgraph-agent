"""
Supervisor with MCP Agent Example

This script demonstrates how to use the Supervisor with MCP agent enabled
for handling complex tasks that require multiple specialized agents.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import supervisor and agents
from src.supervisor.supervisor import Supervisor, SupervisorConfig
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


def process_task(supervisor, query, use_mcp=False):
    """Process a task using the supervisor."""
    print(f"\n=== Processing Task ===")
    print(f"Task: {query}")
    print(f"Using MCP: {use_mcp}")

    # Process query
    result = supervisor.invoke(query=query, stream=False)

    # Print complexity score
    if "complexity_score" in result:
        print(f"\nComplexity Score: {result['complexity_score']:.2f}")

    # Print execution plan if available
    if "execution_plan" in result:
        print("\n=== Execution Plan ===")
        print(json.dumps(result["execution_plan"], indent=2))

    # Print subtask results if available
    if "subtask_results" in result:
        print("\n=== Subtask Results ===")
        print(json.dumps(result["subtask_results"], indent=2))

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

    # Initialize supervisor with MCP enabled
    supervisor_with_mcp = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=False,
            use_mcp=True,
            complexity_threshold=0.6  # Lower threshold to ensure MCP is used for most tasks
        ),
        agents=agents
    )

    # Initialize supervisor without MCP
    supervisor_without_mcp = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=False,
            use_mcp=False
        ),
        agents=agents
    )

    # Example tasks
    simple_tasks = [
        "What is the capital of France?",
        "Generate an image of a cat."
    ]

    complex_tasks = [
        "Research the latest advancements in quantum computing, create an image visualizing quantum entanglement, and evaluate the quality of the information.",
        "Find information about climate change impacts, store this information in a vector database, and generate an image of a sustainable city.",
        "Research the history of artificial intelligence, its current state, and future prospects. Then create a comprehensive report with a visual representation."
    ]

    # Process simple tasks with standard supervisor
    print("\n\n=== SIMPLE TASKS WITH STANDARD SUPERVISOR ===\n")
    for task in simple_tasks:
        process_task(supervisor_without_mcp, task)
        print("\n" + "="*80 + "\n")

    # Process complex tasks with MCP-enabled supervisor
    print("\n\n=== COMPLEX TASKS WITH MCP-ENABLED SUPERVISOR ===\n")
    for task in complex_tasks:
        process_task(supervisor_with_mcp, task)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
