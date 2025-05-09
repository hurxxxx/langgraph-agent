"""
Supervisor with Different MCP Modes Example

This script demonstrates how to use the Supervisor with different MCP modes:
1. Standard MCP: Complex task breakdown and delegation to multiple agents
2. CrewAI MCP: Role-based agent teams with hierarchical structure
3. AutoGen MCP: Conversational multi-agent systems with dynamic agent interactions
4. LangGraph MCP: Graph-based workflows with conditional routing
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


def process_task(supervisor, query, mode_name):
    """Process a task using the supervisor with a specific MCP mode."""
    print(f"\n=== Processing Task with {mode_name} ===")
    print(f"Task: {query}")

    # Process query
    result = supervisor.invoke(query=query, stream=False)

    # Print complexity score if available
    if "complexity_score" in result:
        print(f"\nComplexity Score: {result['complexity_score']:.2f}")

    # Print execution plan if available
    if "execution_plan" in result:
        print("\n=== Execution Plan ===")
        print(json.dumps(result["execution_plan"], indent=2))

    # Print crew plan if available
    if "crew_plan" in result:
        print("\n=== Crew Plan ===")
        print(json.dumps(result["crew_plan"], indent=2))

    # Print conversation plan if available
    if "conversation_plan" in result:
        print("\n=== Conversation Plan ===")
        print(json.dumps(result["conversation_plan"], indent=2))

    # Print graph plan if available
    if "graph_plan" in result:
        print("\n=== Graph Plan ===")
        print(json.dumps(result["graph_plan"], indent=2))

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

    # Initialize supervisors with different MCP modes
    supervisor_standard_mcp = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=False,
            mcp_mode="mcp"
        ),
        agents=agents
    )

    supervisor_crew_mcp = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=False,
            mcp_mode="crew"
        ),
        agents=agents
    )

    supervisor_autogen_mcp = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=False,
            mcp_mode="autogen"
        ),
        agents=agents
    )

    supervisor_langgraph_mcp = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=False,
            mcp_mode="langgraph"
        ),
        agents=agents
    )

    # Example complex task
    complex_task = "Research the latest advancements in quantum computing, create an image visualizing quantum entanglement, and evaluate the quality of the information."

    # Process task with each MCP mode
    print("\n\n" + "="*80)
    print(f"COMPLEX TASK: {complex_task}")
    print("="*80 + "\n")

    # Process with standard MCP
    standard_result = process_task(supervisor_standard_mcp, complex_task, "Standard MCP")
    print("\n" + "-"*80 + "\n")

    # Process with CrewAI MCP
    crew_result = process_task(supervisor_crew_mcp, complex_task, "CrewAI MCP")
    print("\n" + "-"*80 + "\n")

    # Process with AutoGen MCP
    autogen_result = process_task(supervisor_autogen_mcp, complex_task, "AutoGen MCP")
    print("\n" + "-"*80 + "\n")

    # Process with LangGraph MCP
    langgraph_result = process_task(supervisor_langgraph_mcp, complex_task, "LangGraph MCP")
    print("\n" + "-"*80 + "\n")

    # Compare results
    print("\n=== Comparison ===")
    print(f"Standard MCP response length: {len(standard_result['messages'][-1]['content'])}")
    print(f"CrewAI MCP response length: {len(crew_result['messages'][-1]['content'])}")
    print(f"AutoGen MCP response length: {len(autogen_result['messages'][-1]['content'])}")
    print(f"LangGraph MCP response length: {len(langgraph_result['messages'][-1]['content'])}")


if __name__ == "__main__":
    main()
