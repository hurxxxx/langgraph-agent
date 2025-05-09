"""
MCP Comparison Example

This script demonstrates and compares the three different MCP agent implementations:
1. CrewAI Style MCP: Role-based agent teams
2. AutoGen Style MCP: Conversational multi-agent systems
3. LangGraph Style MCP: Graph-based workflows
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import MCP agents and specialized agents
from src.agents.crew_mcp_agent import CrewMCPAgent, CrewMCPAgentConfig
from src.agents.autogen_mcp_agent import AutoGenMCPAgent, AutoGenMCPAgentConfig
from src.agents.langgraph_mcp_agent import LangGraphMCPAgent, LangGraphMCPAgentConfig
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


def process_with_crew_mcp(agents, task):
    """Process a task using the CrewAI-style MCP agent."""
    print(f"\n=== Processing with CrewAI-style MCP ===")
    print(f"Task: {task}")

    # Initialize CrewAI-style MCP agent
    crew_mcp = CrewMCPAgent(
        config=CrewMCPAgentConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            max_agents=4
        ),
        agents=agents
    )

    # Initialize state
    state = {
        "messages": [{"role": "user", "content": task}],
        "agent_outputs": {}
    }

    # Process task
    result = crew_mcp.invoke(state)

    # Print crew plan
    print("\n=== Crew Plan ===")
    print(json.dumps(result.get("crew_plan", {}), indent=2))

    # Print role results
    print("\n=== Role Results ===")
    print(json.dumps(result.get("role_results", {}), indent=2))

    # Print final response
    print("\n=== Final Response ===")
    print(result["messages"][-1]["content"])

    return result


def process_with_autogen_mcp(agents, task):
    """Process a task using the AutoGen-style MCP agent."""
    print(f"\n=== Processing with AutoGen-style MCP ===")
    print(f"Task: {task}")

    # Initialize AutoGen-style MCP agent
    autogen_mcp = AutoGenMCPAgent(
        config=AutoGenMCPAgentConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            max_turns=10
        ),
        agents=agents
    )

    # Initialize state
    state = {
        "messages": [{"role": "user", "content": task}],
        "agent_outputs": {}
    }

    # Process task
    result = autogen_mcp.invoke(state)

    # Print conversation plan
    print("\n=== Conversation Plan ===")
    print(json.dumps(result.get("conversation_plan", {}), indent=2))

    # Print conversation
    print("\n=== Conversation ===")
    for message in result.get("conversation", []):
        if "name" in message:
            print(f"{message.get('name', 'Unknown')}: {message.get('content', '')}")
        else:
            print(f"{message.get('role', 'Unknown')}: {message.get('content', '')}")

    # Print final response
    print("\n=== Final Response ===")
    print(result["messages"][-1]["content"])

    return result


def process_with_langgraph_mcp(agents, task):
    """Process a task using the LangGraph-style MCP agent."""
    print(f"\n=== Processing with LangGraph-style MCP ===")
    print(f"Task: {task}")

    # Initialize LangGraph-style MCP agent
    langgraph_mcp = LangGraphMCPAgent(
        config=LangGraphMCPAgentConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            max_nodes=5
        ),
        agents=agents
    )

    # Initialize state
    state = {
        "messages": [{"role": "user", "content": task}],
        "agent_outputs": {}
    }

    # Process task
    result = langgraph_mcp.invoke(state)

    # Print graph plan
    print("\n=== Graph Plan ===")
    print(json.dumps(result.get("graph_plan", {}), indent=2))

    # Print graph execution
    print("\n=== Graph Execution ===")
    print(json.dumps(result.get("graph_execution", []), indent=2))

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

    # Example complex tasks
    complex_tasks = [
        "Research the latest advancements in quantum computing, create an image visualizing quantum entanglement, and evaluate the quality of the information.",
        "Find information about climate change impacts, store this information in a vector database, and generate an image of a sustainable city.",
        "Research the history of artificial intelligence, its current state, and future prospects. Then create a comprehensive report with a visual representation."
    ]

    # Process each task with each MCP agent
    for i, task in enumerate(complex_tasks):
        print(f"\n\n{'='*80}")
        print(f"TASK {i+1}: {task}")
        print(f"{'='*80}\n")

        # Process with CrewAI-style MCP
        crew_result = process_with_crew_mcp(agents, task)
        print("\n" + "-"*80 + "\n")

        # Process with AutoGen-style MCP
        autogen_result = process_with_autogen_mcp(agents, task)
        print("\n" + "-"*80 + "\n")

        # Process with LangGraph-style MCP
        langgraph_result = process_with_langgraph_mcp(agents, task)
        print("\n" + "-"*80 + "\n")

        # Compare results
        print("\n=== Comparison ===")
        print(f"CrewAI response length: {len(crew_result['messages'][-1]['content'])}")
        print(f"AutoGen response length: {len(autogen_result['messages'][-1]['content'])}")
        print(f"LangGraph response length: {len(langgraph_result['messages'][-1]['content'])}")


if __name__ == "__main__":
    main()
