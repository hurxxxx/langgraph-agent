"""
Basic Usage Example for Multi-Agent Supervisor System

This script demonstrates how to use the multi-agent supervisor system
for processing queries with both streaming and non-streaming responses.
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
    """
    Initialize all specialized agents.

    Returns:
        Dict: Dictionary of agent functions keyed by agent name
    """
    # Initialize search agent with Serper
    search_agent = SearchAgent(
        config=SearchAgentConfig(
            provider="serper",  # Use Serper as the search provider
            max_results=3
        )
    )

    # Initialize vector storage agent
    vector_storage_agent = VectorStorageAgent(
        config=VectorStorageAgentConfig(
            store_type="chroma",
            collection_name="example_collection",
            persist_directory="./example_vector_db"
        )
    )

    # Initialize image generation agent
    image_generation_agent = ImageGenerationAgent(
        config=ImageGenerationAgentConfig(
            provider="dalle",
            dalle_model="dall-e-3",
            image_size="1024x1024"
        )
    )

    # Initialize quality agent
    quality_agent = QualityAgent(
        config=QualityAgentConfig()
    )

    # Return dictionary of agents
    return {
        "search_agent": search_agent,
        "vector_storage_agent": vector_storage_agent,
        "image_generation_agent": image_generation_agent,
        "quality_agent": quality_agent
    }


def non_streaming_example(supervisor, query):
    """
    Example of using the supervisor with a non-streaming response.

    Args:
        supervisor: Initialized supervisor
        query: Query to process
    """
    print("\n=== Non-Streaming Example ===")
    print(f"Query: {query}")

    # Process query
    result = supervisor.invoke(query=query, stream=False)

    # Extract the final response
    final_message = result["messages"][-1]["content"] if result["messages"] else ""

    print("\nResponse:")
    print(final_message)

    # Print agent outputs
    if "agent_outputs" in result:
        print("\nAgent Outputs:")
        for agent_name, output in result["agent_outputs"].items():
            print(f"- {agent_name}: {output}")


def streaming_example(supervisor, query):
    """
    Example of using the supervisor with a streaming response.

    Args:
        supervisor: Initialized supervisor
        query: Query to process
    """
    print("\n=== Streaming Example ===")
    print(f"Query: {query}")

    print("\nResponse (streaming):")

    # Process query with streaming
    # Note: The current implementation doesn't actually stream chunks
    # It just returns the final state with stream=True
    result = supervisor.invoke(query=query, stream=True)

    print("\nFinal Result:")
    if "messages" in result and result["messages"]:
        for i, message in enumerate(result["messages"]):
            print(f"\nMessage {i+1}:")
            print(f"Role: {message.get('role', 'unknown')}")
            print(f"Content: {message.get('content', '')}")

    # Print agent outputs
    if "agent_outputs" in result:
        print("\nAgent Outputs:")
        for agent_name, output in result["agent_outputs"].items():
            print(f"- {agent_name}: {output}")


def main():
    """Main function to run the examples."""
    # Load environment variables
    load_dotenv()

    # Initialize agents
    agents = initialize_agents()

    # Initialize supervisor
    supervisor = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True
        ),
        agents=agents
    )

    # Example queries for testing with real-world scenarios
    queries = [
        "What are the latest advancements in quantum computing?",
        "Generate an image of a futuristic city with flying cars.",
        "Store this information: The capital of France is Paris.",
        "What is the current state of climate change research?",
        "How do large language models like GPT-4 work?"
    ]

    # Run non-streaming examples for search queries
    non_streaming_example(supervisor, queries[0])  # Quantum computing
    non_streaming_example(supervisor, queries[3])  # Climate change
    non_streaming_example(supervisor, queries[4])  # LLMs

    # Run streaming example
    streaming_example(supervisor, queries[1])  # Image generation


if __name__ == "__main__":
    main()
