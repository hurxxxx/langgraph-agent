"""Basic Usage Example for Multi-Agent Supervisor System"""

import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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


def non_streaming_example(supervisor, query):
    """Example of using the supervisor with a non-streaming response."""
    print("\n=== Non-Streaming Example ===")
    print(f"Query: {query}")

    result = supervisor.invoke(query=query, stream=False)
    final_message = result["messages"][-1]["content"] if result["messages"] else ""

    print("\nResponse:")
    print(final_message)

    if "agent_outputs" in result:
        print("\nAgent Outputs:")
        for agent_name, output in result["agent_outputs"].items():
            print(f"- {agent_name}: {output}")


def streaming_example(supervisor, query):
    """Example of using the supervisor with a streaming response."""
    print("\n=== Streaming Example ===")
    print(f"Query: {query}")
    print("\nResponse (streaming):")

    result = supervisor.invoke(query=query, stream=True)

    print("\nFinal Result:")
    if "messages" in result and result["messages"]:
        for i, message in enumerate(result["messages"]):
            print(f"\nMessage {i+1}:")
            print(f"Role: {message.get('role', 'unknown')}")
            print(f"Content: {message.get('content', '')}")

    if "agent_outputs" in result:
        print("\nAgent Outputs:")
        for agent_name, output in result["agent_outputs"].items():
            print(f"- {agent_name}: {output}")


def main():
    """Main function to run the examples."""
    load_dotenv()
    agents = initialize_agents()

    supervisor = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True
        ),
        agents=agents
    )

    queries = [
        "What are the latest advancements in quantum computing?",
        "Generate an image of a futuristic city with flying cars.",
        "Store this information: The capital of France is Paris.",
        "What is the current state of climate change research?",
        "How do large language models like GPT-4 work?"
    ]

    non_streaming_example(supervisor, queries[0])
    non_streaming_example(supervisor, queries[3])
    non_streaming_example(supervisor, queries[4])
    streaming_example(supervisor, queries[1])


if __name__ == "__main__":
    main()
