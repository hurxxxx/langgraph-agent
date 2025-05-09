"""
Test script for the parallel supervisor agent.

This script tests the parallel supervisor agent's ability to break down complex tasks
into subtasks and delegate them to appropriate agents, potentially in parallel.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import supervisor and agents
from src.supervisor.parallel_supervisor import ParallelSupervisor, ParallelSupervisorConfig
from src.agents.search_agent import SearchAgent, SearchAgentConfig
from src.agents.image_generation_agent import ImageGenerationAgent, ImageGenerationAgentConfig
from src.agents.quality_agent import QualityAgent, QualityAgentConfig
from src.agents.vector_storage_agent import VectorStorageAgent, VectorStorageAgentConfig


def test_parallel_supervisor():
    """
    Test the parallel supervisor agent with complex prompts.
    """
    print("Testing parallel supervisor agent with complex prompts...")

    # Load environment variables
    load_dotenv()

    # Check if required API keys are set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    serper_api_key = os.getenv("SERPER_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in the .env file and try again.")
        return

    if not serper_api_key:
        print("Error: SERPER_API_KEY environment variable is not set.")
        print("Please set it in the .env file and try again.")
        return

    try:
        # Initialize agents
        print("Initializing agents...")
        search_agent = SearchAgent(
            config=SearchAgentConfig(
                provider="serper",
                max_results=3
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

        vector_storage_agent = VectorStorageAgent(
            config=VectorStorageAgentConfig(
                store_type="chroma",
                collection_name="test_collection",
                persist_directory="./test_vector_db"
            )
        )

        # Initialize parallel supervisor
        print("Initializing parallel supervisor agent...")
        supervisor = ParallelSupervisor(
            config=ParallelSupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                temperature=0,
                streaming=False,
                system_message="""
                You are a supervisor agent that coordinates multiple specialized agents to solve complex tasks.
                Your job is to:
                1. Understand the user's request
                2. Break down the request into subtasks
                3. Determine which specialized agent(s) should handle each subtask
                4. Coordinate the flow of information between agents
                5. Synthesize a final response for the user

                You can run agents in parallel when appropriate to save time.
                """
            ),
            agents={
                "search_agent": search_agent,
                "image_generation_agent": image_generation_agent,
                "quality_agent": quality_agent,
                "vector_storage_agent": vector_storage_agent
            }
        )

        # Test complex prompts
        test_prompts = [
            "Search for information about climate change, then generate an image of a sustainable city, and finally evaluate the quality of the information",
            "Find the latest research on quantum computing, create an image representing quantum entanglement, and store the information in a vector database",
            "Research the history of artificial intelligence, generate an image showing the evolution of AI, and assess the quality of the research"
        ]

        # Run tests
        for i, prompt in enumerate(test_prompts):
            print(f"\n\nTest {i+1}: {prompt}")
            print("-" * 80)

            try:
                print(f"Processing prompt: {prompt}")
                # Process prompt with timeout handling
                start_time = time.time()
                result = supervisor.invoke(query=prompt, stream=False)
                end_time = time.time()

                print(f"Prompt processed in {end_time - start_time:.2f} seconds")

                # Print subtasks
                if "subtasks" in result:
                    print("\nSubtasks:")
                    for subtask in result["subtasks"]:
                        print(f"- {subtask}")

                # Print agent outputs
                if "agent_outputs" in result:
                    print("\nAgent Outputs:")
                    for agent_name, output in result["agent_outputs"].items():
                        print(f"- {agent_name}: {output}")

                # Print final response
                if "final_response" in result:
                    print("\nFinal Response:")
                    print(result["final_response"])
                elif "messages" in result and result["messages"]:
                    final_message = result["messages"][-1]["content"]
                    print("\nFinal Response:")
                    print(final_message)

                print("\nTest passed!")
            except Exception as e:
                print(f"\nTest failed: {str(e)}")
                import traceback
                traceback.print_exc()

        print("\n\nAll tests completed.")
    except Exception as e:
        print(f"Setup failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_parallel_supervisor()
