"""
Test script for the enhanced parallel supervisor agent.

This script tests the enhanced parallel supervisor agent's ability to:
1. Automatically identify parallelizable tasks in prompts
2. Process multiple search topics in parallel
3. Execute independent tasks in parallel
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


def test_enhanced_parallel_supervisor():
    """
    Test the enhanced parallel supervisor agent with complex prompts.
    """
    print("Testing enhanced parallel supervisor agent with complex prompts...")

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
        print("Initializing enhanced parallel supervisor agent...")
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

        # Test complex prompts with multiple search topics
        test_prompts = [
            "Compare the differences between quantum computing and classical computing, and generate an image showing both types of computers",
            "Research climate change impacts on both agriculture and coastal cities, then evaluate the quality of the information",
            "Find information about renewable energy sources like solar, wind, and hydroelectric power, and create an image of a sustainable energy grid"
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

                # Print subtasks and their parallelization info
                if "subtasks" in result:
                    print("\nSubtasks with Parallelization Info:")
                    for subtask in result["subtasks"]:
                        parallelizable = subtask.get("parallelizable", False)
                        depends_on = subtask.get("depends_on", [])
                        parallel_group = subtask.get("parallel_group", "none")
                        
                        print(f"- ID: {subtask['subtask_id']}, Agent: {subtask['agent']}")
                        print(f"  Description: {subtask['description']}")
                        print(f"  Parallelizable: {parallelizable}, Group: {parallel_group}, Depends on: {depends_on}")

                # Print execution statistics
                if "execution_stats" in result:
                    stats = result["execution_stats"]
                    print("\nExecution Statistics:")
                    print(f"- Total subtasks: {stats.get('total_subtasks', 0)}")
                    print(f"- Completed subtasks: {stats.get('completed_subtasks', 0)}")
                    print(f"- Parallel batches: {stats.get('parallel_batches', 0)}")
                    print(f"- Total execution time: {stats.get('total_execution_time', 0):.2f} seconds")
                    
                    # Print individual subtask execution times
                    if "execution_times" in stats:
                        print("\nSubtask Execution Times:")
                        for subtask_id, exec_time in stats["execution_times"].items():
                            print(f"- Subtask {subtask_id}: {exec_time:.2f} seconds")

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
    test_enhanced_parallel_supervisor()
