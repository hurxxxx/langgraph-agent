"""
Test script for the simplified supervisor agent with Serper search integration.

This script tests the supervisor agent's ability to delegate tasks to the search agent
using Serper for web search.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import supervisor and agents
from src.supervisor.supervisor import Supervisor, SupervisorConfig
from src.agents.search_agent import SearchAgent, SearchAgentConfig


def test_supervisor_with_serper():
    """
    Test the supervisor agent with Serper search integration.
    """
    print("Testing supervisor agent with Serper search integration...")

    # Load environment variables
    load_dotenv()

    # Check if required API keys are set
    serper_api_key = os.getenv("SERPER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not serper_api_key:
        print("Error: SERPER_API_KEY environment variable is not set.")
        print("Please set it in the .env file and try again.")
        return

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in the .env file and try again.")
        return

    try:
        # Initialize search agent with Serper
        print("Initializing search agent with Serper...")
        search_agent = SearchAgent(
            config=SearchAgentConfig(
                provider="serper",
                max_results=3
            )
        )

        # Initialize supervisor
        print("Initializing supervisor agent...")
        supervisor = Supervisor(
            config=SupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                temperature=0,
                streaming=False,
                system_message="""
                You are a supervisor agent that coordinates specialized agents.
                You have access to the following agents:
                - search_agent: For web searches using Serper (Google Search API)

                For this test, always use the search_agent to find information.
                """
            ),
            agents={"search_agent": search_agent}
        )

        # Test queries
        test_queries = [
            "What is LangGraph?",
            "Tell me about the latest developments in AI in 2025",
            "Who won the 2024 Olympics?"
        ]

        # Run tests
        for i, query in enumerate(test_queries):
            print(f"\n\nTest {i+1}: {query}")
            print("-" * 50)

            try:
                print(f"Processing query: {query}")
                # Process query with timeout handling
                start_time = time.time()
                result = supervisor.invoke(query=query, stream=False)
                end_time = time.time()

                print(f"Query processed in {end_time - start_time:.2f} seconds")

                # Extract the final response
                final_message = result["messages"][-1]["content"] if result["messages"] else ""

                print("\nResponse:")
                print(final_message)

                # Print agent outputs
                if "agent_outputs" in result:
                    print("\nAgent Outputs:")
                    for agent_name, output in result["agent_outputs"].items():
                        print(f"- {agent_name}: {output}")

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
    test_supervisor_with_serper()
