"""
Direct test script for the vector storage agent.
"""

import os
import shutil
from dotenv import load_dotenv
from src.agents.vector_storage_agent import VectorStorageAgent, VectorStorageAgentConfig

# Load environment variables
load_dotenv()

def test_vector_storage_agent_direct():
    """Test the vector storage agent directly with a Chroma vector store."""
    # Create test directory if it doesn't exist
    test_dir = "./test_vector_db"
    os.makedirs(test_dir, exist_ok=True)

    print("\n=== Testing vector storage agent with Chroma vector store ===")
    config = VectorStorageAgentConfig(
        store_type="chroma",
        collection_name="test_collection",
        persist_directory=test_dir,
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o",
        streaming=False
    )

    vector_storage_agent = VectorStorageAgent(config)

    # Test 1: Store a document
    print("\n=== Test 1: Store a document ===")
    query = "Please store this document with the content: 'This is a test document about artificial intelligence and machine learning.'"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = vector_storage_agent(state)
    print("\nResponse (excerpt):")
    last_message = updated_state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    print(content[:500] + "..." if len(content) > 500 else content)

    # Test 2: Search for documents
    print("\n=== Test 2: Search for documents ===")
    query = "Search for documents about artificial intelligence"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = vector_storage_agent(state)
    print("\nResponse (excerpt):")
    last_message = updated_state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    print(content[:500] + "..." if len(content) > 500 else content)

    # Test 3: Store a document with metadata
    print("\n=== Test 3: Store a document with metadata ===")
    query = """Please store this document with metadata:
    content: This is a document about neural networks and deep learning.
    metadata: {"author": "AI Researcher", "year": 2025, "tags": ["neural networks", "deep learning"]}
    """
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = vector_storage_agent(state)
    print("\nResponse (excerpt):")
    last_message = updated_state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    print(content[:500] + "..." if len(content) > 500 else content)

    # Clean up test directory
    # shutil.rmtree(test_dir, ignore_errors=True)

    return updated_state

if __name__ == "__main__":
    test_vector_storage_agent_direct()
