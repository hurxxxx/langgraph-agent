"""
Test Vector Retrieval Agent

This script tests the vector retrieval agent's ability to retrieve documents from a vector store
based on semantic similarity to a query.
"""

import os
import sys
import json
import time
import shutil
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the vector retrieval agent
from src.agents.vector_retrieval_agent import VectorRetrievalAgent, VectorRetrievalAgentConfig

# Import LangChain components for setting up test data
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain components not available. Using mock implementations.")
    LANGCHAIN_AVAILABLE = False


def setup_test_vector_store():
    """
    Set up a test vector store with sample documents.

    Returns:
        str: Path to the test vector store
    """
    if not LANGCHAIN_AVAILABLE:
        print("LangChain components not available. Skipping vector store setup.")
        return "./test_vector_db"

    # Create a test directory for the vector store
    test_vector_db = "./test_vector_db"
    if os.path.exists(test_vector_db):
        shutil.rmtree(test_vector_db)

    # Create sample documents
    os.makedirs("./test_docs", exist_ok=True)

    # Create sample documents
    sample_docs = [
        {
            "filename": "paris.txt",
            "content": "Paris is the capital of France. It is known for the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
        },
        {
            "filename": "london.txt",
            "content": "London is the capital of the United Kingdom. It is known for Big Ben, the Tower of London, and Buckingham Palace."
        },
        {
            "filename": "new_york.txt",
            "content": "New York City is the largest city in the United States. It is known for the Statue of Liberty, Times Square, and Central Park."
        },
        {
            "filename": "tokyo.txt",
            "content": "Tokyo is the capital of Japan. It is known for Tokyo Tower, the Imperial Palace, and Shibuya Crossing."
        }
    ]

    # Write sample documents to files
    for doc in sample_docs:
        with open(f"./test_docs/{doc['filename']}", "w") as f:
            f.write(doc["content"])

    # Load documents
    documents = []
    for doc in sample_docs:
        loader = TextLoader(f"./test_docs/{doc['filename']}")
        documents.extend(loader.load())

    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=test_vector_db,
        collection_name="test_collection"
    )

    # Persist vector store
    vector_store.persist()

    # Clean up
    shutil.rmtree("./test_docs")

    return test_vector_db


def test_vector_retrieval_agent():
    """
    Test the vector retrieval agent.
    """
    print("Testing vector retrieval agent...")

    # Set up test vector store
    test_vector_db = setup_test_vector_store()

    # Initialize the vector retrieval agent
    vector_retrieval_agent = VectorRetrievalAgent(
        config=VectorRetrievalAgentConfig(
            store_type="chroma",
            collection_name="test_collection",
            persist_directory=test_vector_db,
            use_cache=True,
            cache_ttl=3600
        )
    )

    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Tell me about London",
        "What is New York known for?",
        "What are some famous landmarks in Tokyo?"
    ]

    for i, query in enumerate(test_queries):
        print(f"\nTest Query {i+1}: {query}")

        # Create state
        state = {
            "messages": [{"role": "user", "content": query}],
            "agent_outputs": {}
        }

        # Process the query
        start_time = time.time()
        updated_state = vector_retrieval_agent(state)
        total_time = time.time() - start_time

        # Print the response
        print(f"\nAgent Response (took {total_time:.2f} seconds):")
        print(updated_state["messages"][-1]["content"])

        # Print the retrieval results
        if "vector_retrieval" in updated_state["agent_outputs"]:
            retrieval_output = updated_state["agent_outputs"]["vector_retrieval"]
            print("\nExecution Time:", retrieval_output["execution_time"], "seconds")

            if retrieval_output["error"]:
                print("\nError:", retrieval_output["error"])
            else:
                print("\nRetrieved Documents (first 2):")
                for i, (doc, score) in enumerate(zip(retrieval_output["documents"][:2], retrieval_output["scores"][:2])):
                    print(f"\nDocument {i+1} (Score: {score:.4f}):")
                    print(doc["content"])

        # Wait a bit between queries
        if i < len(test_queries) - 1:
            print("\nWaiting 2 seconds before next query...")
            time.sleep(2)

    # Clean up
    if os.path.exists(test_vector_db):
        shutil.rmtree(test_vector_db)

    return True


def test_vector_retrieval_agent_with_cache():
    """
    Test the vector retrieval agent with caching.
    """
    print("\nTesting vector retrieval agent with caching...")

    # Set up test vector store with a different directory
    test_vector_db = "./test_vector_db_cache"
    if os.path.exists(test_vector_db):
        shutil.rmtree(test_vector_db)

    # Create a simple in-memory test without using Chroma

    # Initialize the vector retrieval agent
    vector_retrieval_agent = VectorRetrievalAgent(
        config=VectorRetrievalAgentConfig(
            store_type="chroma",
            collection_name="test_collection",
            persist_directory=test_vector_db,
            use_cache=True,
            cache_ttl=3600
        )
    )

    # Test query
    query = "What is the capital of France?"

    # First execution (should be slower)
    print(f"\nFirst execution of: {query}")
    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    start_time = time.time()
    updated_state = vector_retrieval_agent(state)
    first_execution_time = time.time() - start_time

    print(f"First execution took {first_execution_time:.2f} seconds")

    # Second execution (should be faster due to caching)
    print(f"\nSecond execution of: {query}")
    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    start_time = time.time()
    updated_state = vector_retrieval_agent(state)
    second_execution_time = time.time() - start_time

    print(f"Second execution took {second_execution_time:.2f} seconds")

    # Check if caching worked
    if second_execution_time < first_execution_time:
        print("\nCaching is working! Second execution was faster.")
    else:
        print("\nCaching might not be working as expected.")

    # Clean up
    if os.path.exists(test_vector_db):
        shutil.rmtree(test_vector_db)

    return True


def main():
    """Main function to run the tests."""
    # Load environment variables
    load_dotenv()

    # Run the tests
    test_vector_retrieval_agent()
    test_vector_retrieval_agent_with_cache()

    print("\nAll vector retrieval agent tests completed!")


if __name__ == "__main__":
    main()
