"""
Test script for DB RAG and Vector Storage/Search functionality.

This script tests:
1. PostgreSQL setup with pgvector extension
2. Vector storage functionality
3. Vector retrieval functionality
4. SQL RAG functionality

Usage:
    python -m tests.test_vector_db_rag
"""

import os
import sys
import time
import json
import psycopg2
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the necessary components
from src.agents.vector_storage_agent import VectorStorageAgent, VectorStorageAgentConfig, VectorStoreDocument
from src.agents.vector_retrieval_agent import VectorRetrievalAgent, VectorRetrievalAgentConfig
from src.agents.sql_rag_agent import SQLRAGAgent, SQLRAGAgentConfig

# Load environment variables
load_dotenv()

# PostgreSQL connection parameters from environment variables
VECTOR_DB_USER = os.getenv("VECTOR_DB_USER", "postgres")
VECTOR_DB_PASSWORD = os.getenv("VECTOR_DB_PASSWORD", "102938")
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT = os.getenv("VECTOR_DB_PORT", "5432")
VECTOR_DB_NAME = os.getenv("VECTOR_DB_NAME", "vectordb")
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING",
                                      f"postgresql://{VECTOR_DB_USER}:{VECTOR_DB_PASSWORD}@{VECTOR_DB_HOST}:{VECTOR_DB_PORT}/{VECTOR_DB_NAME}")

# Global variables
DB_PROVIDER = os.getenv("VECTOR_DB_PROVIDER", "pgvector")

# Test data
TEST_DOCUMENTS = [
    VectorStoreDocument(
        content="Python is a high-level, interpreted programming language known for its readability and simplicity.",
        metadata={"source": "programming_languages", "category": "python"}
    ),
    VectorStoreDocument(
        content="JavaScript is a scripting language that enables interactive web pages and is an essential part of web applications.",
        metadata={"source": "programming_languages", "category": "javascript"}
    ),
    VectorStoreDocument(
        content="SQL (Structured Query Language) is a domain-specific language used for managing and manipulating relational databases.",
        metadata={"source": "programming_languages", "category": "sql"}
    ),
    VectorStoreDocument(
        content="Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience.",
        metadata={"source": "ai", "category": "machine_learning"}
    ),
    VectorStoreDocument(
        content="Natural Language Processing (NLP) is a field of AI that gives computers the ability to understand text and spoken words.",
        metadata={"source": "ai", "category": "nlp"}
    )
]

TEST_QUERIES = [
    "Tell me about Python programming",
    "What is JavaScript used for?",
    "Explain SQL databases",
    "How does machine learning work?",
    "What is NLP in artificial intelligence?"
]

SQL_TEST_QUERIES = [
    "How many customers do we have in the database?",
    "List all products with their prices",
    "What are the total sales by product category?",
    "Find customers who have placed orders worth more than $500"
]


def check_pgvector_extension():
    """
    Check if the pgvector extension is available in the PostgreSQL database.

    Returns:
        bool: True if pgvector is available, False otherwise
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(PGVECTOR_CONNECTION_STRING)
        cursor = conn.cursor()

        # Check if pgvector extension exists
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        result = cursor.fetchone()

        # Close connection
        cursor.close()
        conn.close()

        return result is not None
    except Exception as e:
        print(f"Error checking pgvector extension: {str(e)}")
        return False


def setup_pgvector():
    """
    Set up the pgvector extension in the PostgreSQL database.

    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(PGVECTOR_CONNECTION_STRING)
        cursor = conn.cursor()

        # Create pgvector extension if it doesn't exist
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Commit changes
        conn.commit()

        # Close connection
        cursor.close()
        conn.close()

        print("pgvector extension set up successfully")
        return True
    except Exception as e:
        print(f"Error setting up pgvector extension: {str(e)}")
        return False


def test_vector_storage():
    """
    Test vector storage functionality.

    Returns:
        bool: True if test was successful, False otherwise
    """
    print("\n=== Testing Vector Storage ===")

    try:
        # Create vector storage agent
        vector_storage_agent = VectorStorageAgent(
            config=VectorStorageAgentConfig(
                store_type=DB_PROVIDER,
                collection_name="test_collection",
                connection_string=PGVECTOR_CONNECTION_STRING
            )
        )

        # Store documents
        print(f"Storing {len(TEST_DOCUMENTS)} documents...")
        result = vector_storage_agent.store_documents(TEST_DOCUMENTS)

        # Print result
        print(f"Storage result: {json.dumps(result, indent=2)}")

        # Check if documents were stored
        if result["count"] == len(TEST_DOCUMENTS):
            print("✅ All documents stored successfully")
            return True
        else:
            print("❌ Not all documents were stored")
            return False
    except Exception as e:
        print(f"❌ Error testing vector storage: {str(e)}")
        return False


def test_vector_retrieval():
    """
    Test vector retrieval functionality.

    Returns:
        bool: True if test was successful, False otherwise
    """
    print("\n=== Testing Vector Retrieval ===")

    try:
        # Create vector retrieval agent
        vector_retrieval_agent = VectorRetrievalAgent(
            config=VectorRetrievalAgentConfig(
                store_type=DB_PROVIDER,
                collection_name="test_collection",
                connection_string=PGVECTOR_CONNECTION_STRING,
                top_k=2
            )
        )

        # Test retrieval for each query
        for i, query in enumerate(TEST_QUERIES):
            print(f"\nQuery {i+1}: {query}")

            # Create state
            state = {
                "messages": [{"role": "user", "content": query}],
                "agent_outputs": {}
            }

            # Process query
            start_time = time.time()
            updated_state = vector_retrieval_agent(state)
            execution_time = time.time() - start_time

            # Print result
            print(f"Response: {updated_state['messages'][-1]['content']}")
            print(f"Execution time: {execution_time:.2f} seconds")

            # Print retrieved documents
            if "vector_retrieval" in updated_state["agent_outputs"]:
                retrieval_output = updated_state["agent_outputs"]["vector_retrieval"]
                print(f"Retrieved {len(retrieval_output['documents'])} documents")
                for j, doc in enumerate(retrieval_output["documents"]):
                    print(f"  Document {j+1}: {doc['content'][:100]}... (score: {retrieval_output['scores'][j]:.4f})")

        return True
    except Exception as e:
        print(f"❌ Error testing vector retrieval: {str(e)}")
        return False


def test_sql_rag():
    """
    Test SQL RAG functionality.

    Returns:
        bool: True if test was successful, False otherwise
    """
    print("\n=== Testing SQL RAG ===")

    try:
        # Create SQL RAG agent
        sql_rag_agent = SQLRAGAgent(
            config=SQLRAGAgentConfig(
                db_type="postgresql",
                connection_string=PGVECTOR_CONNECTION_STRING
            )
        )

        # Test SQL RAG for each query
        for i, query in enumerate(SQL_TEST_QUERIES):
            print(f"\nQuery {i+1}: {query}")

            # Create state
            state = {
                "messages": [{"role": "user", "content": query}],
                "agent_outputs": {}
            }

            # Process query
            start_time = time.time()
            updated_state = sql_rag_agent(state)
            execution_time = time.time() - start_time

            # Print result
            print(f"Response: {updated_state['messages'][-1]['content']}")
            print(f"Execution time: {execution_time:.2f} seconds")

            # Print SQL query
            if "sql_rag" in updated_state["agent_outputs"]:
                sql_output = updated_state["agent_outputs"]["sql_rag"]
                print(f"SQL Query: {sql_output['query']}")
                if "error" in sql_output and sql_output["error"]:
                    print(f"Error: {sql_output['error']}")
                else:
                    print(f"Row count: {sql_output['row_count']}")

        return True
    except Exception as e:
        print(f"❌ Error testing SQL RAG: {str(e)}")
        return False


def main():
    """Main function to run all tests."""
    global DB_PROVIDER

    print("=== DB RAG and Vector Storage/Search Test ===")
    print(f"Using connection string: {PGVECTOR_CONNECTION_STRING}")
    print(f"DB Provider: {DB_PROVIDER}")

    # Check if pgvector extension is available
    if not check_pgvector_extension():
        print("pgvector extension not found. Setting up...")
        if not setup_pgvector():
            print("⚠️ Failed to set up pgvector extension.")
            print("The pgvector extension is not installed on the PostgreSQL server.")
            print("To install pgvector, follow the instructions at: https://github.com/pgvector/pgvector")
            print("Continuing with tests using fallback to Chroma vector store...")
            # Update DB_PROVIDER to use Chroma instead
            DB_PROVIDER = "chroma"
            print(f"Updated DB Provider to: {DB_PROVIDER}")
    else:
        print("✅ pgvector extension is available")

    # Run tests
    storage_success = test_vector_storage()
    retrieval_success = test_vector_retrieval()
    sql_rag_success = test_sql_rag()

    # Print summary
    print("\n=== Test Summary ===")
    print(f"Vector Storage: {'✅ Passed' if storage_success else '❌ Failed'}")
    print(f"Vector Retrieval: {'✅ Passed' if retrieval_success else '❌ Failed'}")
    print(f"SQL RAG: {'✅ Passed' if sql_rag_success else '❌ Failed'}")


if __name__ == "__main__":
    main()
