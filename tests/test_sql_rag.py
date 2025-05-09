"""
Test script for SQL RAG functionality.

This script tests:
1. SQL RAG agent's ability to generate SQL queries from natural language
2. SQL RAG agent's ability to execute SQL queries
3. SQL RAG agent's ability to provide explanations of SQL query results

Usage:
    python -m tests.test_sql_rag
"""

import os
import sys
import time
import json
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the necessary components
from src.agents.sql_rag_agent import SQLRAGAgent, SQLRAGAgentConfig

# Load environment variables
load_dotenv()

# SQL RAG database connection parameters from environment variables
SQL_RAG_DB_USER = os.getenv("SQL_RAG_DB_USER", "postgres")
SQL_RAG_DB_PASSWORD = os.getenv("SQL_RAG_DB_PASSWORD", "102938")
SQL_RAG_DB_HOST = os.getenv("SQL_RAG_DB_HOST", "localhost")
SQL_RAG_DB_PORT = os.getenv("SQL_RAG_DB_PORT", "5432")
SQL_RAG_DB_NAME = os.getenv("SQL_RAG_DB_NAME", "langgraph_agent_db")
SQL_RAG_CONNECTION_STRING = os.getenv("SQL_RAG_CONNECTION_STRING", 
                                     f"postgresql://{SQL_RAG_DB_USER}:{SQL_RAG_DB_PASSWORD}@{SQL_RAG_DB_HOST}:{SQL_RAG_DB_PORT}/{SQL_RAG_DB_NAME}")

# Test queries
TEST_QUERIES = [
    "How many customers do we have in the database?",
    "List all products with their prices",
    "What are the total sales by product category?",
    "Find customers who have placed orders worth more than $500",
    "What is the average order value?",
    "Which product has the highest inventory count?",
    "List all orders with their status and customer information",
    "How many orders are in 'pending' status?"
]


def test_sql_rag():
    """
    Test SQL RAG functionality.
    
    Returns:
        bool: True if test was successful, False otherwise
    """
    print("\n=== Testing SQL RAG ===")
    print(f"Using connection string: {SQL_RAG_CONNECTION_STRING}")
    
    try:
        # Create SQL RAG agent
        sql_rag_agent = SQLRAGAgent(
            config=SQLRAGAgentConfig(
                db_type="postgresql",
                connection_string=SQL_RAG_CONNECTION_STRING
            )
        )
        
        # Test SQL RAG for each query
        for i, query in enumerate(TEST_QUERIES):
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
                    
                    # Print sample results if available
                    if "result" in sql_output and sql_output["result"]:
                        print("Sample results:")
                        for j, row in enumerate(sql_output["result"][:3]):  # Show up to 3 rows
                            print(f"  Row {j+1}: {row}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing SQL RAG: {str(e)}")
        return False


def main():
    """Main function to run the SQL RAG test."""
    print("=== SQL RAG Test ===")
    
    # Run test
    success = test_sql_rag()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"SQL RAG: {'✅ Passed' if success else '❌ Failed'}")


if __name__ == "__main__":
    main()
