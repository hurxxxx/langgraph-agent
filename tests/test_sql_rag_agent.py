"""
Test SQL RAG Agent

This script tests the SQL RAG agent's ability to generate SQL queries from natural language,
execute them against a database, and provide insights based on the results.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the SQL RAG agent
from src.agents.sql_rag_agent import SQLRAGAgent, SQLRAGAgentConfig


def test_sql_rag_agent():
    """
    Test the SQL RAG agent.
    """
    print("Testing SQL RAG agent...")
    
    # Initialize the SQL RAG agent
    sql_rag_agent = SQLRAGAgent(
        config=SQLRAGAgentConfig(
            db_type="postgresql",
            connection_string="postgresql://postgres:102938@localhost:5432/langgraph_agent_db",
            use_cache=True,
            cache_ttl=3600
        )
    )
    
    # Test queries
    test_queries = [
        "How many customers do we have and what are their names?",
        "List all products with a price greater than $500",
        "What is the total amount of all orders?",
        "Which customer has placed the most orders?",
        "Show me the top 3 most expensive products"
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
        updated_state = sql_rag_agent(state)
        total_time = time.time() - start_time
        
        # Print the response
        print(f"\nAgent Response (took {total_time:.2f} seconds):")
        print(updated_state["messages"][-1]["content"])
        
        # Print the SQL query and result
        if "sql_rag" in updated_state["agent_outputs"]:
            sql_output = updated_state["agent_outputs"]["sql_rag"]
            print("\nSQL Query:")
            print(sql_output["query"])
            
            print("\nExecution Time:", sql_output["execution_time"], "seconds")
            print("Row Count:", sql_output["row_count"])
            
            if sql_output["error"]:
                print("\nError:", sql_output["error"])
            else:
                print("\nResult (first 3 rows):")
                for row in sql_output["result"][:3]:
                    print(row)
        
        # Wait a bit between queries
        if i < len(test_queries) - 1:
            print("\nWaiting 2 seconds before next query...")
            time.sleep(2)
    
    return True


def test_sql_rag_agent_with_cache():
    """
    Test the SQL RAG agent with caching.
    """
    print("\nTesting SQL RAG agent with caching...")
    
    # Initialize the SQL RAG agent
    sql_rag_agent = SQLRAGAgent(
        config=SQLRAGAgentConfig(
            db_type="postgresql",
            connection_string="postgresql://postgres:102938@localhost:5432/langgraph_agent_db",
            use_cache=True,
            cache_ttl=3600
        )
    )
    
    # Test query
    query = "How many customers do we have and what are their names?"
    
    # First execution (should be slower)
    print(f"\nFirst execution of: {query}")
    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }
    
    start_time = time.time()
    updated_state = sql_rag_agent(state)
    first_execution_time = time.time() - start_time
    
    print(f"First execution took {first_execution_time:.2f} seconds")
    
    # Second execution (should be faster due to caching)
    print(f"\nSecond execution of: {query}")
    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }
    
    start_time = time.time()
    updated_state = sql_rag_agent(state)
    second_execution_time = time.time() - start_time
    
    print(f"Second execution took {second_execution_time:.2f} seconds")
    
    # Check if caching worked
    if second_execution_time < first_execution_time:
        print("\nCaching is working! Second execution was faster.")
    else:
        print("\nCaching might not be working as expected.")
    
    return True


def test_sql_rag_agent_with_sqlite():
    """
    Test the SQL RAG agent with SQLite.
    """
    print("\nTesting SQL RAG agent with SQLite...")
    
    # Create a test SQLite database
    import sqlite3
    
    db_path = "test_database.sqlite"
    
    # Create database and tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        amount REAL NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES test_customers (id)
    )
    """)
    
    # Insert sample data
    cursor.execute("DELETE FROM test_orders")
    cursor.execute("DELETE FROM test_customers")
    
    cursor.execute("INSERT INTO test_customers VALUES (1, 'Alice', 'alice@example.com')")
    cursor.execute("INSERT INTO test_customers VALUES (2, 'Bob', 'bob@example.com')")
    cursor.execute("INSERT INTO test_customers VALUES (3, 'Charlie', 'charlie@example.com')")
    
    cursor.execute("INSERT INTO test_orders VALUES (1, 1, 100.0)")
    cursor.execute("INSERT INTO test_orders VALUES (2, 1, 200.0)")
    cursor.execute("INSERT INTO test_orders VALUES (3, 2, 300.0)")
    cursor.execute("INSERT INTO test_orders VALUES (4, 3, 400.0)")
    
    conn.commit()
    conn.close()
    
    # Initialize the SQL RAG agent with SQLite
    sql_rag_agent = SQLRAGAgent(
        config=SQLRAGAgentConfig(
            db_type="sqlite",
            sqlite_path=db_path,
            use_cache=True,
            cache_ttl=3600
        )
    )
    
    # Test query
    query = "How many customers do we have and what are their total order amounts?"
    
    print(f"\nQuery: {query}")
    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }
    
    updated_state = sql_rag_agent(state)
    
    # Print the response
    print("\nAgent Response:")
    print(updated_state["messages"][-1]["content"])
    
    # Print the SQL query and result
    if "sql_rag" in updated_state["agent_outputs"]:
        sql_output = updated_state["agent_outputs"]["sql_rag"]
        print("\nSQL Query:")
        print(sql_output["query"])
        
        print("\nExecution Time:", sql_output["execution_time"], "seconds")
        print("Row Count:", sql_output["row_count"])
        
        if sql_output["error"]:
            print("\nError:", sql_output["error"])
        else:
            print("\nResult:")
            for row in sql_output["result"]:
                print(row)
    
    # Clean up
    os.remove(db_path)
    
    return True


def main():
    """Main function to run the tests."""
    # Load environment variables
    load_dotenv()
    
    # Run the tests
    test_sql_rag_agent()
    test_sql_rag_agent_with_cache()
    test_sql_rag_agent_with_sqlite()
    
    print("\nAll SQL RAG agent tests completed!")


if __name__ == "__main__":
    main()
