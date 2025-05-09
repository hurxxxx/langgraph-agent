"""
Direct test script for the SQL RAG agent.
"""

import os
from dotenv import load_dotenv
from src.agents.sql_rag_agent import SQLRAGAgent, SQLRAGAgentConfig

# Load environment variables
load_dotenv()

def test_sql_rag_agent_direct():
    """Test the SQL RAG agent directly with a PostgreSQL database."""
    print("\n=== Testing SQL RAG agent with PostgreSQL database ===")
    config = SQLRAGAgentConfig(
        db_type="postgresql",
        connection_string="postgresql://postgres:102938@localhost:5432/langgraph_agent_db",
        llm_model="gpt-4o",
        streaming=False
    )

    sql_rag_agent = SQLRAGAgent(config)
    query = "What tables are in the database and how many rows does each table have?"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = sql_rag_agent(state)
    print("\nResponse (excerpt):")
    last_message = updated_state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    print(content[:500] + "..." if len(content) > 500 else content)

    # Test with a more complex query
    print("\n=== Testing SQL RAG agent with a more complex query ===")
    query = "Find the top 5 customers by order value and show their total spending"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = sql_rag_agent(state)
    print("\nResponse (excerpt):")
    last_message = updated_state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    print(content[:500] + "..." if len(content) > 500 else content)

    return updated_state

if __name__ == "__main__":
    test_sql_rag_agent_direct()
