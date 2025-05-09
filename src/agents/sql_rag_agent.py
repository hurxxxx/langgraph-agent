"""
SQL RAG Agent for Multi-Agent System

This module implements a SQL RAG agent that can:
- Generate SQL queries from natural language
- Execute SQL queries against a database
- Combine SQL results with LLM responses
- Support multiple database types (PostgreSQL, SQLite)
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
import psycopg2
import sqlite3

# Import utility functions
# Import utility functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.caching import cache_result, MemoryCache

# Import LangChain components
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_community.utilities import SQLDatabase
    from langchain_experimental.sql import SQLDatabaseChain
except ImportError:
    print("Warning: LangChain components not available. Using mock implementations.")
    # Mock implementations for testing
    class ChatOpenAI:
        def __init__(self, model=None, temperature=0, streaming=False):
            self.model = model
            self.temperature = temperature
            self.streaming = streaming

        def invoke(self, messages):
            return {"content": f"Response from {self.model} about {messages[-1]['content']}"}

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class SystemMessage:
        def __init__(self, content):
            self.content = content

    class MessagesPlaceholder:
        def __init__(self, variable_name):
            self.variable_name = variable_name

    class ChatPromptTemplate:
        @classmethod
        def from_messages(cls, messages):
            return cls()

        def format(self, **kwargs):
            return kwargs

    class SQLDatabase:
        @classmethod
        def from_uri(cls, uri):
            return cls()

        def run(self, query):
            return f"Mock result for query: {query}"

    class SQLDatabaseChain:
        @classmethod
        def from_llm(cls, llm, db, verbose=False):
            return cls()

        def run(self, query):
            return f"Mock result for query: {query}"


class SQLQueryResult(BaseModel):
    """Model for a SQL query result."""
    query: str
    result: List[Dict[str, Any]]
    execution_time: float
    row_count: int
    error: Optional[str] = None


class SQLRAGAgentConfig(BaseModel):
    """Configuration for the SQL RAG agent."""
    db_type: Literal["postgresql", "sqlite"] = "postgresql"
    connection_string: Optional[str] = None
    sqlite_path: Optional[str] = None
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-opus-20240229"
    temperature: float = 0
    streaming: bool = True
    max_tokens: int = 4000
    # Caching configuration
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    # System messages
    system_message: str = """
    You are a SQL RAG agent that can generate SQL queries from natural language,
    execute them against a database, and provide insights based on the results.

    Your job is to:
    1. Understand the user's question about data
    2. Generate an appropriate SQL query
    3. Execute the query against the database
    4. Analyze the results
    5. Provide a helpful response that answers the user's question

    Always explain your reasoning and the SQL query you generated.
    If there's an error in the query, explain what went wrong and suggest a fix.
    """
    sql_generation_system_message: str = """
    You are an expert SQL query generator. Your job is to convert natural language questions
    into SQL queries that can be executed against a database.

    The database has the following schema:
    {schema}

    Generate a SQL query that answers the following question: {question}

    Only return the SQL query without any explanations or markdown formatting.
    """


class SQLRAGAgent:
    """
    SQL RAG agent that can generate and execute SQL queries and provide insights.
    """

    def __init__(self, config: SQLRAGAgentConfig = SQLRAGAgentConfig()):
        """
        Initialize the SQL RAG agent.

        Args:
            config: Configuration for the SQL RAG agent
        """
        self.config = config

        # Initialize cache if enabled
        if self.config.use_cache:
            self.cache = MemoryCache(ttl=self.config.cache_ttl)
        else:
            self.cache = None

        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=config.openai_model,
                temperature=config.temperature,
                streaming=config.streaming
            )
        except Exception as e:
            print(f"Warning: Could not initialize ChatOpenAI: {str(e)}")
            # Use a mock implementation
            class MockLLM:
                def invoke(self, messages):
                    return {"content": f"Mock response about SQL query"}
            self.llm = MockLLM()

        # Initialize database connection
        self.db = self._initialize_database()

        # Get database schema
        self.schema = self._get_database_schema()

        # Create prompt templates
        self.sql_generation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=config.sql_generation_system_message.format(
                schema=self.schema,
                question="{question}"
            )),
            HumanMessage(content="{question}")
        ])

        self.response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=config.system_message),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessage(content="SQL query: {query}"),
            SystemMessage(content="Query result: {result}")
        ])

    def _initialize_database(self):
        """
        Initialize the database connection.

        Returns:
            Database connection
        """
        if self.config.db_type == "postgresql":
            if not self.config.connection_string:
                # Use default connection string
                self.config.connection_string = "postgresql://postgres:102938@localhost:5432/langgraph_agent_db"

            try:
                return SQLDatabase.from_uri(self.config.connection_string)
            except Exception as e:
                print(f"Warning: Could not connect to PostgreSQL: {str(e)}")
                return None

        elif self.config.db_type == "sqlite":
            if not self.config.sqlite_path:
                # Use default SQLite path
                self.config.sqlite_path = "database.sqlite"

            try:
                return SQLDatabase.from_uri(f"sqlite:///{self.config.sqlite_path}")
            except Exception as e:
                print(f"Warning: Could not connect to SQLite: {str(e)}")
                return None

        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")

    def _get_database_schema(self):
        """
        Get the database schema.

        Returns:
            str: Database schema
        """
        if not self.db:
            return "Database connection not available."

        try:
            if self.config.db_type == "postgresql":
                conn = psycopg2.connect(self.config.connection_string)
                cursor = conn.cursor()

                # Get all tables
                cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                """)
                tables = [row[0] for row in cursor.fetchall()]

                schema = []
                for table in tables:
                    # Get columns for each table
                    cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    """)
                    columns = cursor.fetchall()

                    schema.append(f"Table: {table}")
                    schema.append("Columns:")
                    for column in columns:
                        schema.append(f"  - {column[0]} ({column[1]}, {'NULL' if column[2] == 'YES' else 'NOT NULL'})")
                    schema.append("")

                conn.close()
                return "\n".join(schema)

            elif self.config.db_type == "sqlite":
                conn = sqlite3.connect(self.config.sqlite_path)
                cursor = conn.cursor()

                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                schema = []
                for table in tables:
                    # Get columns for each table
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()

                    schema.append(f"Table: {table}")
                    schema.append("Columns:")
                    for column in columns:
                        schema.append(f"  - {column[1]} ({column[2]}, {'NOT NULL' if column[3] else 'NULL'})")
                    schema.append("")

                conn.close()
                return "\n".join(schema)

            else:
                return "Unsupported database type."

        except Exception as e:
            print(f"Error getting database schema: {str(e)}")
            return "Error getting database schema."

    def _generate_sql_query(self, question: str) -> str:
        """
        Generate a SQL query from a natural language question.

        Args:
            question: Natural language question

        Returns:
            str: Generated SQL query
        """
        # Use cache if enabled
        if self.cache:
            cached_result = self.cache.get(f"sql_query:{question}")
            if cached_result:
                return cached_result
        if not self.db:
            return "SELECT 'Database connection not available.' AS error"

        try:
            # Generate SQL query using LLM
            response = self.llm.invoke(
                self.sql_generation_prompt.format(question=question)
            )

            # Extract query from response
            if isinstance(response, dict):
                query = response.get("content", "")
            elif hasattr(response, "content"):
                query = response.content
            else:
                query = str(response)

            # Clean up the query
            query = query.strip()
            if query.startswith("```sql"):
                query = query[6:]
            if query.endswith("```"):
                query = query[:-3]

            query = query.strip()

            # Cache the result if caching is enabled
            if self.cache:
                self.cache.set(f"sql_query:{question}", query)

            return query

        except Exception as e:
            print(f"Error generating SQL query: {str(e)}")
            return f"SELECT 'Error generating SQL query: {str(e)}' AS error"

    def _execute_sql_query(self, query: str) -> SQLQueryResult:
        """
        Execute a SQL query.

        Args:
            query: SQL query to execute

        Returns:
            SQLQueryResult: Query result
        """
        if not self.db:
            return SQLQueryResult(
                query=query,
                result=[],
                execution_time=0,
                row_count=0,
                error="Database connection not available."
            )

        try:
            # Execute query
            start_time = time.time()
            result = self.db.run(query)
            execution_time = time.time() - start_time

            # Parse result
            if isinstance(result, str):
                # Try to parse as JSON
                try:
                    parsed_result = json.loads(result)
                    row_count = len(parsed_result)
                except:
                    parsed_result = [{"result": result}]
                    row_count = 1
            elif isinstance(result, list):
                parsed_result = result
                row_count = len(result)
            else:
                parsed_result = [{"result": str(result)}]
                row_count = 1

            return SQLQueryResult(
                query=query,
                result=parsed_result,
                execution_time=execution_time,
                row_count=row_count
            )

        except Exception as e:
            print(f"Error executing SQL query: {str(e)}")
            return SQLQueryResult(
                query=query,
                result=[],
                execution_time=0,
                row_count=0,
                error=str(e)
            )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state update in the multi-agent system.

        Args:
            state: Current state of the system

        Returns:
            Dict[str, Any]: Updated state
        """
        # Extract the message from the last message
        message = state["messages"][-1]["content"]

        # Generate SQL query
        query = self._generate_sql_query(message)

        # Execute query
        query_result = self._execute_sql_query(query)

        # Format result for the LLM
        if query_result.error:
            formatted_result = f"Error: {query_result.error}"
        else:
            formatted_result = json.dumps(query_result.result, indent=2)

        # Generate response using LLM
        response = self.llm.invoke(
            self.response_prompt.format(
                messages=state["messages"],
                query=query,
                result=formatted_result
            )
        )

        # Update state
        state["agent_outputs"]["sql_rag"] = {
            "query": query,
            "result": query_result.result,
            "execution_time": query_result.execution_time,
            "row_count": query_result.row_count,
            "error": query_result.error
        }

        state["messages"].append({"role": "assistant", "content": response.content})

        return state


# Example usage
if __name__ == "__main__":
    # Create SQL RAG agent
    sql_rag_agent = SQLRAGAgent(
        config=SQLRAGAgentConfig(
            db_type="postgresql",
            connection_string="postgresql://postgres:102938@localhost:5432/langgraph_agent_db"
        )
    )

    # Test with a SQL query
    state = {
        "messages": [{"role": "user", "content": "How many customers do we have and what are their names?"}],
        "agent_outputs": {}
    }

    updated_state = sql_rag_agent(state)
    print(updated_state["messages"][-1]["content"])
