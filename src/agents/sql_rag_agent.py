"""
SQL RAG Agent for Multi-Agent System

This module implements a SQL RAG agent using LangGraph's create_react_agent function
and LangChain's SQL database tools.

The agent can:
- Generate SQL queries from natural language
- Execute SQL queries against a database
- Combine SQL results with LLM responses
- Support multiple database types (PostgreSQL, SQLite)
"""

import os
from typing import Dict, List, Any, Optional, Literal, TypedDict, Annotated
from pydantic import BaseModel

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# LangGraph imports
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class SQLRAGAgentConfig(BaseModel):
    """Configuration for the SQL RAG agent."""
    db_type: Literal["postgresql", "sqlite"] = "postgresql"
    connection_string: Optional[str] = None
    sqlite_path: Optional[str] = None
    llm_model: str = "gpt-4o"
    temperature: float = 0
    streaming: bool = True
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

    When you need to run a SQL query, use the query_sql_db tool.
    """


class AgentState(TypedDict):
    """State for the SQL RAG agent."""
    messages: Annotated[list, add_messages]
    agent_outcome: Optional[Dict[str, Any]]


class SQLRAGAgent:
    """
    SQL RAG agent using LangGraph's create_react_agent function.
    """

    def __init__(self, config: Optional[SQLRAGAgentConfig] = None):
        """
        Initialize the SQL RAG agent.

        Args:
            config: Configuration for the SQL RAG agent
        """
        # Load configuration from environment if not provided
        if config is None:
            config = self._load_config_from_env()

        self.config = config

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            streaming=config.streaming
        )

        # Initialize database connection
        self.db = self._initialize_database()

        # Initialize SQL tools
        self.sql_tools = self._initialize_sql_tools()

        # Create ReAct agent with system message
        system_message = self.config.system_message
        self.agent = create_react_agent(
            self.llm,
            self.sql_tools,
            prompt=SystemMessage(content=system_message)
        )

        # Create agent graph
        self.graph = StateGraph(AgentState)
        self.graph.add_node("agent", self.agent)
        self.graph.set_entry_point("agent")
        self.graph.add_edge("agent", END)
        self.compiled_graph = self.graph.compile()

    def _load_config_from_env(self) -> SQLRAGAgentConfig:
        """
        Load configuration from environment variables.

        Returns:
            SQLRAGAgentConfig: Configuration loaded from environment
        """
        # Get database configuration from environment
        db_type = os.getenv("SQL_DB_TYPE", "postgresql")
        connection_string = os.getenv("SQL_CONNECTION_STRING", "postgresql://postgres:102938@localhost:5432/langgraph_agent_db")
        sqlite_path = os.getenv("SQL_SQLITE_PATH", "database.sqlite")

        # Get LLM configuration from environment
        llm_model = os.getenv("SQL_LLM_MODEL", "gpt-4o")
        temperature = float(os.getenv("SQL_TEMPERATURE", "0"))
        streaming = os.getenv("SQL_STREAMING", "true").lower() == "true"

        return SQLRAGAgentConfig(
            db_type=db_type,
            connection_string=connection_string,
            sqlite_path=sqlite_path,
            llm_model=llm_model,
            temperature=temperature,
            streaming=streaming
        )

    def _initialize_database(self) -> Optional[SQLDatabase]:
        """
        Initialize the database connection.

        Returns:
            Optional[SQLDatabase]: Database connection or None if connection fails
        """
        try:
            if self.config.db_type == "postgresql":
                if not self.config.connection_string:
                    # Use default connection string
                    self.config.connection_string = "postgresql://postgres:102938@localhost:5432/langgraph_agent_db"

                return SQLDatabase.from_uri(
                    self.config.connection_string,
                    include_tables=None,  # Include all tables
                    sample_rows_in_table_info=3
                )

            elif self.config.db_type == "sqlite":
                if not self.config.sqlite_path:
                    # Use default SQLite path
                    self.config.sqlite_path = "database.sqlite"

                return SQLDatabase.from_uri(
                    f"sqlite:///{self.config.sqlite_path}",
                    include_tables=None,  # Include all tables
                    sample_rows_in_table_info=3
                )

            else:
                print(f"Unsupported database type: {self.config.db_type}")
                return None

        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            return None

    def _initialize_sql_tools(self) -> List[Any]:
        """
        Initialize SQL tools.

        Returns:
            List[Any]: List of SQL tools
        """
        tools = []

        if self.db:
            # Create SQL query tool
            query_sql_tool = QuerySQLDataBaseTool(
                db=self.db,
                description="Useful for when you need to query a SQL database to answer questions about data."
            )
            tools.append(query_sql_tool)
        else:
            # Create a mock SQL tool if database connection failed
            @tool
            def query_sql_db(query: str) -> str:
                """Execute a SQL query and return the results."""
                return "Error: Database connection not available."

            tools.append(query_sql_db)

        return tools

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state update in the multi-agent system.

        Args:
            state: Current state of the system

        Returns:
            Dict[str, Any]: Updated state
        """
        try:
            # Extract the query from the last message
            if "messages" in state and isinstance(state["messages"], list) and state["messages"]:
                if isinstance(state["messages"][-1], dict) and "content" in state["messages"][-1]:
                    query = state["messages"][-1]["content"]
                else:
                    query = str(state["messages"][-1])
            else:
                query = state.get("query", "")

            # Create input for the agent
            agent_input = {"messages": [{"role": "user", "content": query}]}

            # Run the agent
            result = self.compiled_graph.invoke(agent_input)

            # Update the state with the agent's response
            if "messages" in result and result["messages"]:
                state["messages"] = state.get("messages", [])[:-1] + result["messages"]

            # Store agent outcome in the state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["sql_rag"] = {
                "result": result,
                "query": query
            }

            return state

        except Exception as e:
            # Handle errors gracefully
            error_message = f"SQL RAG agent encountered an error: {str(e)}"
            print(error_message)

            # Update state with error information
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["sql_rag"] = {
                "error": str(e),
                "has_error": True
            }

            # Add error response to messages
            if "messages" in state:
                state["messages"].append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error while processing your SQL query: {str(e)}. Please try again or provide a different query."
                })

            return state

    async def astream(self, state: Dict[str, Any], stream_mode: str = "values") -> Any:
        """
        Stream the agent's response.

        Args:
            state: Current state of the system
            stream_mode: Streaming mode (values, updates, or steps)

        Yields:
            Dict[str, Any]: Streamed response
        """
        try:
            # Extract the query from the last message
            if "messages" in state and isinstance(state["messages"], list) and state["messages"]:
                if isinstance(state["messages"][-1], dict) and "content" in state["messages"][-1]:
                    query = state["messages"][-1]["content"]
                else:
                    query = str(state["messages"][-1])
            else:
                query = state.get("query", "")

            # Create input for the agent
            agent_input = {"messages": [{"role": "user", "content": query}]}

            # Stream the agent's response
            for chunk in self.compiled_graph.stream(
                agent_input,
                stream_mode=stream_mode
            ):
                yield chunk
        except Exception as e:
            # Handle errors gracefully
            error_message = f"Error in SQL RAG agent streaming: {str(e)}"
            print(error_message)
            yield {
                "messages": [
                    {"role": "user", "content": state.get("query", "")},
                    {"role": "assistant", "content": f"I apologize, but I encountered an error while processing your SQL query: {str(e)}. Please try again or provide a different query."}
                ]
            }


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
    print(updated_state["messages"][-1].content if hasattr(updated_state["messages"][-1], "content") else updated_state["messages"][-1])
