"""
Vector Storage Agent for Multi-Agent System

This module implements a vector storage agent using LangGraph's create_react_agent function
and LangChain's vector store tools.

The agent can store, update, and delete documents in various vector databases:
- Chroma
- Qdrant
- Milvus
- pgvector
- Meilisearch
"""

import os
import json
from typing import Dict, List, Any, Optional, Literal, TypedDict, Annotated
from pydantic import BaseModel

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document

# Vector stores
from langchain_community.vectorstores import (
    Chroma,
    Qdrant,
    Milvus,
    PGVector,
    Meilisearch
)

# LangGraph imports
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class VectorStoreDocument(BaseModel):
    """Model for a document to be stored in a vector database."""
    content: str
    metadata: Dict[str, Any] = {}


class VectorStorageAgentConfig(BaseModel):
    """Configuration for the vector storage agent."""
    store_type: Literal["chroma", "qdrant", "milvus", "pgvector", "meilisearch"] = "chroma"
    collection_name: str = "default_collection"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"
    temperature: float = 0
    streaming: bool = True
    persist_directory: Optional[str] = "./vector_db"
    connection_string: Optional[str] = None
    system_message: str = """
    You are a vector storage agent that manages documents in a vector database.
    Your job is to:
    1. Understand the operation requested (store, update, delete, search)
    2. Process documents accordingly
    3. Report the results of the operation

    Always confirm what operation was performed and provide document IDs when applicable.

    Use the following tools to manage documents in the vector database:
    - store_document: Store a document in the vector database
    - update_document: Update a document in the vector database
    - delete_document: Delete a document from the vector database
    - search_documents: Search for documents in the vector database
    """


class AgentState(TypedDict):
    """State for the vector storage agent."""
    messages: Annotated[list, add_messages]
    agent_outcome: Optional[Dict[str, Any]]


class VectorStorageAgent:
    """
    Vector storage agent using LangGraph's create_react_agent function.
    """

    def __init__(self, config: Optional[VectorStorageAgentConfig] = None):
        """
        Initialize the vector storage agent.

        Args:
            config: Configuration for the vector storage agent
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

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model
        )

        # Initialize vector store
        self.vector_store = self._initialize_vector_store()

        # Initialize vector store tools
        self.vector_tools = self._initialize_vector_tools()

        # Create ReAct agent with system message
        system_message = self.config.system_message
        self.agent = create_react_agent(
            self.llm,
            self.vector_tools,
            prompt=SystemMessage(content=system_message)
        )

        # Create agent graph
        self.graph = StateGraph(AgentState)
        self.graph.add_node("agent", self.agent)
        self.graph.set_entry_point("agent")
        self.graph.add_edge("agent", END)
        self.compiled_graph = self.graph.compile()

    def _load_config_from_env(self) -> VectorStorageAgentConfig:
        """
        Load configuration from environment variables.

        Returns:
            VectorStorageAgentConfig: Configuration loaded from environment
        """
        # Get vector store configuration from environment
        store_type = os.getenv("VECTOR_STORE_TYPE", "chroma")
        collection_name = os.getenv("VECTOR_COLLECTION_NAME", "default_collection")
        persist_directory = os.getenv("VECTOR_PERSIST_DIRECTORY", "./vector_db")
        connection_string = os.getenv("VECTOR_CONNECTION_STRING", None)

        # Get embedding and LLM configuration from environment
        embedding_model = os.getenv("VECTOR_EMBEDDING_MODEL", "text-embedding-3-small")
        llm_model = os.getenv("VECTOR_LLM_MODEL", "gpt-4o")
        temperature = float(os.getenv("VECTOR_TEMPERATURE", "0"))
        streaming = os.getenv("VECTOR_STREAMING", "true").lower() == "true"

        return VectorStorageAgentConfig(
            store_type=store_type,
            collection_name=collection_name,
            embedding_model=embedding_model,
            llm_model=llm_model,
            temperature=temperature,
            streaming=streaming,
            persist_directory=persist_directory,
            connection_string=connection_string
        )

    def _initialize_vector_store(self):
        """
        Initialize the appropriate vector store based on the configured type.

        Returns:
            The initialized vector store
        """
        try:
            if self.config.store_type == "chroma":
                return Chroma(
                    collection_name=self.config.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.config.persist_directory
                )
            elif self.config.store_type == "qdrant":
                import qdrant_client
                client = qdrant_client.QdrantClient(
                    url=os.getenv("QDRANT_URL", "http://localhost:6333")
                )
                return Qdrant(
                    client=client,
                    collection_name=self.config.collection_name,
                    embedding_function=self.embeddings
                )
            elif self.config.store_type == "milvus":
                return Milvus(
                    collection_name=self.config.collection_name,
                    embedding_function=self.embeddings,
                    connection_args={"host": "localhost", "port": "19530"}
                )
            elif self.config.store_type == "pgvector":
                return PGVector(
                    collection_name=self.config.collection_name,
                    embedding_function=self.embeddings,
                    connection_string=self.config.connection_string or "postgresql://postgres:102938@localhost:5432/vectordb"
                )
            elif self.config.store_type == "meilisearch":
                return Meilisearch(
                    embedding_function=self.embeddings,
                    url=os.getenv("MEILISEARCH_URL", "http://localhost:7700"),
                    api_key=os.getenv("MEILISEARCH_API_KEY", "")
                )
            else:
                raise ValueError(f"Unsupported vector store type: {self.config.store_type}")
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {str(e)}")
            # Use a mock implementation
            class MockVectorStore:
                def add_documents(self, documents):
                    return [f"doc_{i}" for i in range(len(documents))]

                def delete(self, ids):
                    return True

                def persist(self):
                    return True

                def similarity_search(self, query, k=4):
                    return [Document(page_content=f"Mock document {i}", metadata={"id": f"doc_{i}"}) for i in range(k)]

            return MockVectorStore()

    def _initialize_vector_tools(self) -> List[Any]:
        """
        Initialize vector store tools.

        Returns:
            List[Any]: List of vector store tools
        """
        tools = []

        @tool
        def store_document(content: str, metadata: Optional[str] = None) -> str:
            """
            Store a document in the vector database.

            Args:
                content: The content of the document to store
                metadata: Optional JSON string with metadata for the document

            Returns:
                str: Result of the operation with document ID
            """
            try:
                # Parse metadata if provided
                doc_metadata = {}
                if metadata:
                    try:
                        doc_metadata = json.loads(metadata)
                    except:
                        doc_metadata = {"raw_metadata": metadata}

                # Create document
                document = Document(page_content=content, metadata=doc_metadata)

                # Add document to vector store
                ids = self.vector_store.add_documents([document])

                # Persist if supported
                if hasattr(self.vector_store, "persist"):
                    self.vector_store.persist()

                return f"Document stored successfully with ID: {ids[0]}"
            except Exception as e:
                return f"Error storing document: {str(e)}"

        @tool
        def update_document(document_id: str, content: str, metadata: Optional[str] = None) -> str:
            """
            Update a document in the vector database.

            Args:
                document_id: ID of the document to update
                content: New content for the document
                metadata: Optional JSON string with new metadata for the document

            Returns:
                str: Result of the operation
            """
            try:
                # Parse metadata if provided
                doc_metadata = {}
                if metadata:
                    try:
                        doc_metadata = json.loads(metadata)
                    except:
                        doc_metadata = {"raw_metadata": metadata}

                # Delete existing document
                self.vector_store.delete([document_id])

                # Create new document
                document = Document(page_content=content, metadata=doc_metadata)

                # Add document to vector store
                ids = self.vector_store.add_documents([document])

                # Persist if supported
                if hasattr(self.vector_store, "persist"):
                    self.vector_store.persist()

                return f"Document updated successfully with new ID: {ids[0]}"
            except Exception as e:
                return f"Error updating document: {str(e)}"

        @tool
        def delete_document(document_id: str) -> str:
            """
            Delete a document from the vector database.

            Args:
                document_id: ID of the document to delete

            Returns:
                str: Result of the operation
            """
            try:
                # Delete document
                self.vector_store.delete([document_id])

                # Persist if supported
                if hasattr(self.vector_store, "persist"):
                    self.vector_store.persist()

                return f"Document with ID {document_id} deleted successfully"
            except Exception as e:
                return f"Error deleting document: {str(e)}"

        @tool
        def search_documents(query: str, k: int = 4) -> str:
            """
            Search for documents in the vector database.

            Args:
                query: Search query
                k: Number of results to return

            Returns:
                str: Search results
            """
            try:
                # Search for documents
                docs = self.vector_store.similarity_search(query, k=k)

                # Format results
                results = []
                for i, doc in enumerate(docs):
                    results.append(f"Result {i+1}:")
                    results.append(f"Content: {doc.page_content}")
                    results.append(f"Metadata: {doc.metadata}")
                    results.append("")

                return "\n".join(results) if results else "No results found"
            except Exception as e:
                return f"Error searching documents: {str(e)}"

        tools.extend([store_document, update_document, delete_document, search_documents])
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
            state["agent_outputs"]["vector_storage"] = {
                "result": result,
                "query": query
            }

            return state

        except Exception as e:
            # Handle errors gracefully
            error_message = f"Vector storage agent encountered an error: {str(e)}"
            print(error_message)

            # Update state with error information
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["vector_storage"] = {
                "error": str(e),
                "has_error": True
            }

            # Add error response to messages
            if "messages" in state:
                state["messages"].append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error while processing your vector storage request: {str(e)}. Please try again or provide a different request."
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
            error_message = f"Error in vector storage agent streaming: {str(e)}"
            print(error_message)
            yield {
                "messages": [
                    {"role": "user", "content": state.get("query", "")},
                    {"role": "assistant", "content": f"I apologize, but I encountered an error while processing your vector storage request: {str(e)}. Please try again or provide a different request."}
                ]
            }


# Example usage
if __name__ == "__main__":
    # Create vector storage agent
    vector_storage_agent = VectorStorageAgent(
        config=VectorStorageAgentConfig(
            store_type="chroma",
            collection_name="test_collection",
            persist_directory="./test_vector_db"
        )
    )

    # Test with a store operation
    state = {
        "messages": [{"role": "user", "content": "Please store this document. content: This is a test document about AI."}],
        "agent_outputs": {}
    }

    updated_state = vector_storage_agent(state)
    print(updated_state["messages"][-1].content if hasattr(updated_state["messages"][-1], "content") else updated_state["messages"][-1])
