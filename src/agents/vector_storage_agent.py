"""
Vector Storage Agent for Multi-Agent System

This module implements a vector storage agent that can store, update, and delete documents
in various vector databases:
- Chroma
- Qdrant
- Milvus
- pgvector
- Meilisearch

The agent supports different operations (store, update, delete) and can be configured
to use different vector stores.
"""

import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal, Union
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

# Vector stores
from langchain_community.vectorstores import (
    Chroma,
    Qdrant,
    Milvus,
    PGVector,
    Meilisearch
)


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
    1. Understand the operation requested (store, update, delete)
    2. Process documents accordingly
    3. Report the results of the operation

    Always confirm what operation was performed and provide document IDs.
    """


class VectorStorageAgent:
    """
    Vector storage agent that manages documents in various vector databases.
    """

    def __init__(self, config: VectorStorageAgentConfig = VectorStorageAgentConfig()):
        """
        Initialize the vector storage agent.

        Args:
            config: Configuration for the vector storage agent
        """
        self.config = config

        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.temperature,
                streaming=config.streaming
            )
        except Exception as e:
            print(f"Warning: Could not initialize ChatOpenAI: {str(e)}")
            # Use a mock implementation
            class MockLLM:
                def invoke(self, messages):
                    return {"content": f"Mock response about vector storage operations"}
            self.llm = MockLLM()

        # Initialize embeddings
        try:
            self.embeddings = OpenAIEmbeddings(
                model=config.embedding_model
            )
        except Exception as e:
            print(f"Warning: Could not initialize OpenAIEmbeddings: {str(e)}")
            # Use a mock implementation
            class MockEmbeddings:
                def embed_documents(self, texts):
                    return [[0.1] * 1536 for _ in texts]
                def embed_query(self, text):
                    return [0.1] * 1536
            self.embeddings = MockEmbeddings()

        # Initialize vector store
        self.vector_store = self._initialize_vector_store()

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=config.system_message),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessage(content="Operation results: {operation_results}")
        ])

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
                    connection_string=self.config.connection_string or "postgresql://postgres:postgres@localhost:5432/vectordb"
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

            return MockVectorStore()

    def _parse_operation(self, message: str) -> Dict[str, Any]:
        """
        Parse the operation requested in the message.

        Args:
            message: User message

        Returns:
            Dict: Parsed operation details
        """
        # This is a simplified implementation
        # In a real system, you would use the LLM to parse the operation

        operation = {}

        if "store" in message.lower():
            operation["type"] = "store"
        elif "update" in message.lower():
            operation["type"] = "update"
        elif "delete" in message.lower():
            operation["type"] = "delete"
        else:
            operation["type"] = "unknown"

        # Extract document content (simplified)
        # In a real system, you would use more sophisticated parsing
        if "content:" in message:
            content_start = message.find("content:") + 8
            content_end = message.find("\n", content_start) if "\n" in message[content_start:] else len(message)
            operation["content"] = message[content_start:content_end].strip()

        return operation

    def store_documents(self, documents: List[VectorStoreDocument]) -> Dict[str, Any]:
        """
        Store documents in the vector database.

        Args:
            documents: List of documents to store

        Returns:
            Dict: Results of the operation
        """
        # Convert to LangChain documents
        lc_documents = [
            Document(page_content=doc.content, metadata=doc.metadata)
            for doc in documents
        ]

        # Add documents to vector store
        ids = self.vector_store.add_documents(lc_documents)

        # If the vector store supports persistence, persist it
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()

        return {
            "operation": "store",
            "document_ids": ids,
            "count": len(ids)
        }

    def update_documents(self, documents: List[VectorStoreDocument], ids: List[str]) -> Dict[str, Any]:
        """
        Update documents in the vector database.

        Args:
            documents: List of documents to update
            ids: List of document IDs to update

        Returns:
            Dict: Results of the operation
        """
        # This is a simplified implementation
        # Some vector stores don't support direct updates, so we delete and re-add

        # Delete existing documents
        self.delete_documents(ids)

        # Store new documents
        result = self.store_documents(documents)
        result["operation"] = "update"

        return result

    def delete_documents(self, ids: List[str]) -> Dict[str, Any]:
        """
        Delete documents from the vector database.

        Args:
            ids: List of document IDs to delete

        Returns:
            Dict: Results of the operation
        """
        # Delete documents from vector store
        self.vector_store.delete(ids)

        # If the vector store supports persistence, persist it
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()

        return {
            "operation": "delete",
            "document_ids": ids,
            "count": len(ids)
        }

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

        # Parse the operation
        operation = self._parse_operation(message)

        # Execute the operation
        if operation["type"] == "store":
            # Create a sample document for demonstration
            # In a real system, you would extract this from the message
            documents = [
                VectorStoreDocument(
                    content=operation.get("content", "Sample content"),
                    metadata={"source": "user_message"}
                )
            ]
            result = self.store_documents(documents)
        elif operation["type"] == "update":
            # This is a placeholder - in a real system, you would extract document IDs
            documents = [
                VectorStoreDocument(
                    content=operation.get("content", "Updated content"),
                    metadata={"source": "user_message", "updated": True}
                )
            ]
            result = self.update_documents(documents, ["doc_id_1"])
        elif operation["type"] == "delete":
            # This is a placeholder - in a real system, you would extract document IDs
            result = self.delete_documents(["doc_id_1"])
        else:
            result = {
                "operation": "unknown",
                "error": "Could not determine the requested operation"
            }

        # Generate response using LLM
        response = self.llm.invoke(
            self.prompt.format(
                messages=state["messages"],
                operation_results=str(result)
            )
        )

        # Update state
        state["agent_outputs"]["vector_storage"] = result
        state["messages"].append({"role": "assistant", "content": response.content})

        return state


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
    print(updated_state["messages"][-1]["content"])
