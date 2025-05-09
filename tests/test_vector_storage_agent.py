"""
Unit tests for the Vector Storage Agent.

This module contains tests for the Vector Storage Agent, which is responsible for
storing and retrieving information from vector databases.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the Vector Storage Agent and related components
from src.agents.vector_storage_agent import VectorStorageAgent, VectorStorageAgentConfig


class TestVectorStorageAgentConfig(unittest.TestCase):
    """Tests for the VectorStorageAgentConfig class."""

    def test_default_config(self):
        """Test that default configuration is created correctly."""
        config = VectorStorageAgentConfig()
        
        # Check default values
        self.assertEqual(config.store_type, "chroma")
        self.assertEqual(config.collection_name, "default_collection")
        self.assertEqual(config.persist_directory, "./vector_db")
        self.assertEqual(config.embedding_model, "openai")
        self.assertEqual(config.openai_embedding_model, "text-embedding-3-small")

    def test_custom_config(self):
        """Test that custom configuration is created correctly."""
        config = VectorStorageAgentConfig(
            store_type="pgvector",
            collection_name="custom_collection",
            persist_directory="./custom_vector_db",
            embedding_model="huggingface",
            openai_embedding_model="text-embedding-ada-002"
        )
        
        # Check custom values
        self.assertEqual(config.store_type, "pgvector")
        self.assertEqual(config.collection_name, "custom_collection")
        self.assertEqual(config.persist_directory, "./custom_vector_db")
        self.assertEqual(config.embedding_model, "huggingface")
        self.assertEqual(config.openai_embedding_model, "text-embedding-ada-002")


class TestVectorStorageAgent(unittest.TestCase):
    """Tests for the VectorStorageAgent class."""

    @patch("src.agents.vector_storage_agent.ChatOpenAI")
    def test_initialization(self, mock_chat_openai):
        """Test that VectorStorageAgent is initialized correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a VectorStorageAgent with default config
        with patch("src.agents.vector_storage_agent.OpenAIEmbeddings") as mock_embeddings:
            with patch("src.agents.vector_storage_agent.Chroma") as mock_chroma:
                # Create mock instances
                mock_embeddings_instance = MagicMock()
                mock_embeddings.return_value = mock_embeddings_instance
                
                mock_chroma_instance = MagicMock()
                mock_chroma.return_value = mock_chroma_instance
                
                # Create the agent
                vector_storage_agent = VectorStorageAgent()
                
                # Check that LLM is initialized
                mock_chat_openai.assert_called_once()
                self.assertEqual(vector_storage_agent.llm, mock_llm)
                
                # Check that embeddings are initialized
                mock_embeddings.assert_called_once_with(model="text-embedding-3-small")
                self.assertEqual(vector_storage_agent.embeddings, mock_embeddings_instance)
                
                # Check that vector store is initialized
                mock_chroma.assert_called_once()
                self.assertEqual(vector_storage_agent.vector_store, mock_chroma_instance)

    @patch("src.agents.vector_storage_agent.ChatOpenAI")
    @patch("src.agents.vector_storage_agent.OpenAIEmbeddings")
    @patch("src.agents.vector_storage_agent.Chroma")
    def test_initialize_vector_store_chroma(self, mock_chroma, mock_embeddings, mock_chat_openai):
        """Test that _initialize_vector_store works correctly with Chroma."""
        # Create mock instances
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Create a VectorStorageAgent with Chroma config
        vector_storage_agent = VectorStorageAgent(
            config=VectorStorageAgentConfig(
                store_type="chroma",
                collection_name="test_collection",
                persist_directory="./test_vector_db"
            )
        )
        
        # Check that Chroma is initialized
        mock_chroma.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_embeddings_instance,
            persist_directory="./test_vector_db"
        )
        self.assertEqual(vector_storage_agent.vector_store, mock_chroma_instance)

    @patch("src.agents.vector_storage_agent.ChatOpenAI")
    @patch("src.agents.vector_storage_agent.OpenAIEmbeddings")
    @patch("src.agents.vector_storage_agent.PGVector")
    def test_initialize_vector_store_pgvector(self, mock_pgvector, mock_embeddings, mock_chat_openai):
        """Test that _initialize_vector_store works correctly with PGVector."""
        # Create mock instances
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_pgvector_instance = MagicMock()
        mock_pgvector.return_value = mock_pgvector_instance
        
        # Create a VectorStorageAgent with PGVector config
        vector_storage_agent = VectorStorageAgent(
            config=VectorStorageAgentConfig(
                store_type="pgvector",
                collection_name="test_collection",
                connection_string="postgresql://user:pass@localhost:5432/db"
            )
        )
        
        # Check that PGVector is initialized
        mock_pgvector.assert_called_once()
        self.assertEqual(vector_storage_agent.vector_store, mock_pgvector_instance)

    @patch("src.agents.vector_storage_agent.ChatOpenAI")
    def test_store_information(self, mock_chat_openai):
        """Test that _store_information works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a VectorStorageAgent with a mock vector store
        vector_storage_agent = VectorStorageAgent()
        vector_storage_agent.vector_store = MagicMock()
        vector_storage_agent.vector_store.add_texts.return_value = ["doc1", "doc2"]
        
        # Test storing information
        result = vector_storage_agent._store_information(
            "The capital of France is Paris.",
            {"source": "test"}
        )
        
        # Check that vector store is called
        vector_storage_agent.vector_store.add_texts.assert_called_once_with(
            ["The capital of France is Paris."],
            [{"source": "test"}]
        )
        
        # Check that the result is correct
        self.assertEqual(result, ["doc1", "doc2"])

    @patch("src.agents.vector_storage_agent.ChatOpenAI")
    def test_retrieve_information(self, mock_chat_openai):
        """Test that _retrieve_information works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a VectorStorageAgent with a mock vector store
        vector_storage_agent = VectorStorageAgent()
        vector_storage_agent.vector_store = MagicMock()
        
        # Mock the similarity search
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Paris is the capital of France."
        mock_doc1.metadata = {"source": "test1"}
        
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "The Eiffel Tower is in Paris."
        mock_doc2.metadata = {"source": "test2"}
        
        vector_storage_agent.vector_store.similarity_search.return_value = [mock_doc1, mock_doc2]
        
        # Test retrieving information
        result = vector_storage_agent._retrieve_information("capital of France", k=2)
        
        # Check that vector store is called
        vector_storage_agent.vector_store.similarity_search.assert_called_once_with("capital of France", k=2)
        
        # Check that the result is correct
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["content"], "Paris is the capital of France.")
        self.assertEqual(result[0]["metadata"], {"source": "test1"})
        self.assertEqual(result[1]["content"], "The Eiffel Tower is in Paris.")
        self.assertEqual(result[1]["metadata"], {"source": "test2"})

    @patch("src.agents.vector_storage_agent.ChatOpenAI")
    def test_invoke_store(self, mock_chat_openai):
        """Test that invoke works correctly for storing information."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "I've stored the information that Paris is the capital of France."
        mock_llm.invoke.return_value = mock_response
        
        # Create a VectorStorageAgent with a mock vector store
        vector_storage_agent = VectorStorageAgent()
        vector_storage_agent.vector_store = MagicMock()
        vector_storage_agent.vector_store.add_texts.return_value = ["doc1"]
        
        # Test with a store query
        state = {
            "messages": [{"role": "user", "content": "Store this information: The capital of France is Paris."}],
            "agent_outputs": {}
        }
        
        result = vector_storage_agent(state)
        
        # Check that vector store is called
        vector_storage_agent.vector_store.add_texts.assert_called_once()
        
        # Check that LLM is called
        mock_llm.invoke.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result["messages"][-1]["content"], "I've stored the information that Paris is the capital of France.")
        self.assertIn("vector_storage_agent", result["agent_outputs"])
        self.assertIn("stored", result["agent_outputs"]["vector_storage_agent"])

    @patch("src.agents.vector_storage_agent.ChatOpenAI")
    def test_invoke_retrieve(self, mock_chat_openai):
        """Test that invoke works correctly for retrieving information."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "The capital of France is Paris."
        mock_llm.invoke.return_value = mock_response
        
        # Create a VectorStorageAgent with a mock vector store
        vector_storage_agent = VectorStorageAgent()
        vector_storage_agent.vector_store = MagicMock()
        
        # Mock the similarity search
        mock_doc = MagicMock()
        mock_doc.page_content = "Paris is the capital of France."
        mock_doc.metadata = {"source": "test"}
        
        vector_storage_agent.vector_store.similarity_search.return_value = [mock_doc]
        
        # Test with a retrieve query
        state = {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "agent_outputs": {}
        }
        
        result = vector_storage_agent(state)
        
        # Check that vector store is called
        vector_storage_agent.vector_store.similarity_search.assert_called_once()
        
        # Check that LLM is called
        mock_llm.invoke.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result["messages"][-1]["content"], "The capital of France is Paris.")
        self.assertIn("vector_storage_agent", result["agent_outputs"])
        self.assertIn("retrieved", result["agent_outputs"]["vector_storage_agent"])


if __name__ == "__main__":
    unittest.main()
