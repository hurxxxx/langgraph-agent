"""
Reranking Agent for Multi-Agent System

This module implements a reranking agent that can rerank search results or vector search results
using various reranking providers like Cohere and Pinecone.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

# Import LangChain components
try:
    from langchain_community.retrievers import ContextualCompressionRetriever
    from langchain_cohere import CohereRerank
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    import pinecone
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain components not available. Using mock implementations.")
    LANGCHAIN_AVAILABLE = False
    # Mock implementations for testing
    class ContextualCompressionRetriever:
        def __init__(self, base_compressor=None, base_retriever=None):
            self.base_compressor = base_compressor
            self.base_retriever = base_retriever
            
        def get_relevant_documents(self, query):
            return [{"content": f"Mock reranked result for {query}"}]
    
    class CohereRerank:
        def __init__(self, model=None, top_n=None):
            self.model = model
            self.top_n = top_n
            
        def compress_documents(self, documents, query):
            return documents
    
    class Pinecone:
        def __init__(self, index=None, embedding=None, text_key=None):
            self.index = index
            self.embedding = embedding
            self.text_key = text_key
            
        def as_retriever(self, search_type=None, search_kwargs=None):
            return MockRetriever()
    
    class MockRetriever:
        def get_relevant_documents(self, query):
            return [{"content": f"Mock result for {query}"}]
    
    class OpenAIEmbeddings:
        def __init__(self, model=None):
            self.model = model
    
    # Mock pinecone module
    class pinecone:
        @staticmethod
        def init(api_key=None, environment=None):
            pass
            
        class Index:
            def __init__(self, name):
                self.name = name


class RerankingAgentConfig(BaseModel):
    """Configuration for the reranking agent."""
    provider: Literal["cohere", "pinecone"] = "cohere"
    cohere_model: str = "rerank-english-v3.0"
    cohere_top_n: int = 5
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "gcp-starter"
    pinecone_index_name: str = "langgraph-agent"
    pinecone_text_key: str = "text"
    pinecone_alpha: float = 0.5  # Balance between vector and keyword search
    embedding_model: str = "text-embedding-3-small"
    use_cache: bool = True
    cache_ttl: int = 3600  # Cache TTL in seconds


class RerankingAgent:
    """
    Reranking agent that can rerank search results or vector search results
    using various reranking providers like Cohere and Pinecone.
    """
    
    def __init__(self, config=None):
        """
        Initialize the reranking agent.
        
        Args:
            config: Configuration for the reranking agent
        """
        self.config = config or RerankingAgentConfig()
        self.cache = {} if self.config.use_cache else None
        
        # Initialize reranker based on provider
        if LANGCHAIN_AVAILABLE:
            try:
                if self.config.provider == "cohere":
                    self._init_cohere_reranker()
                elif self.config.provider == "pinecone":
                    self._init_pinecone_reranker()
                else:
                    raise ValueError(f"Unsupported reranking provider: {self.config.provider}")
            except Exception as e:
                print(f"Error initializing reranker: {str(e)}")
                self.reranker = None
        else:
            print("Using mock reranker")
            self.reranker = None
    
    def _init_cohere_reranker(self):
        """Initialize Cohere reranker."""
        self.reranker = CohereRerank(
            model=self.config.cohere_model,
            top_n=self.config.cohere_top_n
        )
        print(f"Initialized Cohere reranker with model: {self.config.cohere_model}")
    
    def _init_pinecone_reranker(self):
        """Initialize Pinecone hybrid search."""
        # Get API key from config or environment
        api_key = self.config.pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key not provided")
        
        # Initialize Pinecone
        pinecone.init(
            api_key=api_key,
            environment=self.config.pinecone_environment
        )
        
        # Get or create index
        try:
            index = pinecone.Index(self.config.pinecone_index_name)
        except Exception as e:
            print(f"Error getting Pinecone index: {str(e)}")
            raise
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        
        # Create Pinecone store
        pinecone_store = Pinecone(
            index=index,
            embedding=embeddings,
            text_key=self.config.pinecone_text_key
        )
        
        # Create hybrid retriever
        self.reranker = pinecone_store.as_retriever(
            search_type="hybrid",
            search_kwargs={"alpha": self.config.pinecone_alpha}
        )
        
        print(f"Initialized Pinecone hybrid search with index: {self.config.pinecone_index_name}")
    
    def _get_cache_key(self, query, documents):
        """
        Generate a cache key for the query and documents.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            
        Returns:
            str: Cache key
        """
        # Create a deterministic representation of the documents
        doc_str = json.dumps([str(doc) for doc in documents], sort_keys=True)
        return f"{query}:{hash(doc_str)}"
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents based on the query.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            
        Returns:
            List[Dict[str, Any]]: Reranked documents
        """
        # Check cache if enabled
        if self.cache is not None:
            cache_key = self._get_cache_key(query, documents)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                cache_time, result = cached_result
                # Check if cache is still valid
                if time.time() - cache_time < self.config.cache_ttl:
                    print(f"Using cached reranking result for query: {query}")
                    return result
        
        # If no reranker is available, return documents as is
        if not self.reranker:
            print("No reranker available, returning documents as is")
            return documents
        
        try:
            # Rerank documents based on provider
            if self.config.provider == "cohere":
                # Cohere reranker expects a specific format
                reranked_docs = self.reranker.compress_documents(documents, query)
            elif self.config.provider == "pinecone":
                # Pinecone hybrid search is a retriever, so we need to use it directly
                reranked_docs = self.reranker.get_relevant_documents(query)
            else:
                reranked_docs = documents
            
            # Cache result if caching is enabled
            if self.cache is not None:
                cache_key = self._get_cache_key(query, documents)
                self.cache[cache_key] = (time.time(), reranked_docs)
            
            return reranked_docs
        except Exception as e:
            print(f"Error reranking documents: {str(e)}")
            # Return original documents on error
            return documents
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state using the reranking agent.
        
        Args:
            state: Current state of the system
            
        Returns:
            Dict[str, Any]: Updated state after reranking
        """
        # Extract query and documents from state
        query = state["messages"][-1]["content"] if state.get("messages") else ""
        documents = state.get("documents", [])
        
        # If no documents are provided, check if there are agent outputs with documents
        if not documents and "agent_outputs" in state:
            for agent_output in state["agent_outputs"].values():
                if isinstance(agent_output, dict) and "documents" in agent_output:
                    documents.extend(agent_output["documents"])
                elif isinstance(agent_output, dict) and "results" in agent_output:
                    documents.extend(agent_output["results"])
        
        # If still no documents, return state as is
        if not documents:
            state["agent_outputs"]["reranking_agent"] = {
                "error": "No documents provided for reranking"
            }
            return state
        
        # Rerank documents
        reranked_docs = self.rerank(query, documents)
        
        # Update state with reranked documents
        state["agent_outputs"]["reranking_agent"] = {
            "reranked_documents": reranked_docs,
            "original_count": len(documents),
            "reranked_count": len(reranked_docs)
        }
        
        # Add a message with the reranking results
        if reranked_docs:
            message = f"Reranked {len(reranked_docs)} documents based on relevance to the query."
            state["messages"].append({"role": "assistant", "content": message})
        
        return state


# Example usage
if __name__ == "__main__":
    # This is just a placeholder for testing
    config = RerankingAgentConfig(
        provider="cohere",
        cohere_model="rerank-english-v3.0",
        cohere_top_n=3
    )
    
    agent = RerankingAgent(config=config)
    
    # Test with some documents
    documents = [
        {"content": "Document about climate change and its effects"},
        {"content": "Information about renewable energy sources"},
        {"content": "Article about electric vehicles and sustainability"},
        {"content": "Research on ocean pollution and plastic waste"},
        {"content": "Study on deforestation and biodiversity loss"}
    ]
    
    state = {
        "messages": [{"role": "user", "content": "Tell me about renewable energy"}],
        "documents": documents
    }
    
    result = agent(state)
    print(json.dumps(result["agent_outputs"]["reranking_agent"], indent=2))
