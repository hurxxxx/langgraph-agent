"""
Base Document Generation Agent for Multi-Agent System

This module implements a base document generation agent that can:
- Generate structured documents based on user requirements
- Support different document formats and styles
- Provide a consistent interface for specialized document agents
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions
from src.utils.file_operations import (
    ensure_directory_exists,
    save_metadata,
    load_metadata,
    generate_unique_filename,
    verify_file_exists,
    get_file_size
)

# Import LangChain components
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    print("Warning: LangChain components not available. Using mock implementations.")
    # Mock implementations for testing
    class ChatOpenAI:
        def __init__(self, model=None, temperature=0, streaming=False):
            self.model = model
            self.temperature = temperature
            self.streaming = streaming

        def invoke(self, messages):
            return {"content": f"Response from {self.model} about document generation"}

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


class DocumentFormat(BaseModel):
    """Model for document format specifications."""
    title: str
    sections: List[str]
    style: str = "default"
    include_toc: bool = False
    include_references: bool = False
    include_appendices: bool = False


class DocumentGenerationResult(BaseModel):
    """Model for a document generation result."""
    content: str
    format: DocumentFormat
    word_count: int
    created_at: float = Field(default_factory=time.time)
    file_path: Optional[str] = None
    metadata_path: Optional[str] = None


class BaseDocumentAgentConfig(BaseModel):
    """Configuration for the base document generation agent."""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-opus-20240229"
    temperature: float = 0
    streaming: bool = True
    max_tokens: int = 4000
    # File storage configuration
    save_documents: bool = True
    documents_dir: str = "./generated_documents"
    metadata_dir: str = "./generated_documents/metadata"
    # System messages
    system_message: str = """
    You are a document generation agent that creates well-structured documents based on user requirements.
    
    Your job is to:
    1. Understand the document requirements
    2. Generate a document with appropriate structure and content
    3. Format the document according to the specified style
    4. Provide the document content and any relevant metadata
    
    Always ensure that the document is well-organized, coherent, and meets the user's requirements.
    """


class BaseDocumentAgent:
    """
    Base document generation agent that creates structured documents.
    """

    def __init__(self, config: BaseDocumentAgentConfig = BaseDocumentAgentConfig()):
        """
        Initialize the base document generation agent.

        Args:
            config: Configuration for the document generation agent
        """
        self.config = config
        
        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=config.openai_model if config.llm_provider == "openai" else config.anthropic_model,
                temperature=config.temperature,
                streaming=config.streaming
            )
        except Exception as e:
            print(f"Warning: Could not initialize ChatOpenAI: {str(e)}")
            # Use a mock implementation
            class MockLLM:
                def invoke(self, messages):
                    return {"content": f"Mock response about document generation"}
            self.llm = MockLLM()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=config.system_message),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessage(content="Document format: {document_format}")
        ])
        
        # Ensure document and metadata directories exist
        if self.config.save_documents:
            self.documents_dir = ensure_directory_exists(self.config.documents_dir)
            self.metadata_dir = ensure_directory_exists(self.config.metadata_dir)
            print(f"Document generation agent initialized with document directory: {self.documents_dir}")
            print(f"Document generation agent initialized with metadata directory: {self.metadata_dir}")

    def _extract_document_requirements(self, message: str) -> Dict[str, Any]:
        """
        Extract document requirements from the message.

        Args:
            message: User message

        Returns:
            Dict[str, Any]: Extracted document requirements
        """
        # This is a simplified implementation
        # In a real system, you would use the LLM to extract the requirements
        
        # Default requirements
        requirements = {
            "title": "Untitled Document",
            "sections": ["Introduction", "Main Content", "Conclusion"],
            "style": "default",
            "include_toc": False,
            "include_references": False,
            "include_appendices": False
        }
        
        # Extract title if specified
        if "title:" in message.lower():
            title_line = [line for line in message.split("\n") if "title:" in line.lower()]
            if title_line:
                requirements["title"] = title_line[0].split("title:")[1].strip()
        
        # Extract style if specified
        if "style:" in message.lower():
            style_line = [line for line in message.split("\n") if "style:" in line.lower()]
            if style_line:
                requirements["style"] = style_line[0].split("style:")[1].strip()
        
        # Extract sections if specified
        if "sections:" in message.lower():
            sections_start = message.lower().find("sections:")
            sections_text = message[sections_start:].split("\n", 1)[0]
            sections = sections_text.split("sections:")[1].strip()
            if sections:
                requirements["sections"] = [s.strip() for s in sections.split(",")]
        
        # Extract other options
        requirements["include_toc"] = "table of contents" in message.lower() or "toc" in message.lower()
        requirements["include_references"] = "references" in message.lower() or "bibliography" in message.lower()
        requirements["include_appendices"] = "appendix" in message.lower() or "appendices" in message.lower()
        
        return requirements

    def _save_document(self, content: str, format: DocumentFormat) -> Dict[str, Any]:
        """
        Save a document to disk.

        Args:
            content: Document content
            format: Document format

        Returns:
            Dict[str, Any]: Information about the saved document
        """
        if not self.config.save_documents:
            return {
                "file_path": None,
                "metadata_path": None
            }
        
        try:
            # Generate a unique filename
            file_path = generate_unique_filename(
                self.documents_dir,
                prefix=format.title.replace(" ", "_").lower(),
                extension=".md"
            )
            
            # Save the document
            with open(file_path, "w") as f:
                f.write(content)
            
            # Create metadata
            metadata = {
                "title": format.title,
                "sections": format.sections,
                "style": format.style,
                "include_toc": format.include_toc,
                "include_references": format.include_references,
                "include_appendices": format.include_appendices,
                "word_count": len(content.split()),
                "created_at": time.time(),
                "file_path": file_path
            }
            
            # Save metadata
            metadata_path = os.path.join(
                self.metadata_dir,
                f"{os.path.basename(file_path).split('.')[0]}.json"
            )
            save_metadata(metadata, metadata_path)
            
            return {
                "file_path": file_path,
                "metadata_path": metadata_path
            }
        except Exception as e:
            print(f"Error saving document: {str(e)}")
            return {
                "file_path": None,
                "metadata_path": None,
                "error": str(e)
            }

    def generate_document(self, message: str) -> DocumentGenerationResult:
        """
        Generate a document based on the message.

        Args:
            message: User message

        Returns:
            DocumentGenerationResult: Generated document
        """
        # Extract document requirements
        requirements = self._extract_document_requirements(message)
        
        # Create document format
        document_format = DocumentFormat(**requirements)
        
        # Generate document content
        response = self.llm.invoke(
            self.prompt.format(
                messages=[{"role": "user", "content": message}],
                document_format=document_format.model_dump_json()
            )
        )
        
        # Extract content
        if isinstance(response, dict):
            content = response.get("content", "")
        elif hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)
        
        # Save document if configured
        save_result = self._save_document(content, document_format)
        
        # Create result
        result = DocumentGenerationResult(
            content=content,
            format=document_format,
            word_count=len(content.split()),
            file_path=save_result.get("file_path"),
            metadata_path=save_result.get("metadata_path")
        )
        
        return result

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
        
        # Generate document
        try:
            result = self.generate_document(message)
            
            # Format the result for the LLM
            formatted_result = f"""
            Document generated successfully!
            Title: {result.format.title}
            Style: {result.format.style}
            Word Count: {result.word_count}
            """
            
            if result.file_path:
                formatted_result += f"\nSaved to: {result.file_path}"
            
            error = None
        except Exception as e:
            formatted_result = f"Error generating document: {str(e)}"
            error = str(e)
            result = None
        
        # Generate response using LLM
        response = self.llm.invoke(
            self.prompt.format(
                messages=state["messages"],
                document_format=formatted_result
            )
        )
        
        # Update state
        if result:
            state["agent_outputs"]["document_generation"] = result.model_dump()
        else:
            state["agent_outputs"]["document_generation"] = {"error": error}
        
        state["messages"].append({"role": "assistant", "content": response.content})
        
        return state


# Example usage
if __name__ == "__main__":
    # Create document generation agent
    document_agent = BaseDocumentAgent()
    
    # Test with a document generation request
    state = {
        "messages": [{"role": "user", "content": "Generate a report about climate change with sections: Introduction, Current Trends, Future Projections, Mitigation Strategies, Conclusion"}],
        "agent_outputs": {}
    }
    
    updated_state = document_agent(state)
    print(updated_state["messages"][-1]["content"])
