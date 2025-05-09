"""
ReAct Document Generation Agent using LangGraph's create_react_agent

This module implements a document generation agent using LangGraph's create_react_agent function
and various document generation tools.
"""

import os
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Literal, Union, Annotated
from typing_extensions import TypedDict
from datetime import datetime
from pydantic import BaseModel, Field

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool, Tool
from langchain_core.language_models import BaseLanguageModel

# LangGraph components
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Utility functions
from src.utils.file_operations import (
    ensure_directory_exists,
    save_metadata,
    generate_unique_filename
)


class DocumentFormat(BaseModel):
    """Model for document format specifications."""
    title: str
    sections: List[str]
    style: str = "default"
    include_toc: bool = False
    include_references: bool = False
    include_appendices: bool = False


class ReactDocumentAgentConfig(BaseModel):
    """Configuration for the ReAct document generation agent."""
    # LLM configuration
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    temperature: float = 0
    streaming: bool = True

    # Document generation configuration
    document_type: str = "generic"
    default_sections: List[str] = ["Introduction", "Main Content", "Conclusion"]
    default_style: str = "default"

    # File storage configuration
    save_documents: bool = True
    documents_dir: str = "./generated_documents"
    metadata_dir: str = "./generated_documents/metadata"

    # System messages
    system_message: str = """
    You are a document generation agent that creates high-quality, well-structured documents based on user requirements.

    Your job is to:
    1. Analyze the user's document request carefully
    2. Extract key requirements like title, sections, style, and content needs
    3. Generate a well-structured document that meets these requirements
    4. Save the document and provide a summary of what was created

    Use the available tools to:
    - Extract document requirements from the user's request
    - Generate the document content
    - Save the document to a file

    Always ensure the document is well-organized, coherent, and meets the user's requirements.
    """


class ReactDocumentAgent:
    """
    Document generation agent using LangGraph's create_react_agent function.
    Supports various document types and formats.
    """

    def __init__(self, config=None, llm=None):
        """
        Initialize the document generation agent.

        Args:
            config: Configuration for the document generation agent
            llm: Optional language model to use (if not provided, one will be created based on config)
        """
        # Initialize configuration
        self.config = config or ReactDocumentAgentConfig()

        # Create directories if they don't exist
        if self.config.save_documents:
            self.documents_dir = ensure_directory_exists(self.config.documents_dir)
            self.metadata_dir = ensure_directory_exists(self.config.metadata_dir)

        # Initialize LLM based on provider or use provided LLM
        if llm:
            self.llm = llm
        elif self.config.llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=self.config.openai_model,
                temperature=self.config.temperature,
                streaming=self.config.streaming
            )
        else:  # anthropic
            self.llm = ChatAnthropic(
                model=self.config.anthropic_model,
                temperature=self.config.temperature,
                streaming=self.config.streaming
            )

        # Initialize document generation tools
        self.tools = self._initialize_tools()

        # Create ReAct agent with system message
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=SystemMessage(content=self.config.system_message)
        )

        # Create agent graph
        self.graph = StateGraph(self._get_graph_state_schema())
        self.graph.add_node("agent", self.agent)
        self.graph.set_entry_point("agent")
        self.graph.add_edge("agent", END)
        self.compiled_graph = self.graph.compile()

    def _get_graph_state_schema(self):
        """
        Get the state schema for the agent graph.

        Returns:
            TypedDict: State schema class
        """
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            agent_outcome: Optional[Dict[str, Any]]

        return AgentState

    def _initialize_tools(self) -> List[Any]:
        """
        Initialize document generation tools.

        Returns:
            List[Any]: List of initialized tools
        """
        # Create a reference to the agent instance for use in the tools
        agent_instance = self

        # Define tool functions without decorators
        def extract_document_requirements_func(message: str) -> str:
            """
            Extract document requirements from the user's message.

            Args:
                message: The user's message containing document requirements

            Returns:
                str: JSON string of extracted requirements
            """
            # Default requirements
            requirements = {
                "title": "Untitled Document",
                "sections": agent_instance.config.default_sections,
                "style": agent_instance.config.default_style,
                "include_toc": False,
                "include_references": False,
                "include_appendices": False,
                "document_type": agent_instance.config.document_type
            }

            # Extract title if present
            if "title:" in message.lower():
                title_match = message.lower().split("title:")[1].split("\n")[0].strip()
                if title_match:
                    requirements["title"] = title_match

            # Extract sections if present
            if "sections:" in message.lower() or "section:" in message.lower():
                sections_text = message.lower().split("sections:")[1].split("\n")[0].strip() if "sections:" in message.lower() else message.lower().split("section:")[1].split("\n")[0].strip()
                if sections_text:
                    sections = [s.strip() for s in sections_text.split(",")]
                    requirements["sections"] = sections

            # Extract style if present
            if "style:" in message.lower():
                style_match = message.lower().split("style:")[1].split("\n")[0].strip()
                if style_match:
                    requirements["style"] = style_match

            # Check for table of contents
            requirements["include_toc"] = "table of contents" in message.lower() or "toc" in message.lower()

            # Check for references
            requirements["include_references"] = "references" in message.lower() or "bibliography" in message.lower()

            # Check for appendices
            requirements["include_appendices"] = "appendix" in message.lower() or "appendices" in message.lower()

            return json.dumps(requirements, indent=2)

        def generate_document_content_func(requirements_json: str) -> str:
            """
            Generate document content based on requirements.

            Args:
                requirements_json: JSON string of document requirements

            Returns:
                str: Generated document content
            """
            try:
                requirements = json.loads(requirements_json)

                # Create a prompt for document generation
                prompt = f"""
                Generate a {requirements.get('document_type', 'document')} with the following specifications:

                Title: {requirements.get('title', 'Untitled Document')}
                Sections: {', '.join(requirements.get('sections', agent_instance.config.default_sections))}
                Style: {requirements.get('style', agent_instance.config.default_style)}
                Include Table of Contents: {requirements.get('include_toc', False)}
                Include References: {requirements.get('include_references', False)}
                Include Appendices: {requirements.get('include_appendices', False)}

                The document should be well-structured, informative, and formatted in Markdown.
                """

                # Generate the document using the LLM
                response = agent_instance.llm.invoke([
                    {"role": "system", "content": f"You are a professional document writer specializing in {requirements.get('document_type', 'document')} creation."},
                    {"role": "user", "content": prompt}
                ])

                # Extract content from response
                if hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)

                return content

            except Exception as e:
                return f"Error generating document content: {str(e)}"

        def save_document_func(content: str, requirements_json: str) -> str:
            """
            Save the document to a file and return the file path.

            Args:
                content: Document content to save
                requirements_json: JSON string of document requirements

            Returns:
                str: JSON string with file path and metadata
            """
            if not agent_instance.config.save_documents:
                return json.dumps({"saved": False, "reason": "Document saving is disabled"})

            try:
                requirements = json.loads(requirements_json)

                # Create a unique filename
                title = requirements.get("title", "Untitled Document")
                sanitized_title = title.replace(" ", "_").lower()
                file_path = generate_unique_filename(
                    agent_instance.documents_dir,
                    prefix=sanitized_title,
                    extension=".md"
                )

                # Save the document
                with open(file_path, "w") as f:
                    f.write(content)

                # Count words
                word_count = len(content.split())

                # Create metadata
                metadata = {
                    "title": requirements.get("title", "Untitled Document"),
                    "document_type": requirements.get("document_type", agent_instance.config.document_type),
                    "sections": requirements.get("sections", agent_instance.config.default_sections),
                    "style": requirements.get("style", agent_instance.config.default_style),
                    "include_toc": requirements.get("include_toc", False),
                    "include_references": requirements.get("include_references", False),
                    "include_appendices": requirements.get("include_appendices", False),
                    "word_count": word_count,
                    "created_at": datetime.now().isoformat(),
                    "file_path": file_path
                }

                # Save metadata
                metadata_path = os.path.join(
                    agent_instance.metadata_dir,
                    f"{os.path.basename(file_path).split('.')[0]}.json"
                )
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                result = {
                    "saved": True,
                    "file_path": file_path,
                    "metadata_path": metadata_path,
                    "word_count": word_count,
                    "title": requirements.get("title", "Untitled Document")
                }

                return json.dumps(result, indent=2)

            except Exception as e:
                return json.dumps({"saved": False, "error": str(e)})

        # Create Tool objects
        extract_document_requirements_tool = Tool(
            name="extract_document_requirements",
            description="Extract document requirements from the user's message",
            func=extract_document_requirements_func
        )

        generate_document_content_tool = Tool(
            name="generate_document_content",
            description="Generate document content based on requirements",
            func=generate_document_content_func
        )

        save_document_tool = Tool(
            name="save_document",
            description="Save the document to a file and return the file path",
            func=save_document_func
        )

        return [extract_document_requirements_tool, generate_document_content_tool, save_document_tool]

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

            # Check if we have a subtask in the state (used by MCP)
            if "current_subtask" in state and state["current_subtask"]:
                subtask = state["current_subtask"]
                if "description" in subtask:
                    # Use the subtask description as the query
                    query = subtask["description"]

            # Create input for the agent
            agent_input = {"messages": [{"role": "user", "content": query}]}

            # Run the agent
            result = self.compiled_graph.invoke(agent_input)

            # Update the state with the agent's response
            if "messages" in result and result["messages"]:
                # Convert LangChain message objects to dictionaries if needed
                processed_messages = []
                for msg in result["messages"]:
                    if hasattr(msg, "content") and hasattr(msg, "type"):
                        processed_messages.append({
                            "role": "assistant" if msg.type == "ai" else msg.type,
                            "content": msg.content
                        })
                    else:
                        processed_messages.append(msg)

                state["messages"] = state.get("messages", [])[:-1] + processed_messages

            # Store agent outcome in the state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["document_generation_agent"] = {
                "result": result,
                "query": query
            }

            return state

        except Exception as e:
            # Handle errors gracefully
            error_message = f"Document generation agent encountered an error: {str(e)}"

            # Update state with error information
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["document_generation_agent"] = {
                "error": str(e),
                "has_error": True
            }

            # If this is a subtask in MCP, mark it for potential fallback
            if "current_subtask" in state:
                state["agent_outputs"]["document_generation_agent"]["needs_fallback"] = True

            # Add error response to messages
            if "messages" in state:
                state["messages"].append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error while generating the document: {str(e)}. Please try again with different requirements."
                })

            return state

    def stream(self, state: Dict[str, Any], stream_mode: str = "values"):
        """
        Stream the agent's response.

        Args:
            state: Current state of the system
            stream_mode: Streaming mode ("values" or "steps")

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
                # Process the chunk to handle LangChain message objects
                if "messages" in chunk and chunk["messages"]:
                    processed_messages = []
                    for msg in chunk["messages"]:
                        if hasattr(msg, "content") and hasattr(msg, "type"):
                            processed_messages.append({
                                "role": "assistant" if msg.type == "ai" else msg.type,
                                "content": msg.content
                            })
                        else:
                            processed_messages.append(msg)

                    chunk["messages"] = processed_messages

                yield chunk

        except Exception as e:
            # Handle errors gracefully
            yield {
                "messages": [
                    {"role": "user", "content": state.get("query", "")},
                    {"role": "assistant", "content": f"I apologize, but I encountered an error while generating the document: {str(e)}. Please try again with different requirements."}
                ]
            }
