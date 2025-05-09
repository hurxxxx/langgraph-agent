"""
Image Generation Agent using LangGraph's create_react_agent

This module implements an image generation agent using LangGraph's create_react_agent function
and OpenAI's DALL-E image generation capabilities.
"""

import os
import json
import uuid
import time
import hashlib
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Image generation tools
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool

# LangGraph components
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph


class ImageGenerationAgentConfig(BaseModel):
    """
    Configuration for the image generation agent.
    """
    # LLM configuration
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    temperature: float = 0
    streaming: bool = True

    # Image generation configuration
    provider: Literal["dalle", "gpt-image"] = "gpt-image"
    dalle_model: str = "dall-e-3"
    gpt_image_model: str = "gpt-image-1"
    image_size: str = "1024x1024"
    image_quality: Literal["standard", "high"] = "standard"
    image_style: Optional[str] = None

    # File storage configuration
    save_images: bool = True
    images_dir: str = "./generated_images"
    metadata_dir: str = "./generated_images/metadata"

    # System messages
    system_message: str = """
    You are an image generation agent that creates high-quality images based on user descriptions.

    Your job is to:
    1. Analyze the user's image request carefully
    2. Refine the description to create a detailed, clear prompt for the image generation model
    3. Generate an image that matches the user's requirements
    4. Provide the image URL and a brief description of what was generated

    For Ghibli-style images, incorporate elements like:
    - Soft, painterly art style with attention to natural details
    - Vibrant but not overly saturated colors
    - Whimsical, fantastical elements blended with realistic settings
    - Attention to lighting, atmosphere, and environmental details

    Always ensure the image is appropriate and follows content guidelines.
    """


class ImageGenerationAgent:
    """
    Image generation agent using LangGraph's create_react_agent function.
    Supports DALL-E and GPT-4 Vision for image generation.
    """

    def __init__(self, config=None):
        """
        Initialize the image generation agent.

        Args:
            config: Configuration for the image generation agent
        """
        # Initialize configuration
        self.config = config or ImageGenerationAgentConfig()

        # Create directories if they don't exist
        if self.config.save_images:
            os.makedirs(self.config.images_dir, exist_ok=True)
            os.makedirs(self.config.metadata_dir, exist_ok=True)

        # Initialize LLM based on provider
        if self.config.llm_provider == "openai":
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

        # Initialize image generation tool
        self.image_tool = self._initialize_image_tool()

        # Create ReAct agent with system message
        self.agent = create_react_agent(
            self.llm,
            [self.image_tool],
            prompt=SystemMessage(content=self.config.system_message)
        )

        # Create agent graph
        AgentState = self._get_graph_state_schema()
        self.graph = StateGraph(AgentState)
        self.graph.add_node("agent", self.agent)
        self.graph.set_entry_point("agent")
        self.graph.add_edge("agent", END)
        self.compiled_graph = self.graph.compile()

    def _initialize_image_tool(self):
        """
        Initialize the image generation tool based on the provider.

        Returns:
            Tool: Initialized image generation tool
        """
        # Configure DALL-E API wrapper
        dalle_wrapper = DallEAPIWrapper(
            model=self.config.dalle_model if self.config.provider == "dalle" else self.config.gpt_image_model,
            size=self.config.image_size,
            quality=self.config.image_quality,
            style=self.config.image_style
        )

        # Create DALL-E tool
        return OpenAIDALLEImageGenerationTool(api_wrapper=dalle_wrapper)

    def _get_graph_state_schema(self):
        """
        Get the state schema for the agent graph.

        Returns:
            TypedDict: State schema class
        """
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph.message import add_messages

        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            agent_outcome: Optional[Dict[str, Any]]

        return AgentState

    def _save_image_metadata(self, prompt: str, image_url: str, file_path: str):
        """
        Save metadata for the generated image.

        Args:
            prompt: The prompt used to generate the image
            image_url: URL of the generated image
            file_path: Path where the image is saved

        Returns:
            str: Path to the metadata file
        """
        if not self.config.save_images:
            return None

        # Create a unique ID for the image
        image_id = str(uuid.uuid4())

        # Create metadata
        metadata = {
            "id": image_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model": self.config.dalle_model if self.config.provider == "dalle" else self.config.gpt_image_model,
            "size": self.config.image_size,
            "quality": self.config.image_quality,
            "style": self.config.image_style,
            "image_url": image_url,
            "file_path": file_path
        }

        # Save metadata to file
        metadata_path = os.path.join(self.config.metadata_dir, f"{image_id}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata_path

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state update in the multi-agent system.

        Args:
            state: Current state of the system

        Returns:
            Dict[str, Any]: Updated state
        """
        try:
            # Extract the prompt from the last message
            if "messages" in state and isinstance(state["messages"], list) and state["messages"]:
                if isinstance(state["messages"][-1], dict) and "content" in state["messages"][-1]:
                    prompt = state["messages"][-1]["content"]
                else:
                    prompt = str(state["messages"][-1])
            else:
                prompt = state.get("query", "")

            # Check if we have a subtask in the state (used by MCP)
            if "current_subtask" in state and state["current_subtask"]:
                subtask = state["current_subtask"]
                if "description" in subtask:
                    # Use the subtask description as the prompt
                    prompt = subtask["description"]

            # Create input for the agent
            agent_input = {"messages": [{"role": "user", "content": prompt}]}

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
            state["agent_outputs"]["image_generation_agent"] = {
                "result": result,
                "prompt": prompt
            }

            return state

        except Exception as e:
            # Handle errors gracefully
            error_message = f"Image generation agent encountered an error: {str(e)}"

            # Update state with error information
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["image_generation_agent"] = {
                "error": str(e),
                "has_error": True
            }

            # If this is a subtask in MCP, mark it for potential fallback
            if "current_subtask" in state:
                state["agent_outputs"]["image_generation_agent"]["needs_fallback"] = True

            # Add error response to messages
            if "messages" in state:
                state["messages"].append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error while generating the image: {str(e)}. Please try again with a different description."
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
            # Extract the prompt from the last message
            if "messages" in state and isinstance(state["messages"], list) and state["messages"]:
                if isinstance(state["messages"][-1], dict) and "content" in state["messages"][-1]:
                    prompt = state["messages"][-1]["content"]
                else:
                    prompt = str(state["messages"][-1])
            else:
                prompt = state.get("query", "")

            # Create input for the agent
            agent_input = {"messages": [{"role": "user", "content": prompt}]}

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
                    {"role": "assistant", "content": f"I apologize, but I encountered an error while generating the image: {str(e)}. Please try again with a different description."}
                ]
            }
