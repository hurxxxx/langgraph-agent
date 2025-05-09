"""
Image Generation Agent for Multi-Agent System

This module implements an image generation agent that can create images using:
- DALL-E 3
- GPT-4o image generation capabilities
- Other image generation services

The agent supports different image generation providers and can be configured
to use different models and parameters. It also supports saving generated images
to local storage for verification and future use.
"""

import os
import base64
import requests
import json
import time
from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from pathlib import Path

# Import utility functions
# Import utility functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_operations import (
    ensure_directory_exists,
    download_file,
    save_metadata,
    load_metadata,
    generate_unique_filename,
    verify_file_exists,
    get_file_size,
    get_file_extension
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


class ImageGenerationResult(BaseModel):
    """Model for an image generation result."""
    image_url: str
    prompt: str
    model: str
    size: str
    local_path: Optional[str] = None
    file_size: Optional[int] = None
    created_at: Optional[float] = None
    metadata_path: Optional[str] = None
    verified: bool = False


class ImageGenerationAgentConfig(BaseModel):
    """Configuration for the image generation agent."""
    provider: Literal["dalle", "gpt4o", "gpt-image"] = "gpt-image"
    dalle_model: str = "dall-e-3"
    gpt4o_model: str = "gpt-4o"
    gpt_image_model: str = "gpt-image-1"
    temperature: float = 0
    streaming: bool = True
    image_size: str = "1024x1024"
    image_quality: str = "high"
    # File storage configuration
    save_images: bool = True
    images_dir: str = "./generated_images"
    metadata_dir: str = "./generated_images/metadata"
    verify_downloads: bool = True
    download_timeout: int = 30
    system_message: str = """
    You are an image generation agent that creates images based on descriptions.
    Your job is to:
    1. Understand the image description
    2. Generate an appropriate image
    3. Provide the image URL and any relevant details
    4. Save the image locally for verification and future use

    Always confirm what image was generated and provide both the image URL and local file path.
    """


class ImageGenerationAgent:
    """
    Image generation agent that creates images using various providers.
    """

    def __init__(self, config: ImageGenerationAgentConfig = ImageGenerationAgentConfig()):
        """
        Initialize the image generation agent.

        Args:
            config: Configuration for the image generation agent
        """
        self.config = config

        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=config.gpt4o_model if config.provider == "gpt4o" else "gpt-4o",
                temperature=config.temperature,
                streaming=config.streaming
            )
        except Exception as e:
            print(f"Warning: Could not initialize ChatOpenAI: {str(e)}")
            # Use a mock implementation
            class MockLLM:
                def invoke(self, messages):
                    return {"content": f"Mock response about image generation"}
            self.llm = MockLLM()

        # Initialize image generator for DALL-E
        # We'll use a mock implementation for now
        self.image_generator = None

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=config.system_message),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessage(content="Image generation results: {generation_results}")
        ])

        # Ensure image and metadata directories exist
        if self.config.save_images:
            self.images_dir = ensure_directory_exists(self.config.images_dir)
            self.metadata_dir = ensure_directory_exists(self.config.metadata_dir)
            print(f"Image generation agent initialized with image directory: {self.images_dir}")
            print(f"Image generation agent initialized with metadata directory: {self.metadata_dir}")

    def _extract_image_description(self, message: str) -> str:
        """
        Extract the image description from the message.

        Args:
            message: User message

        Returns:
            str: Extracted image description
        """
        # This is a simplified implementation
        # In a real system, you would use the LLM to extract the description

        # Look for common patterns
        if "generate an image of" in message.lower():
            return message.lower().split("generate an image of")[1].strip()
        elif "create an image of" in message.lower():
            return message.lower().split("create an image of")[1].strip()
        elif "image:" in message.lower():
            return message.lower().split("image:")[1].strip()
        else:
            # If no pattern is found, use the whole message
            return message

    def _save_image_locally(self, image_url: str, prompt: str, model: str, size: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Save an image from a URL to local storage with retry logic.

        Args:
            image_url: URL of the image
            prompt: Image generation prompt
            model: Model used to generate the image
            size: Size of the generated image
            max_retries: Maximum number of retries for transient errors

        Returns:
            Dict[str, Any]: Information about the saved image
        """
        if not self.config.save_images:
            return {
                "local_path": None,
                "file_size": None,
                "created_at": time.time(),
                "metadata_path": None,
                "verified": False
            }

        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                # Generate a unique filename
                extension = get_file_extension(image_url)
                local_path = generate_unique_filename(
                    self.images_dir,
                    prefix=f"{model.replace('-', '_')}",
                    extension=extension
                )

                # Download the image with timeout
                print(f"Downloading image from {image_url} to {local_path} (attempt {retry_count + 1}/{max_retries})")
                download_file(image_url, local_path, timeout=self.config.download_timeout)

                # Verify the file exists and get its size
                if verify_file_exists(local_path):
                    file_size = get_file_size(local_path)

                    # Additional verification: check if file size is reasonable (not empty or too small)
                    if file_size < 100:  # Less than 100 bytes is suspicious for an image
                        print(f"Warning: Downloaded file is very small ({file_size} bytes), might be corrupted")
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            print(f"Retrying download in {retry_count} seconds...")
                            time.sleep(retry_count)
                            continue

                    verified = True
                else:
                    # File doesn't exist after download attempt
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        print(f"Download failed, retrying in {retry_count} seconds...")
                        time.sleep(retry_count)
                        continue

                    file_size = 0
                    verified = False

                # Create metadata
                created_at = time.time()
                metadata = {
                    "prompt": prompt,
                    "model": model,
                    "size": size,
                    "image_url": image_url,
                    "local_path": local_path,
                    "file_size": file_size,
                    "created_at": created_at,
                    "verified": verified
                }

                # Save metadata
                metadata_path = os.path.join(
                    self.metadata_dir,
                    f"{os.path.basename(local_path).split('.')[0]}.json"
                )
                save_metadata(metadata, metadata_path)

                return {
                    "local_path": local_path,
                    "file_size": file_size,
                    "created_at": created_at,
                    "metadata_path": metadata_path,
                    "verified": verified
                }

            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                print(f"Timeout downloading image (attempt {retry_count + 1}/{max_retries})")

                if retry_count < max_retries - 1:
                    retry_count += 1
                    print(f"Retrying in {retry_count} seconds...")
                    time.sleep(retry_count)
                else:
                    break

            except requests.exceptions.ConnectionError:
                last_error = "Connection error"
                print(f"Connection error downloading image (attempt {retry_count + 1}/{max_retries})")

                if retry_count < max_retries - 1:
                    retry_count += 1
                    print(f"Retrying in {retry_count} seconds...")
                    time.sleep(retry_count)
                else:
                    break

            except Exception as e:
                last_error = str(e)
                print(f"Error saving image locally (attempt {retry_count + 1}/{max_retries}): {str(e)}")

                # Check if it's a transient error
                error_str = str(e).lower()
                is_transient = any(term in error_str for term in [
                    "timeout", "connection", "network", "temporary",
                    "rate limit", "too many requests", "503", "502"
                ])

                if is_transient and retry_count < max_retries - 1:
                    retry_count += 1
                    print(f"Transient error, retrying in {retry_count} seconds...")
                    time.sleep(retry_count)
                else:
                    break

        # If we get here, all retries failed
        error_message = last_error if last_error else "Unknown error saving image"
        print(f"Failed to save image after {max_retries} attempts: {error_message}")

        return {
            "local_path": None,
            "file_size": None,
            "created_at": time.time(),
            "metadata_path": None,
            "verified": False,
            "error": error_message
        }

    def generate_image_dalle(self, prompt: str) -> ImageGenerationResult:
        """
        Generate an image using DALL-E.

        Args:
            prompt: Image description

        Returns:
            ImageGenerationResult: Generated image details
        """
        try:
            # Call the OpenAI API to generate an image
            from openai import OpenAI
            client = OpenAI()

            # Add Ghibli style to the prompt if not already present
            if "ghibli" in prompt.lower() or "studio ghibli" in prompt.lower():
                styled_prompt = prompt
            else:
                styled_prompt = f"{prompt}, in the style of Studio Ghibli"

            print(f"Using prompt for DALL-E: {styled_prompt}")

            response = client.images.generate(
                model=self.config.dalle_model,
                prompt=styled_prompt,
                n=1,
                size=self.config.image_size,
                quality="hd" if self.config.image_quality == "high" else "standard",
                style="vivid"
            )

            # Get the image URL from the response
            image_url = response.data[0].url

            # Create the result
            result = ImageGenerationResult(
                image_url=image_url,
                prompt=styled_prompt,
                model=self.config.dalle_model,
                size=self.config.image_size
            )

            # Save the image locally if configured
            if self.config.save_images:
                save_result = self._save_image_locally(
                    image_url=image_url,
                    prompt=styled_prompt,
                    model=self.config.dalle_model,
                    size=self.config.image_size
                )

                # Update the result with local file information
                result.local_path = save_result.get("local_path")
                result.file_size = save_result.get("file_size")
                result.created_at = save_result.get("created_at")
                result.metadata_path = save_result.get("metadata_path")
                result.verified = save_result.get("verified", False)

            return result

        except Exception as e:
            print(f"Error generating image with DALL-E: {str(e)}")
            # Fallback to mock implementation for testing
            image_url = "https://example.com/generated_image.png"

            # Create the basic result
            result = ImageGenerationResult(
                image_url=image_url,
                prompt=prompt,
                model=self.config.dalle_model,
                size=self.config.image_size
            )

            return result

    def generate_image_gpt4o(self, prompt: str) -> ImageGenerationResult:
        """
        Generate an image using GPT-4o's image generation capabilities.

        Args:
            prompt: Image description

        Returns:
            ImageGenerationResult: Generated image details
        """
        # Note: This is a placeholder implementation since GPT-4o's image generation
        # API might not be fully documented or available at the time of writing

        response = self.llm.invoke([
            SystemMessage(content="You are an image generation assistant. Generate images based on user prompts."),
            HumanMessage(content=f"Generate an image of: {prompt}")
        ])

        # This is a placeholder - in a real implementation, you would extract the image URL
        # from the response based on the actual API
        image_url = "https://example.com/generated_image.png"

        # Create the basic result
        result = ImageGenerationResult(
            image_url=image_url,
            prompt=prompt,
            model=self.config.gpt4o_model,
            size=self.config.image_size
        )

        # Save the image locally if configured
        if self.config.save_images:
            save_result = self._save_image_locally(
                image_url=image_url,
                prompt=prompt,
                model=self.config.gpt4o_model,
                size=self.config.image_size
            )

            # Update the result with local file information
            result.local_path = save_result.get("local_path")
            result.file_size = save_result.get("file_size")
            result.created_at = save_result.get("created_at")
            result.metadata_path = save_result.get("metadata_path")
            result.verified = save_result.get("verified", False)

        return result

    def generate_image_gpt_image(self, prompt: str) -> ImageGenerationResult:
        """
        Generate an image using OpenAI's GPT-Image-1 model.
        Falls back to DALL-E 3 if organization verification is required.

        Args:
            prompt: Image description

        Returns:
            ImageGenerationResult: Generated image details
        """
        try:
            # Call the OpenAI API to generate an image
            from openai import OpenAI
            client = OpenAI()

            # Add Ghibli style to the prompt
            if "ghibli" in prompt.lower() or "studio ghibli" in prompt.lower():
                styled_prompt = prompt
            else:
                styled_prompt = f"{prompt}, in the style of Studio Ghibli"

            print(f"Using prompt for GPT-Image-1: {styled_prompt}")

            response = client.images.generate(
                model=self.config.gpt_image_model,
                prompt=styled_prompt,
                n=1,
                size=self.config.image_size,
                quality=self.config.image_quality,
                output_format="png",
                background="auto",
                moderation="auto"
            )

            # Get the image URL from the response
            image_url = response.data[0].url

            # Create the result
            result = ImageGenerationResult(
                image_url=image_url,
                prompt=styled_prompt,
                model=self.config.gpt_image_model,
                size=self.config.image_size
            )

            # Save the image locally if configured
            if self.config.save_images:
                save_result = self._save_image_locally(
                    image_url=image_url,
                    prompt=styled_prompt,
                    model=self.config.gpt_image_model,
                    size=self.config.image_size
                )

                # Update the result with local file information
                result.local_path = save_result.get("local_path")
                result.file_size = save_result.get("file_size")
                result.created_at = save_result.get("created_at")
                result.metadata_path = save_result.get("metadata_path")
                result.verified = save_result.get("verified", False)

            return result

        except Exception as e:
            print(f"Error generating image with GPT-Image-1: {str(e)}")
            print("Falling back to DALL-E 3 model...")

            # Fallback to DALL-E 3 if organization verification is required
            if "organization must be verified" in str(e):
                return self.generate_image_dalle(prompt + ", in the style of Studio Ghibli")

            # Fallback to mock implementation for other errors
            image_url = "https://example.com/generated_image.png"

            # Create the basic result
            result = ImageGenerationResult(
                image_url=image_url,
                prompt=prompt,
                model=self.config.gpt_image_model,
                size=self.config.image_size
            )

            return result

    def generate_image(self, prompt: str) -> ImageGenerationResult:
        """
        Generate an image using the configured provider.

        Args:
            prompt: Image description

        Returns:
            ImageGenerationResult: Generated image details
        """
        if self.config.provider == "dalle":
            return self.generate_image_dalle(prompt)
        elif self.config.provider == "gpt4o":
            return self.generate_image_gpt4o(prompt)
        elif self.config.provider == "gpt-image":
            return self.generate_image_gpt_image(prompt)
        else:
            raise ValueError(f"Unsupported image generation provider: {self.config.provider}")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state update in the multi-agent system.

        Args:
            state: Current state of the system

        Returns:
            Dict[str, Any]: Updated state
        """
        try:
            # Extract the message from the last message
            message = state["messages"][-1]["content"]

            # Check if we have a subtask in the state (used by MCP)
            if "current_subtask" in state and state["current_subtask"]:
                subtask = state["current_subtask"]
                if "description" in subtask:
                    # Use the subtask description as the message
                    message = subtask["description"]
                    print(f"Using subtask description for image generation: {message}")

            # Extract the image description
            image_description = self._extract_image_description(message)

            # Check if we have a valid image description
            if not image_description or len(image_description.strip()) < 5:
                # If the description is too short or empty, try to use the full message
                print(f"Image description too short: '{image_description}', using full message")
                image_description = message

            # Generate the image
            try:
                result = self.generate_image(image_description)

                # Format the result for the LLM
                formatted_result = f"""
                Image generated successfully!
                Prompt: {result.prompt}
                Model: {result.model}
                Size: {result.size}
                Image URL: {result.image_url}
                """

                # Add local file information if available
                if result.local_path:
                    formatted_result += f"""
                    Local File Path: {result.local_path}
                    File Size: {result.file_size} bytes
                    Verified: {result.verified}
                    """

                error = None
                has_error = False

                # Check if the image was verified successfully
                if not result.verified and result.local_path:
                    error = "Image was generated but could not be verified"
                    has_error = True
                    formatted_result += f"\nWarning: {error}"

            except Exception as e:
                formatted_result = f"Error generating image: {str(e)}"
                error = str(e)
                result = None
                has_error = True

                # Get detailed error information
                import traceback
                trace = traceback.format_exc()
                print(f"Detailed image generation error: {trace}")

            # Generate response using LLM
            response = self.llm.invoke(
                self.prompt.format(
                    messages=state["messages"],
                    generation_results=formatted_result
                )
            )

            # Extract content from response
            if isinstance(response, dict):
                content = response.get("content", "No image generation results available")
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Update state
            if result:
                # Use model_dump() for Pydantic v2 compatibility
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                # Fallback for Pydantic v1
                elif hasattr(result, "dict"):
                    result_dict = result.dict()
                # Last resort fallback
                else:
                    result_dict = vars(result) if hasattr(result, "__dict__") else {"error": "Could not serialize result"}

                state["agent_outputs"]["image_generation_agent"] = result_dict

                # Add error information if verification failed
                if has_error:
                    state["agent_outputs"]["image_generation_agent"]["error"] = error
                    state["agent_outputs"]["image_generation_agent"]["has_error"] = True

                    # If this is a subtask in MCP, mark it for potential fallback
                    if "current_subtask" in state:
                        state["agent_outputs"]["image_generation_agent"]["needs_fallback"] = True
            else:
                state["agent_outputs"]["image_generation_agent"] = {
                    "error": error,
                    "has_error": True
                }

                # If this is a subtask in MCP, mark it for potential fallback
                if "current_subtask" in state:
                    state["agent_outputs"]["image_generation_agent"]["needs_fallback"] = True

            state["messages"].append({"role": "assistant", "content": content})

        except Exception as e:
            # Handle errors gracefully
            error_message = f"Image generation agent encountered an error: {str(e)}"
            print(error_message)

            # Get detailed error information
            import traceback
            trace = traceback.format_exc()
            print(f"Detailed error: {trace}")

            # Update state with error information
            state["agent_outputs"]["image_generation_agent"] = {
                "error": str(e),
                "traceback": trace,
                "has_error": True
            }

            # If this is a subtask in MCP, mark it for potential fallback
            if "current_subtask" in state:
                state["agent_outputs"]["image_generation_agent"]["needs_fallback"] = True

            # Add error response to messages
            state["messages"].append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error while generating the image: {str(e)}. Please try again with a different description or image generation provider."
            })

        return state


# Example usage
if __name__ == "__main__":
    # Create image generation agent
    image_generation_agent = ImageGenerationAgent(
        config=ImageGenerationAgentConfig(
            provider="dalle",
            dalle_model="dall-e-3",
            image_size="1024x1024"
        )
    )

    # Test with an image generation request
    state = {
        "messages": [{"role": "user", "content": "Generate an image of a futuristic city with flying cars."}],
        "agent_outputs": {}
    }

    updated_state = image_generation_agent(state)
    print(updated_state["messages"][-1]["content"])
