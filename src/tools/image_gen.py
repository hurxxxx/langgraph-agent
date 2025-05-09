"""
Image Generation Tools

This module implements image generation tools for the LangGraph Agent system.
These tools provide the ability to generate images from text descriptions.
"""

import os
import json
import uuid
import hashlib
import requests
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from src.tools.base import BaseToolConfig, BaseTool, StructuredTool


class DALLEImageGenerationToolConfig(BaseToolConfig):
    """Configuration for the DALL-E image generation tool."""
    
    name: str = "dalle_image_generation"
    description: str = "Generate images from text descriptions using DALL-E."
    openai_api_key: Optional[str] = None
    model: str = "dall-e-3"
    size: str = "1024x1024"
    quality: Literal["standard", "high"] = "standard"
    style: Optional[str] = None
    save_images: bool = True
    images_dir: str = "./generated_images"
    metadata_dir: str = "./generated_images/metadata"


class DALLEImageGenerationTool(StructuredTool):
    """
    Image generation tool using OpenAI's DALL-E.
    
    This tool generates images from text descriptions using DALL-E.
    """
    
    def __init__(self, config: Optional[DALLEImageGenerationToolConfig] = None):
        """
        Initialize the DALL-E image generation tool.
        
        Args:
            config: Configuration for the DALL-E image generation tool
        """
        super().__init__(config)
        
        # Create directories if they don't exist
        if self.config.save_images:
            os.makedirs(self.config.images_dir, exist_ok=True)
            os.makedirs(self.config.metadata_dir, exist_ok=True)
    
    def _get_default_config(self) -> DALLEImageGenerationToolConfig:
        """
        Get the default configuration for the DALL-E image generation tool.
        
        Returns:
            DALLEImageGenerationToolConfig: Default configuration
        """
        return DALLEImageGenerationToolConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def run_structured(self, prompt: str) -> str:
        """
        Generate an image from a text description using DALL-E.
        
        Args:
            prompt: Text description of the image to generate
            
        Returns:
            str: URL of the generated image
        """
        try:
            # Get the API key
            api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required")
            
            # Import OpenAI client
            from openai import OpenAI
            
            # Create the client
            client = OpenAI(api_key=api_key)
            
            # Generate the image
            response = client.images.generate(
                model=self.config.model,
                prompt=prompt,
                size=self.config.size,
                quality=self.config.quality,
                style=self.config.style,
                n=1
            )
            
            # Get the image URL
            image_url = response.data[0].url
            
            # Save the image if configured
            if self.config.save_images:
                file_path = self._save_image(image_url, prompt)
                return f"Image generated and saved to {file_path}\nURL: {image_url}"
            else:
                return f"Image generated: {image_url}"
        except Exception as e:
            return f"Error generating image with DALL-E: {str(e)}"
    
    def _save_image(self, image_url: str, prompt: str) -> str:
        """
        Save an image from a URL.
        
        Args:
            image_url: URL of the image to save
            prompt: Text description of the image
            
        Returns:
            str: Path to the saved image
        """
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            filename = f"dalle_{timestamp}_{prompt_hash}.png"
            file_path = os.path.join(self.config.images_dir, filename)
            
            # Download the image
            response = requests.get(image_url)
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            # Save metadata
            self._save_image_metadata(prompt, image_url, file_path)
            
            return file_path
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return "Error saving image"
    
    def _save_image_metadata(self, prompt: str, image_url: str, file_path: str) -> str:
        """
        Save metadata for the generated image.
        
        Args:
            prompt: The prompt used to generate the image
            image_url: URL of the generated image
            file_path: Path where the image is saved
            
        Returns:
            str: Path to the metadata file
        """
        # Create a unique ID for the image
        image_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = {
            "id": image_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model": self.config.model,
            "size": self.config.size,
            "quality": self.config.quality,
            "style": self.config.style,
            "image_url": image_url,
            "file_path": file_path
        }
        
        # Save metadata to file
        metadata_path = os.path.join(self.config.metadata_dir, f"{image_id}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path


class GPTImageGenerationToolConfig(BaseToolConfig):
    """Configuration for the GPT-Image generation tool."""
    
    name: str = "gpt_image_generation"
    description: str = "Generate images from text descriptions using GPT-Image."
    openai_api_key: Optional[str] = None
    model: str = "gpt-image-1"
    size: str = "1024x1024"
    quality: Literal["standard", "high"] = "standard"
    style: Optional[str] = None
    save_images: bool = True
    images_dir: str = "./generated_images"
    metadata_dir: str = "./generated_images/metadata"


class GPTImageGenerationTool(StructuredTool):
    """
    Image generation tool using OpenAI's GPT-Image.
    
    This tool generates images from text descriptions using GPT-Image.
    """
    
    def __init__(self, config: Optional[GPTImageGenerationToolConfig] = None):
        """
        Initialize the GPT-Image generation tool.
        
        Args:
            config: Configuration for the GPT-Image generation tool
        """
        super().__init__(config)
        
        # Create directories if they don't exist
        if self.config.save_images:
            os.makedirs(self.config.images_dir, exist_ok=True)
            os.makedirs(self.config.metadata_dir, exist_ok=True)
    
    def _get_default_config(self) -> GPTImageGenerationToolConfig:
        """
        Get the default configuration for the GPT-Image generation tool.
        
        Returns:
            GPTImageGenerationToolConfig: Default configuration
        """
        return GPTImageGenerationToolConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def run_structured(self, prompt: str) -> str:
        """
        Generate an image from a text description using GPT-Image.
        
        Args:
            prompt: Text description of the image to generate
            
        Returns:
            str: URL of the generated image
        """
        try:
            # Get the API key
            api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required")
            
            # Import OpenAI client
            from openai import OpenAI
            
            # Create the client
            client = OpenAI(api_key=api_key)
            
            # Generate the image
            response = client.images.generate(
                model=self.config.model,
                prompt=prompt,
                size=self.config.size,
                quality=self.config.quality,
                style=self.config.style,
                n=1
            )
            
            # Get the image URL
            image_url = response.data[0].url
            
            # Save the image if configured
            if self.config.save_images:
                file_path = self._save_image(image_url, prompt)
                return f"Image generated and saved to {file_path}\nURL: {image_url}"
            else:
                return f"Image generated: {image_url}"
        except Exception as e:
            return f"Error generating image with GPT-Image: {str(e)}"
    
    def _save_image(self, image_url: str, prompt: str) -> str:
        """
        Save an image from a URL.
        
        Args:
            image_url: URL of the image to save
            prompt: Text description of the image
            
        Returns:
            str: Path to the saved image
        """
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            filename = f"gpt_image_{timestamp}_{prompt_hash}.png"
            file_path = os.path.join(self.config.images_dir, filename)
            
            # Download the image
            response = requests.get(image_url)
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            # Save metadata
            self._save_image_metadata(prompt, image_url, file_path)
            
            return file_path
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return "Error saving image"
    
    def _save_image_metadata(self, prompt: str, image_url: str, file_path: str) -> str:
        """
        Save metadata for the generated image.
        
        Args:
            prompt: The prompt used to generate the image
            image_url: URL of the generated image
            file_path: Path where the image is saved
            
        Returns:
            str: Path to the metadata file
        """
        # Create a unique ID for the image
        image_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = {
            "id": image_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model": self.config.model,
            "size": self.config.size,
            "quality": self.config.quality,
            "style": self.config.style,
            "image_url": image_url,
            "file_path": file_path
        }
        
        # Save metadata to file
        metadata_path = os.path.join(self.config.metadata_dir, f"{image_id}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path


# Example usage
if __name__ == "__main__":
    # Create DALL-E image generation tool
    dalle_tool = DALLEImageGenerationTool(
        config=DALLEImageGenerationToolConfig(
            model="dall-e-3",
            size="1024x1024",
            quality="standard",
            save_images=True
        )
    )
    
    # Generate an image with DALL-E
    dalle_result = dalle_tool.run("A beautiful sunset over the ocean")
    print(dalle_result)
    print("\n" + "-" * 50 + "\n")
    
    # Create GPT-Image generation tool
    gpt_image_tool = GPTImageGenerationTool(
        config=GPTImageGenerationToolConfig(
            model="gpt-image-1",
            size="1024x1024",
            quality="high",
            save_images=True
        )
    )
    
    # Generate an image with GPT-Image
    gpt_image_result = gpt_image_tool.run("A Ghibli-style image of a peaceful forest")
    print(gpt_image_result)
