"""
Test Image Generation Agent with File Storage

This script tests the image generation agent's ability to generate images
and save them to local storage for verification and future use.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the image generation agent
from src.agents.image_generation_agent import ImageGenerationAgent, ImageGenerationAgentConfig
from src.utils.file_operations import verify_file_exists, get_file_size


def test_image_generation_with_storage():
    """
    Test the image generation agent with file storage.
    """
    print("Testing image generation agent with file storage...")
    
    # Create a test directory for images
    test_images_dir = "./test_generated_images"
    test_metadata_dir = "./test_generated_images/metadata"
    
    # Initialize the image generation agent
    image_generation_agent = ImageGenerationAgent(
        config=ImageGenerationAgentConfig(
            provider="dalle",
            dalle_model="dall-e-3",
            image_size="1024x1024",
            save_images=True,
            images_dir=test_images_dir,
            metadata_dir=test_metadata_dir
        )
    )
    
    # Test with an image generation request
    state = {
        "messages": [{"role": "user", "content": "Generate an image of a futuristic city with flying cars."}],
        "agent_outputs": {}
    }
    
    # Process the request
    updated_state = image_generation_agent(state)
    
    # Print the response
    print("\nAgent Response:")
    print(updated_state["messages"][-1]["content"])
    
    # Check if the image was generated and saved
    if "image_generation" in updated_state["agent_outputs"]:
        image_output = updated_state["agent_outputs"]["image_generation"]
        
        print("\nImage Generation Output:")
        print(f"Image URL: {image_output.get('image_url')}")
        print(f"Local Path: {image_output.get('local_path')}")
        print(f"File Size: {image_output.get('file_size')} bytes")
        print(f"Verified: {image_output.get('verified')}")
        
        # Verify the file exists
        local_path = image_output.get("local_path")
        if local_path and verify_file_exists(local_path):
            print(f"\nVerified: Image file exists at {local_path}")
            print(f"File size: {get_file_size(local_path)} bytes")
        else:
            print("\nWarning: Image file does not exist or could not be verified")
        
        # Check metadata
        metadata_path = image_output.get("metadata_path")
        if metadata_path and verify_file_exists(metadata_path):
            print(f"\nVerified: Metadata file exists at {metadata_path}")
            
            # Load and print metadata
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                print("\nMetadata:")
                print(json.dumps(metadata, indent=2))
            except Exception as e:
                print(f"\nError loading metadata: {str(e)}")
        else:
            print("\nWarning: Metadata file does not exist or could not be verified")
    else:
        print("\nError: Image generation output not found in agent outputs")
    
    return updated_state


def main():
    """Main function to run the test."""
    # Load environment variables
    load_dotenv()
    
    # Run the test
    test_image_generation_with_storage()


if __name__ == "__main__":
    main()
