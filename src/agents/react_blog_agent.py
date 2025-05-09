"""
ReAct Blog Writer Agent using LangGraph's create_react_agent

This module implements a blog writer agent using LangGraph's create_react_agent function
and specialized blog writing tools.
"""

import os
import json
from typing import Dict, List, Any, Optional, Literal

# Import base document agent
from src.agents.react_document_agent import ReactDocumentAgent, ReactDocumentAgentConfig

# LangChain components
from langchain_core.tools import tool, Tool


class ReactBlogAgentConfig(ReactDocumentAgentConfig):
    """Configuration for the ReAct blog writer agent."""
    # Document type
    document_type: str = "blog"

    # Blog-specific configuration
    default_sections: List[str] = ["Introduction", "Main Points", "Conclusion"]
    tone: Literal["casual", "professional", "technical", "humorous"] = "casual"
    target_audience: str = "general"
    include_images: bool = True
    include_seo_keywords: bool = True

    # File storage configuration
    documents_dir: str = "./generated_documents/blogs"
    metadata_dir: str = "./generated_documents/blogs/metadata"

    # System message
    system_message: str = """
    You are a blog writer agent that creates engaging, well-structured blog posts based on user requirements.

    Your job is to:
    1. Analyze the user's blog post request carefully
    2. Extract key requirements like title, tone, target audience, and content needs
    3. Generate an engaging blog post that meets these requirements
    4. Save the blog post and provide a summary of what was created

    Blog posts should include:
    - An attention-grabbing introduction
    - Well-structured content with clear headings
    - A tone appropriate for the target audience
    - Engaging language and formatting
    - A strong conclusion with call-to-action when appropriate

    Use the available tools to:
    - Extract blog post requirements from the user's request
    - Generate the blog post content
    - Save the blog post to a file

    Always ensure the blog post is engaging, well-organized, and meets the user's requirements.
    """


class ReactBlogAgent(ReactDocumentAgent):
    """
    Blog writer agent using LangGraph's create_react_agent function.
    Specializes in creating engaging blog posts with different tones and styles.
    """

    def __init__(self, config=None, llm=None):
        """
        Initialize the blog writer agent.

        Args:
            config: Configuration for the blog writer agent
            llm: Optional language model to use
        """
        # Initialize with blog-specific configuration
        super().__init__(config or ReactBlogAgentConfig(), llm)

    def _initialize_tools(self) -> List[Any]:
        """
        Initialize blog writing tools.

        Returns:
            List[Any]: List of initialized tools
        """
        # Get base tools from parent class
        base_tools = super()._initialize_tools()

        # Create a reference to the agent instance for use in the tools
        agent_instance = self

        # Define tool functions without decorators
        def extract_blog_requirements_func(message: str) -> str:
            """
            Extract blog-specific requirements from the user's message.

            Args:
                message: The user's message containing blog post requirements

            Returns:
                str: JSON string of extracted requirements
            """
            # Default requirements
            requirements = {
                "title": "Untitled Blog Post",
                "sections": agent_instance.config.default_sections,
                "style": "blog",
                "include_toc": False,
                "include_references": False,
                "include_appendices": False,
                "document_type": "blog",
                "tone": agent_instance.config.tone,
                "target_audience": agent_instance.config.target_audience,
                "include_images": agent_instance.config.include_images,
                "include_seo_keywords": agent_instance.config.include_seo_keywords
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

            # Extract tone if present
            if "tone:" in message.lower():
                tone_text = message.lower().split("tone:")[1].split("\n")[0].strip()
                if "casual" in tone_text:
                    requirements["tone"] = "casual"
                elif "professional" in tone_text:
                    requirements["tone"] = "professional"
                elif "technical" in tone_text:
                    requirements["tone"] = "technical"
                elif "humorous" in tone_text:
                    requirements["tone"] = "humorous"

            # Extract target audience if present
            if "audience:" in message.lower() or "target audience:" in message.lower():
                audience_text = message.lower().split("audience:")[1].split("\n")[0].strip() if "audience:" in message.lower() else message.lower().split("target audience:")[1].split("\n")[0].strip()
                if audience_text:
                    requirements["target_audience"] = audience_text

            # Check for images
            requirements["include_images"] = "image" in message.lower() or "images" in message.lower() or "picture" in message.lower() or "pictures" in message.lower()

            # Check for SEO keywords
            requirements["include_seo_keywords"] = "seo" in message.lower() or "keyword" in message.lower() or "keywords" in message.lower()

            return json.dumps(requirements, indent=2)

        def generate_blog_content_func(requirements_json: str) -> str:
            """
            Generate blog post content based on requirements.

            Args:
                requirements_json: JSON string of blog post requirements

            Returns:
                str: Generated blog post content
            """
            try:
                requirements = json.loads(requirements_json)

                # Create a prompt for blog post generation
                prompt = f"""
                Generate an engaging blog post with the following specifications:

                Title: {requirements.get('title', 'Untitled Blog Post')}
                Sections: {', '.join(requirements.get('sections', agent_instance.config.default_sections))}
                Tone: {requirements.get('tone', agent_instance.config.tone)}
                Target Audience: {requirements.get('target_audience', agent_instance.config.target_audience)}
                Include Images: {requirements.get('include_images', True)}
                Include SEO Keywords: {requirements.get('include_seo_keywords', True)}

                The blog post should be well-structured, engaging, and formatted in Markdown.
                Use language appropriate for the specified tone and target audience.
                If including images, describe them in text (e.g., [Image: Title - Description]).
                If including SEO keywords, suggest 3-5 relevant keywords at the end of the post.
                """

                # Generate the blog post using the LLM
                response = agent_instance.llm.invoke([
                    {"role": "system", "content": f"You are a professional blog writer specializing in creating engaging content with a {requirements.get('tone', agent_instance.config.tone)} tone."},
                    {"role": "user", "content": prompt}
                ])

                # Extract content from response
                if hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)

                # Calculate reading time (average reading speed: 200-250 words per minute)
                word_count = len(content.split())
                reading_time_minutes = max(1, round(word_count / 225))

                # Add reading time to the blog post
                content = f"*Reading time: {reading_time_minutes} minute{'s' if reading_time_minutes > 1 else ''}*\n\n{content}"

                return content

            except Exception as e:
                return f"Error generating blog post content: {str(e)}"

        # Create Tool objects
        extract_blog_requirements_tool = Tool(
            name="extract_blog_requirements",
            description="Extract blog-specific requirements from the user's message",
            func=extract_blog_requirements_func
        )

        generate_blog_content_tool = Tool(
            name="generate_blog_content",
            description="Generate blog post content based on requirements",
            func=generate_blog_content_func
        )

        # Replace the base document tools with blog-specific tools
        blog_tools = []

        # Add the blog-specific tools
        blog_tools.append(extract_blog_requirements_tool)
        blog_tools.append(generate_blog_content_tool)

        # Add the save_document tool from base tools
        for tool in base_tools:
            if tool.name == "save_document":
                blog_tools.append(tool)
                break

        return blog_tools
