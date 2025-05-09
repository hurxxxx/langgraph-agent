"""
Blog Writer Agent for Multi-Agent System

This module implements a specialized document generation agent for blog posts and articles.
It extends the base document agent with blog-specific features.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import base document agent
from src.agents.document_generation.base_document_agent import (
    BaseDocumentAgent,
    BaseDocumentAgentConfig,
    DocumentFormat,
    DocumentGenerationResult
)


class BlogFormat(DocumentFormat):
    """Model for blog format specifications."""
    target_audience: str = "general"
    tone: Literal["casual", "professional", "humorous", "educational"] = "casual"
    include_images: bool = False
    include_links: bool = True
    seo_keywords: List[str] = []
    reading_time_minutes: Optional[int] = None


class BlogGenerationResult(DocumentGenerationResult):
    """Model for a blog generation result."""
    format: BlogFormat


class BlogWriterAgentConfig(BaseDocumentAgentConfig):
    """Configuration for the blog writer agent."""
    documents_dir: str = "./generated_documents/blogs"
    metadata_dir: str = "./generated_documents/blogs/metadata"
    system_message: str = """
    You are a blog writer agent that creates engaging, well-structured blog posts and articles based on user requirements.

    Your job is to:
    1. Understand the blog post requirements
    2. Generate an engaging blog post with appropriate structure and content
    3. Format the blog post according to the specified style and tone
    4. Include SEO keywords, links, and other blog-specific elements
    5. Provide the blog post content and any relevant metadata

    Always ensure that the blog post is engaging, readable, and tailored to the target audience.
    Use an appropriate tone, include hooks to capture reader interest, and organize content for easy scanning.
    """


class BlogWriterAgent(BaseDocumentAgent):
    """
    Blog writer agent that creates blog posts and articles.
    """

    def __init__(self, config: BlogWriterAgentConfig = BlogWriterAgentConfig()):
        """
        Initialize the blog writer agent.

        Args:
            config: Configuration for the blog writer agent
        """
        super().__init__(config)
        self.config = config

    def _extract_document_requirements(self, message: str) -> Dict[str, Any]:
        """
        Extract blog requirements from the message.

        Args:
            message: User message

        Returns:
            Dict[str, Any]: Extracted blog requirements
        """
        # Get base requirements
        requirements = super()._extract_document_requirements(message)

        # Default blog sections if none specified
        if requirements["sections"] == ["Introduction", "Main Content", "Conclusion"]:
            requirements["sections"] = [
                "Introduction",
                "Main Points",
                "Conclusion",
                "Call to Action"
            ]

        # Blog-specific requirements
        requirements["include_images"] = "images" in message.lower() or "pictures" in message.lower()
        requirements["include_links"] = not ("no links" in message.lower())

        # Extract target audience
        if "audience:" in message.lower():
            audience_line = [line for line in message.split("\n") if "audience:" in line.lower()]
            if audience_line:
                requirements["target_audience"] = audience_line[0].split("audience:")[1].strip()
        elif "for beginners" in message.lower():
            requirements["target_audience"] = "beginners"
        elif "for experts" in message.lower() or "for professionals" in message.lower():
            requirements["target_audience"] = "experts"
        else:
            requirements["target_audience"] = "general"

        # Extract tone
        if "tone:" in message.lower():
            tone_line = [line for line in message.split("\n") if "tone:" in line.lower()]
            if tone_line:
                tone = tone_line[0].split("tone:")[1].strip().lower()
                if "casual" in tone or "informal" in tone:
                    requirements["tone"] = "casual"
                elif "professional" in tone or "formal" in tone:
                    requirements["tone"] = "professional"
                elif "humor" in tone or "funny" in tone:
                    requirements["tone"] = "humorous"
                elif "educational" in tone or "informative" in tone:
                    requirements["tone"] = "educational"
        elif "casual" in message.lower() or "informal" in message.lower():
            requirements["tone"] = "casual"
        elif "professional" in message.lower() or "formal" in message.lower():
            requirements["tone"] = "professional"
        elif "humor" in message.lower() or "funny" in message.lower():
            requirements["tone"] = "humorous"
        elif "educational" in message.lower() or "informative" in message.lower():
            requirements["tone"] = "educational"

        # Extract SEO keywords
        if "keywords:" in message.lower():
            keywords_line = [line for line in message.split("\n") if "keywords:" in line.lower()]
            if keywords_line:
                keywords = keywords_line[0].split("keywords:")[1].strip()
                requirements["seo_keywords"] = [k.strip() for k in keywords.split(",")]
        elif "seo:" in message.lower():
            seo_line = [line for line in message.split("\n") if "seo:" in line.lower()]
            if seo_line:
                keywords = seo_line[0].split("seo:")[1].strip()
                requirements["seo_keywords"] = [k.strip() for k in keywords.split(",")]

        # Estimate reading time (rough estimate based on 200 words per minute)
        if "word count:" in message.lower():
            word_count_line = [line for line in message.split("\n") if "word count:" in line.lower()]
            if word_count_line:
                try:
                    word_count = int(word_count_line[0].split("word count:")[1].strip().split()[0])
                    requirements["reading_time_minutes"] = max(1, round(word_count / 200))
                except:
                    pass

        return requirements

    def generate_document(self, message: str) -> BlogGenerationResult:
        """
        Generate a blog post based on the message.

        Args:
            message: User message

        Returns:
            BlogGenerationResult: Generated blog post
        """
        # Extract blog requirements
        requirements = self._extract_document_requirements(message)

        # Create blog format
        blog_format = BlogFormat(**requirements)

        # Generate blog content
        response = self.llm.invoke(
            self.prompt.format(
                messages=[{"role": "user", "content": message}],
                document_format=blog_format.model_dump_json()
            )
        )

        # Extract content
        if isinstance(response, dict):
            content = response.get("content", "")
        elif hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        # Calculate reading time if not specified
        word_count = len(content.split())
        if not blog_format.reading_time_minutes:
            blog_format.reading_time_minutes = max(1, round(word_count / 200))

        # Save blog post if configured
        save_result = self._save_document(content, blog_format)

        # Create result
        result = BlogGenerationResult(
            content=content,
            format=blog_format,
            word_count=word_count,
            file_path=save_result.get("file_path"),
            metadata_path=save_result.get("metadata_path"),
            created_at=time.time()
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

        # Generate blog post
        try:
            result = self.generate_document(message)

            # Format the result for the LLM
            formatted_result = f"""
            Blog post generated successfully!
            Title: {result.format.title}
            Target Audience: {result.format.target_audience}
            Tone: {result.format.tone}
            Word Count: {result.word_count}
            Reading Time: {result.format.reading_time_minutes} minutes
            """

            if result.format.seo_keywords:
                formatted_result += f"\nSEO Keywords: {', '.join(result.format.seo_keywords)}"

            if result.file_path:
                formatted_result += f"\nSaved to: {result.file_path}"

            error = None
        except Exception as e:
            formatted_result = f"Error generating blog post: {str(e)}"
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
            state["agent_outputs"]["blog_generation"] = result.model_dump()
        else:
            state["agent_outputs"]["blog_generation"] = {"error": error}

        state["messages"].append({"role": "assistant", "content": response.content})

        return state


# Example usage
if __name__ == "__main__":
    # Create blog writer agent
    blog_agent = BlogWriterAgent()

    # Test with a blog generation request
    state = {
        "messages": [{"role": "user", "content": "Write a blog post about machine learning for beginners with a casual tone. Keywords: AI, machine learning, beginners guide"}],
        "agent_outputs": {}
    }

    updated_state = blog_agent(state)
    print(updated_state["messages"][-1]["content"])
