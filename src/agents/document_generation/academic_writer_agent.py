"""
Academic Writer Agent for Multi-Agent System

This module implements a specialized document generation agent for academic papers.
It extends the base document agent with academic-specific features.
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


CitationStyle = Literal["APA", "MLA", "Chicago", "Harvard", "IEEE"]
"""Citation style for academic papers."""


class AcademicFormat(DocumentFormat):
    """Model for academic paper format specifications."""
    abstract: bool = True
    citation_style: CitationStyle = "APA"
    include_figures: bool = False
    include_tables: bool = False
    include_equations: bool = False
    field_of_study: str = "General"
    keywords: List[str] = []


class AcademicGenerationResult(DocumentGenerationResult):
    """Model for an academic paper generation result."""
    format: AcademicFormat


class AcademicWriterAgentConfig(BaseDocumentAgentConfig):
    """Configuration for the academic writer agent."""
    documents_dir: str = "./generated_documents/academic"
    metadata_dir: str = "./generated_documents/academic/metadata"
    system_message: str = """
    You are an academic writer agent that creates scholarly, well-structured academic papers based on user requirements.

    Your job is to:
    1. Understand the academic paper requirements
    2. Generate a scholarly paper with appropriate structure and content
    3. Format the paper according to the specified citation style
    4. Include abstract, references, and other academic-specific elements
    5. Provide the paper content and any relevant metadata

    Always ensure that the paper is well-organized, coherent, and meets academic standards.
    Use formal academic language, cite sources properly, and maintain a scholarly tone throughout.
    """


class AcademicWriterAgent(BaseDocumentAgent):
    """
    Academic writer agent that creates scholarly papers.
    """

    def __init__(self, config: AcademicWriterAgentConfig = AcademicWriterAgentConfig()):
        """
        Initialize the academic writer agent.

        Args:
            config: Configuration for the academic writer agent
        """
        super().__init__(config)
        self.config = config

    def _extract_document_requirements(self, message: str) -> Dict[str, Any]:
        """
        Extract academic paper requirements from the message.

        Args:
            message: User message

        Returns:
            Dict[str, Any]: Extracted academic paper requirements
        """
        # Get base requirements
        requirements = super()._extract_document_requirements(message)

        # Default academic paper sections if none specified
        if requirements["sections"] == ["Introduction", "Main Content", "Conclusion"]:
            requirements["sections"] = [
                "Abstract",
                "Introduction",
                "Literature Review",
                "Methodology",
                "Results",
                "Discussion",
                "Conclusion",
                "References"
            ]

        # Academic-specific requirements
        requirements["abstract"] = not ("no abstract" in message.lower())
        requirements["include_figures"] = "figures" in message.lower() or "diagrams" in message.lower()
        requirements["include_tables"] = "tables" in message.lower() or "data tables" in message.lower()
        requirements["include_equations"] = "equations" in message.lower() or "formulas" in message.lower()

        # Extract citation style
        if "citation style:" in message.lower() or "citation:" in message.lower():
            citation_line = [line for line in message.split("\n")
                            if "citation style:" in line.lower() or "citation:" in line.lower()]
            if citation_line:
                citation_text = citation_line[0].split(":")[1].strip().upper()
                if "APA" in citation_text:
                    requirements["citation_style"] = "APA"
                elif "MLA" in citation_text:
                    requirements["citation_style"] = "MLA"
                elif "CHICAGO" in citation_text:
                    requirements["citation_style"] = "Chicago"
                elif "HARVARD" in citation_text:
                    requirements["citation_style"] = "Harvard"
                elif "IEEE" in citation_text:
                    requirements["citation_style"] = "IEEE"
        elif "APA" in message.upper():
            requirements["citation_style"] = "APA"
        elif "MLA" in message.upper():
            requirements["citation_style"] = "MLA"
        elif "CHICAGO" in message.upper():
            requirements["citation_style"] = "Chicago"
        elif "HARVARD" in message.upper():
            requirements["citation_style"] = "Harvard"
        elif "IEEE" in message.upper():
            requirements["citation_style"] = "IEEE"

        # Extract field of study
        if "field:" in message.lower():
            field_line = [line for line in message.split("\n") if "field:" in line.lower()]
            if field_line:
                requirements["field_of_study"] = field_line[0].split("field:")[1].strip()

        # Extract keywords
        if "keywords:" in message.lower():
            keywords_line = [line for line in message.split("\n") if "keywords:" in line.lower()]
            if keywords_line:
                keywords = keywords_line[0].split("keywords:")[1].strip()
                requirements["keywords"] = [k.strip() for k in keywords.split(",")]

        return requirements

    def generate_document(self, message: str) -> AcademicGenerationResult:
        """
        Generate an academic paper based on the message.

        Args:
            message: User message

        Returns:
            AcademicGenerationResult: Generated academic paper
        """
        # Extract academic paper requirements
        requirements = self._extract_document_requirements(message)

        # Create academic format
        academic_format = AcademicFormat(**requirements)

        # Generate academic paper content
        response = self.llm.invoke(
            self.prompt.format(
                messages=[{"role": "user", "content": message}],
                document_format=academic_format.model_dump_json()
            )
        )

        # Extract content
        if isinstance(response, dict):
            content = response.get("content", "")
        elif hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        # Save academic paper if configured
        save_result = self._save_document(content, academic_format)

        # Create result
        result = AcademicGenerationResult(
            content=content,
            format=academic_format,
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

        # Generate academic paper
        try:
            result = self.generate_document(message)

            # Format the result for the LLM
            formatted_result = f"""
            Academic paper generated successfully!
            Title: {result.format.title}
            Field of Study: {result.format.field_of_study}
            Citation Style: {result.format.citation_style}
            Word Count: {result.word_count}
            """

            if result.format.keywords:
                formatted_result += f"\nKeywords: {', '.join(result.format.keywords)}"

            if result.file_path:
                formatted_result += f"\nSaved to: {result.file_path}"

            error = None
        except Exception as e:
            formatted_result = f"Error generating academic paper: {str(e)}"
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
            state["agent_outputs"]["academic_generation"] = result.model_dump()
        else:
            state["agent_outputs"]["academic_generation"] = {"error": error}

        state["messages"].append({"role": "assistant", "content": response.content})

        return state


# Example usage
if __name__ == "__main__":
    # Create academic writer agent
    academic_agent = AcademicWriterAgent()

    # Test with an academic paper generation request
    state = {
        "messages": [{"role": "user", "content": "Write an academic paper about quantum computing with APA citation style. Include figures and equations."}],
        "agent_outputs": {}
    }

    updated_state = academic_agent(state)
    print(updated_state["messages"][-1]["content"])
