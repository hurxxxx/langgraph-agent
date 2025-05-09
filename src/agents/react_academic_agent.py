"""
ReAct Academic Writer Agent using LangGraph's create_react_agent

This module implements an academic writer agent using LangGraph's create_react_agent function
and specialized academic writing tools.
"""

import os
import json
from typing import Dict, List, Any, Optional, Literal

# Import base document agent
from src.agents.react_document_agent import ReactDocumentAgent, ReactDocumentAgentConfig

# LangChain components
from langchain_core.tools import tool, Tool


class ReactAcademicAgentConfig(ReactDocumentAgentConfig):
    """Configuration for the ReAct academic writer agent."""
    # Document type
    document_type: str = "academic"

    # Academic-specific configuration
    default_sections: List[str] = ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Discussion", "Conclusion", "References"]
    citation_style: Literal["APA", "MLA", "Chicago", "Harvard", "IEEE"] = "APA"
    include_abstract: bool = True
    include_literature_review: bool = True
    include_methodology: bool = True
    include_results: bool = True
    include_discussion: bool = True
    include_references: bool = True

    # File storage configuration
    documents_dir: str = "./generated_documents/academic"
    metadata_dir: str = "./generated_documents/academic/metadata"

    # System message
    system_message: str = """
    You are an academic writer agent that creates scholarly, well-structured academic papers based on user requirements.

    Your job is to:
    1. Analyze the user's academic paper request carefully
    2. Extract key requirements like title, sections, citation style, and content needs
    3. Generate a scholarly paper that meets these requirements
    4. Save the paper and provide a summary of what was created

    Academic papers should include:
    - A clear abstract summarizing the paper
    - Proper introduction and literature review
    - Well-defined methodology section
    - Clear presentation of results and discussion
    - Proper citations and references according to the specified style

    Use the available tools to:
    - Extract academic paper requirements from the user's request
    - Generate the academic paper content
    - Save the academic paper to a file

    Always ensure the academic paper is scholarly, well-organized, and meets the user's requirements.
    """


class ReactAcademicAgent(ReactDocumentAgent):
    """
    Academic writer agent using LangGraph's create_react_agent function.
    Specializes in creating scholarly papers with proper citations and academic structure.
    """

    def __init__(self, config=None, llm=None):
        """
        Initialize the academic writer agent.

        Args:
            config: Configuration for the academic writer agent
            llm: Optional language model to use
        """
        # Initialize with academic-specific configuration
        super().__init__(config or ReactAcademicAgentConfig(), llm)

    def _initialize_tools(self) -> List[Any]:
        """
        Initialize academic writing tools.

        Returns:
            List[Any]: List of initialized tools
        """
        # Get base tools from parent class
        base_tools = super()._initialize_tools()

        # Create a reference to the agent instance for use in the tools
        agent_instance = self

        # Define tool functions without decorators
        def extract_academic_requirements_func(message: str) -> str:
            """
            Extract academic-specific requirements from the user's message.

            Args:
                message: The user's message containing academic paper requirements

            Returns:
                str: JSON string of extracted requirements
            """
            # Default requirements
            requirements = {
                "title": "Untitled Academic Paper",
                "sections": agent_instance.config.default_sections,
                "style": "academic",
                "include_toc": True,
                "include_references": True,
                "include_appendices": False,
                "document_type": "academic",
                "citation_style": agent_instance.config.citation_style,
                "include_abstract": agent_instance.config.include_abstract,
                "include_literature_review": agent_instance.config.include_literature_review,
                "include_methodology": agent_instance.config.include_methodology,
                "include_results": agent_instance.config.include_results,
                "include_discussion": agent_instance.config.include_discussion
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

            # Extract citation style if present
            if "citation:" in message.lower() or "citation style:" in message.lower() or "citations:" in message.lower():
                citation_text = message.lower().split("citation:")[1].split("\n")[0].strip() if "citation:" in message.lower() else message.lower().split("citation style:")[1].split("\n")[0].strip() if "citation style:" in message.lower() else message.lower().split("citations:")[1].split("\n")[0].strip()
                if "apa" in citation_text:
                    requirements["citation_style"] = "APA"
                elif "mla" in citation_text:
                    requirements["citation_style"] = "MLA"
                elif "chicago" in citation_text:
                    requirements["citation_style"] = "Chicago"
                elif "harvard" in citation_text:
                    requirements["citation_style"] = "Harvard"
                elif "ieee" in citation_text:
                    requirements["citation_style"] = "IEEE"

            # Check for abstract
            requirements["include_abstract"] = "abstract" in message.lower()

            # Check for literature review
            requirements["include_literature_review"] = "literature review" in message.lower() or "literature" in message.lower()

            # Check for methodology
            requirements["include_methodology"] = "methodology" in message.lower() or "method" in message.lower() or "methods" in message.lower()

            # Check for results
            requirements["include_results"] = "results" in message.lower() or "result" in message.lower() or "findings" in message.lower()

            # Check for discussion
            requirements["include_discussion"] = "discussion" in message.lower() or "discuss" in message.lower()

            return json.dumps(requirements, indent=2)

        def generate_academic_content_func(requirements_json: str) -> str:
            """
            Generate academic paper content based on requirements.

            Args:
                requirements_json: JSON string of academic paper requirements

            Returns:
                str: Generated academic paper content
            """
            try:
                requirements = json.loads(requirements_json)

                # Create a prompt for academic paper generation
                prompt = f"""
                Generate a scholarly academic paper with the following specifications:

                Title: {requirements.get('title', 'Untitled Academic Paper')}
                Sections: {', '.join(requirements.get('sections', agent_instance.config.default_sections))}
                Citation Style: {requirements.get('citation_style', agent_instance.config.citation_style)}
                Include Abstract: {requirements.get('include_abstract', True)}
                Include Literature Review: {requirements.get('include_literature_review', True)}
                Include Methodology: {requirements.get('include_methodology', True)}
                Include Results: {requirements.get('include_results', True)}
                Include Discussion: {requirements.get('include_discussion', True)}
                Include References: {requirements.get('include_references', True)}

                The paper should be well-structured, scholarly, and formatted in Markdown.
                Use formal academic language appropriate for scholarly publication.
                Include proper citations according to the specified citation style.
                Generate realistic but fictional references for demonstration purposes.
                """

                # Generate the academic paper using the LLM
                response = agent_instance.llm.invoke([
                    {"role": "system", "content": "You are a professional academic writer specializing in creating scholarly papers with proper academic structure and citations."},
                    {"role": "user", "content": prompt}
                ])

                # Extract content from response
                if hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)

                return content

            except Exception as e:
                return f"Error generating academic paper content: {str(e)}"

        # Create Tool objects
        extract_academic_requirements_tool = Tool(
            name="extract_academic_requirements",
            description="Extract academic-specific requirements from the user's message",
            func=extract_academic_requirements_func
        )

        generate_academic_content_tool = Tool(
            name="generate_academic_content",
            description="Generate academic paper content based on requirements",
            func=generate_academic_content_func
        )

        # Replace the base document tools with academic-specific tools
        academic_tools = []

        # Add the academic-specific tools
        academic_tools.append(extract_academic_requirements_tool)
        academic_tools.append(generate_academic_content_tool)

        # Add the save_document tool from base tools
        for tool in base_tools:
            if tool.name == "save_document":
                academic_tools.append(tool)
                break

        return academic_tools
