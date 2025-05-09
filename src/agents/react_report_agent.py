"""
ReAct Report Writer Agent using LangGraph's create_react_agent

This module implements a report writer agent using LangGraph's create_react_agent function
and specialized report generation tools.
"""

import os
import json
from typing import Dict, List, Any, Optional, Literal

# Import base document agent
from src.agents.react_document_agent import ReactDocumentAgent, ReactDocumentAgentConfig

# LangChain components
from langchain_core.tools import tool, Tool


class ReactReportAgentConfig(ReactDocumentAgentConfig):
    """Configuration for the ReAct report writer agent."""
    # Document type
    document_type: str = "report"

    # Report-specific configuration
    default_sections: List[str] = ["Executive Summary", "Introduction", "Findings", "Analysis", "Recommendations", "Conclusion"]
    formality_level: Literal["high", "medium", "low"] = "high"
    include_executive_summary: bool = True
    include_recommendations: bool = True
    include_charts: bool = False

    # File storage configuration
    documents_dir: str = "./generated_documents/reports"
    metadata_dir: str = "./generated_documents/reports/metadata"

    # System message
    system_message: str = """
    You are a report writer agent that creates professional, well-structured reports based on user requirements.

    Your job is to:
    1. Analyze the user's report request carefully
    2. Extract key requirements like title, sections, and content needs
    3. Generate a well-structured report that meets these requirements
    4. Save the report and provide a summary of what was created

    Reports should include:
    - An executive summary that concisely presents the key findings
    - Clear section headings and organization
    - Professional language and formatting
    - Data-driven analysis where appropriate
    - Actionable recommendations

    Use the available tools to:
    - Extract report requirements from the user's request
    - Generate the report content
    - Save the report to a file

    Always ensure the report is well-organized, professional, and meets the user's requirements.
    """


class ReactReportAgent(ReactDocumentAgent):
    """
    Report writer agent using LangGraph's create_react_agent function.
    Specializes in creating formal reports with executive summaries and recommendations.
    """

    def __init__(self, config=None, llm=None):
        """
        Initialize the report writer agent.

        Args:
            config: Configuration for the report writer agent
            llm: Optional language model to use
        """
        # Initialize with report-specific configuration
        super().__init__(config or ReactReportAgentConfig(), llm)

    def _initialize_tools(self) -> List[Any]:
        """
        Initialize report generation tools.

        Returns:
            List[Any]: List of initialized tools
        """
        # Get base tools from parent class
        base_tools = super()._initialize_tools()

        # Create a reference to the agent instance for use in the tools
        agent_instance = self

        # Define tool functions without decorators
        def extract_report_requirements_func(message: str) -> str:
            """
            Extract report-specific requirements from the user's message.

            Args:
                message: The user's message containing report requirements

            Returns:
                str: JSON string of extracted requirements
            """
            # Default requirements
            requirements = {
                "title": "Untitled Report",
                "sections": agent_instance.config.default_sections,
                "style": "formal",
                "include_toc": True,
                "include_references": True,
                "include_appendices": False,
                "document_type": "report",
                "formality_level": agent_instance.config.formality_level,
                "include_executive_summary": agent_instance.config.include_executive_summary,
                "include_recommendations": agent_instance.config.include_recommendations,
                "include_charts": agent_instance.config.include_charts
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

            # Check for executive summary
            requirements["include_executive_summary"] = "executive summary" in message.lower()

            # Check for recommendations
            requirements["include_recommendations"] = "recommendations" in message.lower() or "recommendation" in message.lower()

            # Check for charts
            requirements["include_charts"] = "chart" in message.lower() or "charts" in message.lower() or "graph" in message.lower() or "graphs" in message.lower()

            # Check for formality level
            if "formal" in message.lower():
                requirements["formality_level"] = "high"
            elif "informal" in message.lower():
                requirements["formality_level"] = "low"
            elif "semi-formal" in message.lower() or "semi formal" in message.lower():
                requirements["formality_level"] = "medium"

            return json.dumps(requirements, indent=2)

        def generate_report_content_func(requirements_json: str) -> str:
            """
            Generate report content based on requirements.

            Args:
                requirements_json: JSON string of report requirements

            Returns:
                str: Generated report content
            """
            try:
                requirements = json.loads(requirements_json)

                # Create a prompt for report generation
                prompt = f"""
                Generate a professional report with the following specifications:

                Title: {requirements.get('title', 'Untitled Report')}
                Sections: {', '.join(requirements.get('sections', agent_instance.config.default_sections))}
                Formality Level: {requirements.get('formality_level', agent_instance.config.formality_level)}
                Include Executive Summary: {requirements.get('include_executive_summary', True)}
                Include Recommendations: {requirements.get('include_recommendations', True)}
                Include Charts: {requirements.get('include_charts', False)}
                Include Table of Contents: {requirements.get('include_toc', True)}
                Include References: {requirements.get('include_references', True)}
                Include Appendices: {requirements.get('include_appendices', False)}

                The report should be well-structured, informative, and formatted in Markdown.
                Use professional language appropriate for a formal business or academic context.
                If including charts, describe them in text (e.g., [Chart: Title - Description]).
                """

                # Generate the report using the LLM
                response = agent_instance.llm.invoke([
                    {"role": "system", "content": "You are a professional report writer specializing in creating formal, well-structured reports."},
                    {"role": "user", "content": prompt}
                ])

                # Extract content from response
                if hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)

                return content

            except Exception as e:
                return f"Error generating report content: {str(e)}"

        # Create Tool objects
        extract_report_requirements_tool = Tool(
            name="extract_report_requirements",
            description="Extract report-specific requirements from the user's message",
            func=extract_report_requirements_func
        )

        generate_report_content_tool = Tool(
            name="generate_report_content",
            description="Generate report content based on requirements",
            func=generate_report_content_func
        )

        # Replace the base document tools with report-specific tools
        report_tools = []

        # Add the report-specific tools
        report_tools.append(extract_report_requirements_tool)
        report_tools.append(generate_report_content_tool)

        # Add the save_document tool from base tools
        for tool in base_tools:
            if tool.name == "save_document":
                report_tools.append(tool)
                break

        return report_tools
