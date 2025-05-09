"""
ReAct Proposal Writer Agent using LangGraph's create_react_agent

This module implements a proposal writer agent using LangGraph's create_react_agent function
and specialized proposal writing tools.
"""

import os
import json
from typing import Dict, List, Any, Optional, Literal

# Import base document agent
from src.agents.react_document_agent import ReactDocumentAgent, ReactDocumentAgentConfig

# LangChain components
from langchain_core.tools import tool, Tool


class ReactProposalAgentConfig(ReactDocumentAgentConfig):
    """Configuration for the ReAct proposal writer agent."""
    # Document type
    document_type: str = "proposal"

    # Proposal-specific configuration
    default_sections: List[str] = ["Executive Summary", "Introduction", "Project Overview", "Methodology", "Timeline", "Budget", "Team", "Conclusion"]
    proposal_type: Literal["business", "grant", "project", "research"] = "business"
    include_budget: bool = True
    include_timeline: bool = True
    include_team: bool = True
    include_executive_summary: bool = True

    # File storage configuration
    documents_dir: str = "./generated_documents/proposals"
    metadata_dir: str = "./generated_documents/proposals/metadata"

    # System message
    system_message: str = """
    You are a proposal writer agent that creates persuasive, well-structured proposals based on user requirements.

    Your job is to:
    1. Analyze the user's proposal request carefully
    2. Extract key requirements like title, proposal type, and content needs
    3. Generate a persuasive proposal that meets these requirements
    4. Save the proposal and provide a summary of what was created

    Proposals should include:
    - An executive summary that clearly presents the value proposition
    - Clear project overview and objectives
    - Detailed methodology and approach
    - Realistic timeline and milestones
    - Comprehensive budget breakdown
    - Team qualifications and responsibilities
    - Strong conclusion with next steps

    Use the available tools to:
    - Extract proposal requirements from the user's request
    - Generate the proposal content
    - Save the proposal to a file

    Always ensure the proposal is persuasive, well-organized, and meets the user's requirements.
    """


class ReactProposalAgent(ReactDocumentAgent):
    """
    Proposal writer agent using LangGraph's create_react_agent function.
    Specializes in creating persuasive business proposals with budgets and timelines.
    """

    def __init__(self, config=None, llm=None):
        """
        Initialize the proposal writer agent.

        Args:
            config: Configuration for the proposal writer agent
            llm: Optional language model to use
        """
        # Initialize with proposal-specific configuration
        super().__init__(config or ReactProposalAgentConfig(), llm)

    def _initialize_tools(self) -> List[Any]:
        """
        Initialize proposal writing tools.

        Returns:
            List[Any]: List of initialized tools
        """
        # Get base tools from parent class
        base_tools = super()._initialize_tools()

        # Create a reference to the agent instance for use in the tools
        agent_instance = self

        # Define tool functions without decorators
        def extract_proposal_requirements_func(message: str) -> str:
            """
            Extract proposal-specific requirements from the user's message.

            Args:
                message: The user's message containing proposal requirements

            Returns:
                str: JSON string of extracted requirements
            """
            # Default requirements
            requirements = {
                "title": "Untitled Proposal",
                "sections": agent_instance.config.default_sections,
                "style": "professional",
                "include_toc": True,
                "include_references": False,
                "include_appendices": True,
                "document_type": "proposal",
                "proposal_type": agent_instance.config.proposal_type,
                "include_budget": agent_instance.config.include_budget,
                "include_timeline": agent_instance.config.include_timeline,
                "include_team": agent_instance.config.include_team,
                "include_executive_summary": agent_instance.config.include_executive_summary
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

            # Extract proposal type if present
            if "proposal type:" in message.lower() or "type:" in message.lower():
                type_text = message.lower().split("proposal type:")[1].split("\n")[0].strip() if "proposal type:" in message.lower() else message.lower().split("type:")[1].split("\n")[0].strip()
                if "business" in type_text:
                    requirements["proposal_type"] = "business"
                elif "grant" in type_text:
                    requirements["proposal_type"] = "grant"
                elif "project" in type_text:
                    requirements["proposal_type"] = "project"
                elif "research" in type_text:
                    requirements["proposal_type"] = "research"

            # Check for budget
            requirements["include_budget"] = "budget" in message.lower() or "cost" in message.lower() or "pricing" in message.lower()

            # Check for timeline
            requirements["include_timeline"] = "timeline" in message.lower() or "schedule" in message.lower() or "milestones" in message.lower()

            # Check for team
            requirements["include_team"] = "team" in message.lower() or "personnel" in message.lower() or "staff" in message.lower()

            # Check for executive summary
            requirements["include_executive_summary"] = "executive summary" in message.lower() or "summary" in message.lower()

            return json.dumps(requirements, indent=2)

        def generate_proposal_content_func(requirements_json: str) -> str:
            """
            Generate proposal content based on requirements.

            Args:
                requirements_json: JSON string of proposal requirements

            Returns:
                str: Generated proposal content
            """
            try:
                requirements = json.loads(requirements_json)

                # Create a prompt for proposal generation
                prompt = f"""
                Generate a persuasive {requirements.get('proposal_type', agent_instance.config.proposal_type)} proposal with the following specifications:

                Title: {requirements.get('title', 'Untitled Proposal')}
                Sections: {', '.join(requirements.get('sections', agent_instance.config.default_sections))}
                Include Executive Summary: {requirements.get('include_executive_summary', True)}
                Include Budget: {requirements.get('include_budget', True)}
                Include Timeline: {requirements.get('include_timeline', True)}
                Include Team: {requirements.get('include_team', True)}

                The proposal should be well-structured, persuasive, and formatted in Markdown.
                Use professional language appropriate for a business context.
                If including a budget, create a realistic budget table with line items and costs.
                If including a timeline, create a realistic timeline with milestones and dates.
                If including team information, describe key team members, their roles, and qualifications.
                """

                # Generate the proposal using the LLM
                response = agent_instance.llm.invoke([
                    {"role": "system", "content": f"You are a professional proposal writer specializing in creating persuasive {requirements.get('proposal_type', agent_instance.config.proposal_type)} proposals."},
                    {"role": "user", "content": prompt}
                ])

                # Extract content from response
                if hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)

                return content

            except Exception as e:
                return f"Error generating proposal content: {str(e)}"

        # Create Tool objects
        extract_proposal_requirements_tool = Tool(
            name="extract_proposal_requirements",
            description="Extract proposal-specific requirements from the user's message",
            func=extract_proposal_requirements_func
        )

        generate_proposal_content_tool = Tool(
            name="generate_proposal_content",
            description="Generate proposal content based on requirements",
            func=generate_proposal_content_func
        )

        # Replace the base document tools with proposal-specific tools
        proposal_tools = []

        # Add the proposal-specific tools
        proposal_tools.append(extract_proposal_requirements_tool)
        proposal_tools.append(generate_proposal_content_tool)

        # Add the save_document tool from base tools
        for tool in base_tools:
            if tool.name == "save_document":
                proposal_tools.append(tool)
                break

        return proposal_tools
