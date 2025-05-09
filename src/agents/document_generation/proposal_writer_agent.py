"""
Proposal Writer Agent for Multi-Agent System

This module implements a specialized document generation agent for business proposals.
It extends the base document agent with proposal-specific features.
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


class ProposalFormat(DocumentFormat):
    """Model for proposal format specifications."""
    executive_summary: bool = True
    include_budget: bool = True
    include_timeline: bool = True
    include_deliverables: bool = True
    proposal_type: Literal["business", "project", "grant", "research"] = "business"
    target_audience: str = "business"


class ProposalGenerationResult(DocumentGenerationResult):
    """Model for a proposal generation result."""
    format: ProposalFormat


class ProposalWriterAgentConfig(BaseDocumentAgentConfig):
    """Configuration for the proposal writer agent."""
    documents_dir: str = "./generated_documents/proposals"
    metadata_dir: str = "./generated_documents/proposals/metadata"
    system_message: str = """
    You are a proposal writer agent that creates persuasive, well-structured business proposals based on user requirements.
    
    Your job is to:
    1. Understand the proposal requirements
    2. Generate a compelling proposal with appropriate structure and content
    3. Format the proposal according to the specified style
    4. Include executive summaries, budgets, timelines, and other proposal-specific elements
    5. Provide the proposal content and any relevant metadata
    
    Always ensure that the proposal is well-organized, persuasive, and meets professional standards.
    Use clear, concise language, highlight value propositions, and include specific, actionable details.
    """


class ProposalWriterAgent(BaseDocumentAgent):
    """
    Proposal writer agent that creates business proposals.
    """

    def __init__(self, config: ProposalWriterAgentConfig = ProposalWriterAgentConfig()):
        """
        Initialize the proposal writer agent.

        Args:
            config: Configuration for the proposal writer agent
        """
        super().__init__(config)
        self.config = config

    def _extract_document_requirements(self, message: str) -> Dict[str, Any]:
        """
        Extract proposal requirements from the message.

        Args:
            message: User message

        Returns:
            Dict[str, Any]: Extracted proposal requirements
        """
        # Get base requirements
        requirements = super()._extract_document_requirements(message)
        
        # Default proposal sections if none specified
        if requirements["sections"] == ["Introduction", "Main Content", "Conclusion"]:
            requirements["sections"] = [
                "Executive Summary",
                "Problem Statement",
                "Proposed Solution",
                "Methodology",
                "Timeline",
                "Budget",
                "Team",
                "Conclusion"
            ]
        
        # Proposal-specific requirements
        requirements["executive_summary"] = not ("no executive summary" in message.lower())
        requirements["include_budget"] = not ("no budget" in message.lower())
        requirements["include_timeline"] = not ("no timeline" in message.lower())
        requirements["include_deliverables"] = not ("no deliverables" in message.lower())
        
        # Determine proposal type
        if "business proposal" in message.lower():
            requirements["proposal_type"] = "business"
        elif "project proposal" in message.lower():
            requirements["proposal_type"] = "project"
        elif "grant proposal" in message.lower():
            requirements["proposal_type"] = "grant"
        elif "research proposal" in message.lower():
            requirements["proposal_type"] = "research"
        
        # Extract target audience
        if "audience:" in message.lower():
            audience_line = [line for line in message.split("\n") if "audience:" in line.lower()]
            if audience_line:
                requirements["target_audience"] = audience_line[0].split("audience:")[1].strip()
        elif "for investors" in message.lower():
            requirements["target_audience"] = "investors"
        elif "for clients" in message.lower():
            requirements["target_audience"] = "clients"
        elif "for management" in message.lower():
            requirements["target_audience"] = "management"
        elif "for government" in message.lower():
            requirements["target_audience"] = "government"
        
        return requirements

    def generate_document(self, message: str) -> ProposalGenerationResult:
        """
        Generate a proposal based on the message.

        Args:
            message: User message

        Returns:
            ProposalGenerationResult: Generated proposal
        """
        # Extract proposal requirements
        requirements = self._extract_document_requirements(message)
        
        # Create proposal format
        proposal_format = ProposalFormat(**requirements)
        
        # Generate proposal content
        response = self.llm.invoke(
            self.prompt.format(
                messages=[{"role": "user", "content": message}],
                document_format=proposal_format.model_dump_json()
            )
        )
        
        # Extract content
        if isinstance(response, dict):
            content = response.get("content", "")
        elif hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)
        
        # Save proposal if configured
        save_result = self._save_document(content, proposal_format)
        
        # Create result
        result = ProposalGenerationResult(
            content=content,
            format=proposal_format,
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
        
        # Generate proposal
        try:
            result = self.generate_document(message)
            
            # Format the result for the LLM
            formatted_result = f"""
            Proposal generated successfully!
            Title: {result.format.title}
            Proposal Type: {result.format.proposal_type}
            Target Audience: {result.format.target_audience}
            Word Count: {result.word_count}
            """
            
            features = []
            if result.format.executive_summary:
                features.append("Executive Summary")
            if result.format.include_budget:
                features.append("Budget")
            if result.format.include_timeline:
                features.append("Timeline")
            if result.format.include_deliverables:
                features.append("Deliverables")
            
            if features:
                formatted_result += f"\nFeatures: {', '.join(features)}"
            
            if result.file_path:
                formatted_result += f"\nSaved to: {result.file_path}"
            
            error = None
        except Exception as e:
            formatted_result = f"Error generating proposal: {str(e)}"
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
            state["agent_outputs"]["proposal_generation"] = result.model_dump()
        else:
            state["agent_outputs"]["proposal_generation"] = {"error": error}
        
        state["messages"].append({"role": "assistant", "content": response.content})
        
        return state


# Example usage
if __name__ == "__main__":
    # Create proposal writer agent
    proposal_agent = ProposalWriterAgent()
    
    # Test with a proposal generation request
    state = {
        "messages": [{"role": "user", "content": "Write a business proposal for a new software product for investors. Include budget and timeline."}],
        "agent_outputs": {}
    }
    
    updated_state = proposal_agent(state)
    print(updated_state["messages"][-1]["content"])
