"""
Document Generation Agents for Multi-Agent System (Part 2)

This module implements additional document generation agents:
1. Academic Writer Agent: Research papers with citations
2. Proposal Writer Agent: Business proposals with budgets and timelines
3. Planning Document Agent: Project plans and specifications
"""

import os
import json
import time
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field

# Import from part 1
from .document_generation import BaseDocumentAgent, BaseDocumentAgentConfig

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage


class AcademicWriterAgentConfig(BaseDocumentAgentConfig):
    """Configuration for the academic writer agent."""
    documents_dir: str = "./generated_documents/academic"
    metadata_dir: str = "./generated_documents/academic/metadata"
    citation_style: Literal["APA", "MLA", "Chicago", "Harvard"] = "APA"
    field_of_study: str = "general"
    include_abstract: bool = True
    include_figures: bool = True
    include_equations: bool = True
    include_references: bool = True


class AcademicWriterAgent(BaseDocumentAgent):
    """
    Academic writer agent that can create research papers with citations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the academic writer agent.
        
        Args:
            config: Configuration for the academic writer agent
        """
        super().__init__(config or AcademicWriterAgentConfig())
    
    def _generate_document(self, params):
        """
        Generate an academic paper based on parameters.
        
        Args:
            params: Academic paper parameters
            
        Returns:
            str: Generated academic paper
        """
        # Create prompt for academic paper generation
        system_message = f"""
        You are a professional academic writer. Create a research paper with the following specifications:
        
        Title: {params.get('title', 'Untitled Academic Paper')}
        Topic: {params.get('topic', '')}
        Sections: {', '.join(params.get('sections', ['Abstract', 'Introduction', 'Literature Review', 'Methodology', 'Results', 'Discussion', 'Conclusion', 'References']))}
        Citation Style: {self.config.citation_style}
        Field of Study: {self.config.field_of_study}
        
        Additional requirements:
        - Include abstract: {self.config.include_abstract}
        - Include figures: {self.config.include_figures}
        - Include equations: {self.config.include_equations}
        - Include references: {self.config.include_references}
        {json.dumps(params.get('requirements', {}), indent=2)}
        
        Format the paper with Markdown, including appropriate headings, lists, and emphasis.
        Start with "Document format: Academic Paper" followed by a blank line and then "Title: {params.get('title', 'Untitled Academic Paper')}".
        """
        
        # Generate academic paper
        response = self.llm.invoke([
            SystemMessage(content=system_message),
            {"role": "user", "content": f"Write an academic paper about {params.get('topic', '')}"}
        ])
        
        return response.content
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state using the academic writer agent.
        
        Args:
            state: Current state of the system
            
        Returns:
            Dict[str, Any]: Updated state after academic paper generation
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else ""
        
        # Parse document request
        params = self._parse_document_request(query)
        params["document_type"] = "academic paper"
        
        # Add default sections for academic papers if not specified
        if not params.get("sections"):
            params["sections"] = ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Discussion", "Conclusion", "References"]
        
        # Generate academic paper
        academic_content = self._generate_document(params)
        
        # Count words
        word_count = self._count_words(academic_content)
        
        # Prepare metadata
        metadata = {
            "title": params.get("title", "Untitled Academic Paper"),
            "sections": params.get("sections", []),
            "citation_style": self.config.citation_style,
            "field_of_study": self.config.field_of_study,
            "include_abstract": self.config.include_abstract,
            "include_figures": self.config.include_figures,
            "include_equations": self.config.include_equations,
            "include_references": self.config.include_references,
            "word_count": word_count
        }
        
        # Save academic paper if enabled
        file_path = self._save_document(academic_content, metadata)
        
        # Update metadata with file path
        if file_path:
            metadata["file_path"] = file_path
        
        # Update state with academic paper generation output
        state["agent_outputs"]["academic_generation"] = {
            "content": academic_content,
            "word_count": word_count,
            "format": {
                **params,
                "citation_style": self.config.citation_style,
                "field_of_study": self.config.field_of_study
            },
            "file_path": file_path
        }
        
        # Add a message with the academic paper generation result
        message = f"I've written an academic paper titled '{params.get('title', 'Untitled Academic Paper')}' with {word_count} words using {self.config.citation_style} citation style."
        if file_path:
            message += f" The paper has been saved to {file_path}."
        
        state["messages"].append({"role": "assistant", "content": message})
        
        return state


class ProposalWriterAgentConfig(BaseDocumentAgentConfig):
    """Configuration for the proposal writer agent."""
    documents_dir: str = "./generated_documents/proposals"
    metadata_dir: str = "./generated_documents/proposals/metadata"
    proposal_type: Literal["business", "research", "grant", "project"] = "business"
    target_audience: str = "stakeholders"
    include_budget: bool = True
    include_timeline: bool = True
    include_executive_summary: bool = True


class ProposalWriterAgent(BaseDocumentAgent):
    """
    Proposal writer agent that can create business proposals with budgets and timelines.
    """
    
    def __init__(self, config=None):
        """
        Initialize the proposal writer agent.
        
        Args:
            config: Configuration for the proposal writer agent
        """
        super().__init__(config or ProposalWriterAgentConfig())
    
    def _generate_document(self, params):
        """
        Generate a proposal based on parameters.
        
        Args:
            params: Proposal parameters
            
        Returns:
            str: Generated proposal
        """
        # Create prompt for proposal generation
        system_message = f"""
        You are a professional proposal writer. Create a {self.config.proposal_type} proposal with the following specifications:
        
        Title: {params.get('title', 'Untitled Proposal')}
        Topic: {params.get('topic', '')}
        Sections: {', '.join(params.get('sections', ['Executive Summary', 'Introduction', 'Objectives', 'Methodology', 'Budget', 'Timeline', 'Conclusion']))}
        Target Audience: {self.config.target_audience}
        
        Additional requirements:
        - Include executive summary: {self.config.include_executive_summary}
        - Include budget: {self.config.include_budget}
        - Include timeline: {self.config.include_timeline}
        {json.dumps(params.get('requirements', {}), indent=2)}
        
        Format the proposal with Markdown, including appropriate headings, lists, and emphasis.
        Start with "Document format: {self.config.proposal_type.capitalize()} Proposal" followed by a blank line and then "Title: {params.get('title', 'Untitled Proposal')}".
        """
        
        # Generate proposal
        response = self.llm.invoke([
            SystemMessage(content=system_message),
            {"role": "user", "content": f"Write a {self.config.proposal_type} proposal about {params.get('topic', '')}"}
        ])
        
        return response.content
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state using the proposal writer agent.
        
        Args:
            state: Current state of the system
            
        Returns:
            Dict[str, Any]: Updated state after proposal generation
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else ""
        
        # Parse document request
        params = self._parse_document_request(query)
        params["document_type"] = f"{self.config.proposal_type} proposal"
        
        # Add default sections for proposals if not specified
        if not params.get("sections"):
            params["sections"] = ["Executive Summary", "Introduction", "Objectives", "Methodology", "Budget", "Timeline", "Conclusion"]
        
        # Generate proposal
        proposal_content = self._generate_document(params)
        
        # Count words
        word_count = self._count_words(proposal_content)
        
        # Prepare metadata
        metadata = {
            "title": params.get("title", "Untitled Proposal"),
            "sections": params.get("sections", []),
            "proposal_type": self.config.proposal_type,
            "target_audience": self.config.target_audience,
            "include_budget": self.config.include_budget,
            "include_timeline": self.config.include_timeline,
            "include_executive_summary": self.config.include_executive_summary,
            "word_count": word_count
        }
        
        # Save proposal if enabled
        file_path = self._save_document(proposal_content, metadata)
        
        # Update metadata with file path
        if file_path:
            metadata["file_path"] = file_path
        
        # Update state with proposal generation output
        state["agent_outputs"]["proposal_generation"] = {
            "content": proposal_content,
            "word_count": word_count,
            "format": {
                **params,
                "proposal_type": self.config.proposal_type,
                "target_audience": self.config.target_audience
            },
            "file_path": file_path
        }
        
        # Add a message with the proposal generation result
        message = f"I've written a {self.config.proposal_type} proposal titled '{params.get('title', 'Untitled Proposal')}' with {word_count} words."
        if file_path:
            message += f" The proposal has been saved to {file_path}."
        
        state["messages"].append({"role": "assistant", "content": message})
        
        return state
