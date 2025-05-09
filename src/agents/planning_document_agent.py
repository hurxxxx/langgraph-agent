"""
Planning Document Agent for Multi-Agent System

This module implements a planning document agent that can create project plans and specifications.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field

# Import from document_generation
from .document_generation import BaseDocumentAgent, BaseDocumentAgentConfig

# Import LangChain components
from langchain_core.messages import SystemMessage


class PlanningDocumentAgentConfig(BaseDocumentAgentConfig):
    """Configuration for the planning document agent."""
    documents_dir: str = "./generated_documents/plans"
    metadata_dir: str = "./generated_documents/plans/metadata"
    plan_type: Literal["project", "strategic", "implementation", "technical"] = "project"
    include_gantt_chart: bool = True
    include_resource_allocation: bool = True
    include_risk_assessment: bool = True
    include_success_metrics: bool = True


class PlanningDocumentAgent(BaseDocumentAgent):
    """
    Planning document agent that can create project plans and specifications.
    """
    
    def __init__(self, config=None):
        """
        Initialize the planning document agent.
        
        Args:
            config: Configuration for the planning document agent
        """
        super().__init__(config or PlanningDocumentAgentConfig())
        
        # Create directories if they don't exist
        if self.config.save_documents:
            os.makedirs(self.config.documents_dir, exist_ok=True)
            os.makedirs(self.config.metadata_dir, exist_ok=True)
    
    def _generate_document(self, params):
        """
        Generate a planning document based on parameters.
        
        Args:
            params: Planning document parameters
            
        Returns:
            str: Generated planning document
        """
        # Create prompt for planning document generation
        system_message = f"""
        You are a professional project planner. Create a {self.config.plan_type} plan with the following specifications:
        
        Title: {params.get('title', 'Untitled Plan')}
        Topic: {params.get('topic', '')}
        Sections: {', '.join(params.get('sections', ['Executive Summary', 'Project Overview', 'Objectives', 'Scope', 'Timeline', 'Resources', 'Risk Assessment', 'Success Metrics']))}
        
        Additional requirements:
        - Include Gantt chart: {self.config.include_gantt_chart}
        - Include resource allocation: {self.config.include_resource_allocation}
        - Include risk assessment: {self.config.include_risk_assessment}
        - Include success metrics: {self.config.include_success_metrics}
        {json.dumps(params.get('requirements', {}), indent=2)}
        
        Format the plan with Markdown, including appropriate headings, lists, and emphasis.
        Start with "Document format: {self.config.plan_type.capitalize()} Plan" followed by a blank line and then "Title: {params.get('title', 'Untitled Plan')}".
        
        If including a Gantt chart, use ASCII art or markdown tables to represent it.
        For resource allocation, create a detailed table with roles, responsibilities, and time commitments.
        For risk assessment, include probability, impact, and mitigation strategies.
        For success metrics, define clear KPIs and measurement methods.
        """
        
        # Generate planning document
        response = self.llm.invoke([
            SystemMessage(content=system_message),
            {"role": "user", "content": f"Create a {self.config.plan_type} plan for {params.get('topic', '')}"}
        ])
        
        return response.content
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state using the planning document agent.
        
        Args:
            state: Current state of the system
            
        Returns:
            Dict[str, Any]: Updated state after planning document generation
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else ""
        
        # Parse document request
        params = self._parse_document_request(query)
        params["document_type"] = f"{self.config.plan_type} plan"
        
        # Add default sections for planning documents if not specified
        if not params.get("sections"):
            params["sections"] = ["Executive Summary", "Project Overview", "Objectives", "Scope", "Timeline", "Resources", "Risk Assessment", "Success Metrics"]
        
        # Generate planning document
        plan_content = self._generate_document(params)
        
        # Count words
        word_count = self._count_words(plan_content)
        
        # Prepare metadata
        metadata = {
            "title": params.get("title", "Untitled Plan"),
            "sections": params.get("sections", []),
            "plan_type": self.config.plan_type,
            "include_gantt_chart": self.config.include_gantt_chart,
            "include_resource_allocation": self.config.include_resource_allocation,
            "include_risk_assessment": self.config.include_risk_assessment,
            "include_success_metrics": self.config.include_success_metrics,
            "word_count": word_count
        }
        
        # Save planning document if enabled
        file_path = self._save_document(plan_content, metadata)
        
        # Update metadata with file path
        if file_path:
            metadata["file_path"] = file_path
        
        # Update state with planning document generation output
        state["agent_outputs"]["planning_generation"] = {
            "content": plan_content,
            "word_count": word_count,
            "format": {
                **params,
                "plan_type": self.config.plan_type
            },
            "file_path": file_path
        }
        
        # Add a message with the planning document generation result
        message = f"I've created a {self.config.plan_type} plan titled '{params.get('title', 'Untitled Plan')}' with {word_count} words."
        if file_path:
            message += f" The plan has been saved to {file_path}."
        
        state["messages"].append({"role": "assistant", "content": message})
        
        return state
