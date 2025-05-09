"""
Report Writer Agent for Multi-Agent System

This module implements a specialized document generation agent for formal reports.
It extends the base document agent with report-specific features.
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


class ReportFormat(DocumentFormat):
    """Model for report format specifications."""
    executive_summary: bool = True
    include_charts: bool = False
    include_tables: bool = False
    include_recommendations: bool = True
    formality_level: Literal["high", "medium", "low"] = "high"


class ReportGenerationResult(DocumentGenerationResult):
    """Model for a report generation result."""
    format: ReportFormat


class ReportWriterAgentConfig(BaseDocumentAgentConfig):
    """Configuration for the report writer agent."""
    documents_dir: str = "./generated_documents/reports"
    metadata_dir: str = "./generated_documents/reports/metadata"
    system_message: str = """
    You are a report writer agent that creates formal, well-structured reports based on user requirements.
    
    Your job is to:
    1. Understand the report requirements
    2. Generate a professional report with appropriate structure and content
    3. Format the report according to the specified style
    4. Include executive summaries, recommendations, and other report-specific elements
    5. Provide the report content and any relevant metadata
    
    Always ensure that the report is well-organized, coherent, and meets professional standards.
    Use formal language, cite sources where appropriate, and provide actionable recommendations.
    """


class ReportWriterAgent(BaseDocumentAgent):
    """
    Report writer agent that creates formal reports.
    """

    def __init__(self, config: ReportWriterAgentConfig = ReportWriterAgentConfig()):
        """
        Initialize the report writer agent.

        Args:
            config: Configuration for the report writer agent
        """
        super().__init__(config)
        self.config = config

    def _extract_document_requirements(self, message: str) -> Dict[str, Any]:
        """
        Extract report requirements from the message.

        Args:
            message: User message

        Returns:
            Dict[str, Any]: Extracted report requirements
        """
        # Get base requirements
        requirements = super()._extract_document_requirements(message)
        
        # Default report sections if none specified
        if requirements["sections"] == ["Introduction", "Main Content", "Conclusion"]:
            requirements["sections"] = [
                "Executive Summary",
                "Introduction",
                "Background",
                "Methodology",
                "Findings",
                "Analysis",
                "Recommendations",
                "Conclusion"
            ]
        
        # Report-specific requirements
        requirements["executive_summary"] = "executive summary" in message.lower()
        requirements["include_charts"] = "charts" in message.lower() or "graphs" in message.lower()
        requirements["include_tables"] = "tables" in message.lower() or "data tables" in message.lower()
        requirements["include_recommendations"] = "recommendations" in message.lower()
        
        # Determine formality level
        if "formal" in message.lower() or "professional" in message.lower():
            requirements["formality_level"] = "high"
        elif "informal" in message.lower() or "casual" in message.lower():
            requirements["formality_level"] = "low"
        else:
            requirements["formality_level"] = "medium"
        
        return requirements

    def generate_document(self, message: str) -> ReportGenerationResult:
        """
        Generate a report based on the message.

        Args:
            message: User message

        Returns:
            ReportGenerationResult: Generated report
        """
        # Extract report requirements
        requirements = self._extract_document_requirements(message)
        
        # Create report format
        report_format = ReportFormat(**requirements)
        
        # Generate report content
        response = self.llm.invoke(
            self.prompt.format(
                messages=[{"role": "user", "content": message}],
                document_format=report_format.model_dump_json()
            )
        )
        
        # Extract content
        if isinstance(response, dict):
            content = response.get("content", "")
        elif hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)
        
        # Save report if configured
        save_result = self._save_document(content, report_format)
        
        # Create result
        result = ReportGenerationResult(
            content=content,
            format=report_format,
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
        
        # Generate report
        try:
            result = self.generate_document(message)
            
            # Format the result for the LLM
            formatted_result = f"""
            Report generated successfully!
            Title: {result.format.title}
            Style: {result.format.style}
            Formality Level: {result.format.formality_level}
            Word Count: {result.word_count}
            Sections: {", ".join(result.format.sections)}
            """
            
            if result.file_path:
                formatted_result += f"\nSaved to: {result.file_path}"
            
            error = None
        except Exception as e:
            formatted_result = f"Error generating report: {str(e)}"
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
            state["agent_outputs"]["report_generation"] = result.model_dump()
        else:
            state["agent_outputs"]["report_generation"] = {"error": error}
        
        state["messages"].append({"role": "assistant", "content": response.content})
        
        return state


# Example usage
if __name__ == "__main__":
    # Create report writer agent
    report_agent = ReportWriterAgent()
    
    # Test with a report generation request
    state = {
        "messages": [{"role": "user", "content": "Generate a formal report about renewable energy trends with charts and recommendations"}],
        "agent_outputs": {}
    }
    
    updated_state = report_agent(state)
    print(updated_state["messages"][-1]["content"])
