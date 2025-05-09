"""
Document Generation Agents for Multi-Agent System

This module implements various document generation agents that can create different types of documents:
1. Base Document Agent: Generic document generation
2. Report Writer Agent: Formal reports with executive summaries
3. Blog Writer Agent: Blog posts and articles with different tones
4. Academic Writer Agent: Research papers with citations
5. Proposal Writer Agent: Business proposals with budgets and timelines
6. Planning Document Agent: Project plans and specifications
"""

import os
import json
import time
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage


class BaseDocumentAgentConfig(BaseModel):
    """Configuration for the base document generation agent."""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-opus-20240229"
    temperature: float = 0.7
    documents_dir: str = "./generated_documents"
    metadata_dir: str = "./generated_documents/metadata"
    save_documents: bool = True
    max_retries: int = 3
    retry_delay: int = 2


class BaseDocumentAgent:
    """
    Base document generation agent that can create generic documents.
    """
    
    def __init__(self, config=None):
        """
        Initialize the document generation agent.
        
        Args:
            config: Configuration for the document generation agent
        """
        self.config = config or BaseDocumentAgentConfig()
        
        # Create directories if they don't exist
        if self.config.save_documents:
            os.makedirs(self.config.documents_dir, exist_ok=True)
            os.makedirs(self.config.metadata_dir, exist_ok=True)
        
        # Initialize LLM
        if self.config.llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=self.config.openai_model,
                temperature=self.config.temperature
            )
        else:
            self.llm = ChatAnthropic(
                model=self.config.anthropic_model,
                temperature=self.config.temperature
            )
    
    def _generate_document_id(self):
        """
        Generate a unique document ID.
        
        Returns:
            str: Unique document ID
        """
        timestamp = int(time.time())
        random_suffix = hashlib.md5(uuid.uuid4().bytes).hexdigest()[:8]
        return f"untitled_document_{timestamp}_{random_suffix}"
    
    def _save_document(self, content, metadata):
        """
        Save the document to a file.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            str: Path to the saved document
        """
        if not self.config.save_documents:
            return None
        
        # Generate document ID if not provided
        doc_id = metadata.get("id", self._generate_document_id())
        
        # Create file path
        file_path = os.path.join(self.config.documents_dir, f"{doc_id}.md")
        metadata_path = os.path.join(self.config.metadata_dir, f"{doc_id}.json")
        
        # Save document
        with open(file_path, "w") as f:
            f.write(content)
        
        # Update metadata with file path
        metadata["file_path"] = file_path
        metadata["created_at"] = time.time()
        
        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return file_path
    
    def _count_words(self, text):
        """
        Count the number of words in a text.
        
        Args:
            text: Text to count words in
            
        Returns:
            int: Number of words
        """
        return len(text.split())
    
    def _parse_document_request(self, query):
        """
        Parse a document generation request to extract parameters.
        
        Args:
            query: User query
            
        Returns:
            Dict: Document parameters
        """
        # Use the LLM to parse the request
        system_message = """
        You are a document request parser. Extract the following information from the user's request:
        - Document type (report, blog post, academic paper, proposal, plan, etc.)
        - Title (if specified, otherwise "Untitled Document")
        - Topic or subject
        - Sections to include
        - Style or tone (formal, casual, technical, etc.)
        - Any specific requirements (word count, citations, etc.)
        
        Return the information as a JSON object with the following structure:
        {
            "document_type": "...",
            "title": "...",
            "topic": "...",
            "sections": ["...", "..."],
            "style": "...",
            "requirements": {
                "word_count": 0,
                "include_citations": false,
                "include_images": false
            }
        }
        """
        
        response = self.llm.invoke([
            SystemMessage(content=system_message),
            {"role": "user", "content": query}
        ])
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            content = response.content
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Default parameters
                return {
                    "document_type": "generic",
                    "title": "Untitled Document",
                    "topic": query,
                    "sections": ["Introduction", "Body", "Conclusion"],
                    "style": "default",
                    "requirements": {}
                }
        except Exception as e:
            print(f"Error parsing document request: {str(e)}")
            # Default parameters
            return {
                "document_type": "generic",
                "title": "Untitled Document",
                "topic": query,
                "sections": ["Introduction", "Body", "Conclusion"],
                "style": "default",
                "requirements": {}
            }
    
    def _generate_document(self, params):
        """
        Generate a document based on parameters.
        
        Args:
            params: Document parameters
            
        Returns:
            str: Generated document
        """
        # Create prompt for document generation
        system_message = f"""
        You are a professional document writer. Create a {params.get('document_type', 'document')} with the following specifications:
        
        Title: {params.get('title', 'Untitled Document')}
        Topic: {params.get('topic', '')}
        Sections: {', '.join(params.get('sections', ['Introduction', 'Body', 'Conclusion']))}
        Style: {params.get('style', 'default')}
        
        Additional requirements:
        {json.dumps(params.get('requirements', {}), indent=2)}
        
        Format the document with Markdown, including appropriate headings, lists, and emphasis.
        Start with "Document format: {params.get('document_type', 'Generic')}" followed by a blank line and then "Title: {params.get('title', 'Untitled Document')}".
        """
        
        # Generate document
        response = self.llm.invoke([
            SystemMessage(content=system_message),
            {"role": "user", "content": f"Generate a {params.get('document_type', 'document')} about {params.get('topic', '')}"}
        ])
        
        return response.content
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state using the document generation agent.
        
        Args:
            state: Current state of the system
            
        Returns:
            Dict[str, Any]: Updated state after document generation
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else ""
        
        # Parse document request
        params = self._parse_document_request(query)
        
        # Generate document
        document_content = self._generate_document(params)
        
        # Count words
        word_count = self._count_words(document_content)
        
        # Prepare metadata
        metadata = {
            "title": params.get("title", "Untitled Document"),
            "sections": params.get("sections", []),
            "style": params.get("style", "default"),
            "include_toc": params.get("requirements", {}).get("include_toc", False),
            "include_references": params.get("requirements", {}).get("include_references", False),
            "include_appendices": params.get("requirements", {}).get("include_appendices", False),
            "word_count": word_count
        }
        
        # Save document if enabled
        file_path = self._save_document(document_content, metadata)
        
        # Update metadata with file path
        if file_path:
            metadata["file_path"] = file_path
        
        # Update state with document generation output
        state["agent_outputs"]["document_generation"] = {
            "content": document_content,
            "word_count": word_count,
            "format": params,
            "file_path": file_path
        }
        
        # Add a message with the document generation result
        message = f"I've generated a {params.get('document_type', 'document')} titled '{params.get('title', 'Untitled Document')}' with {word_count} words."
        if file_path:
            message += f" The document has been saved to {file_path}."
        
        state["messages"].append({"role": "assistant", "content": message})
        
        return state


class ReportWriterAgentConfig(BaseDocumentAgentConfig):
    """Configuration for the report writer agent."""
    documents_dir: str = "./generated_documents/reports"
    metadata_dir: str = "./generated_documents/reports/metadata"
    formality_level: Literal["high", "medium", "low"] = "high"
    include_executive_summary: bool = True
    include_recommendations: bool = True
    include_charts: bool = True


class ReportWriterAgent(BaseDocumentAgent):
    """
    Report writer agent that can create formal reports with executive summaries and recommendations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the report writer agent.
        
        Args:
            config: Configuration for the report writer agent
        """
        super().__init__(config or ReportWriterAgentConfig())
    
    def _generate_document(self, params):
        """
        Generate a report based on parameters.
        
        Args:
            params: Report parameters
            
        Returns:
            str: Generated report
        """
        # Create prompt for report generation
        system_message = f"""
        You are a professional report writer. Create a formal report with the following specifications:
        
        Title: {params.get('title', 'Untitled Report')}
        Topic: {params.get('topic', '')}
        Sections: {', '.join(params.get('sections', ['Executive Summary', 'Introduction', 'Findings', 'Recommendations', 'Conclusion']))}
        Formality Level: {self.config.formality_level}
        
        Additional requirements:
        - Include an executive summary: {self.config.include_executive_summary}
        - Include recommendations: {self.config.include_recommendations}
        - Include charts and figures: {self.config.include_charts}
        {json.dumps(params.get('requirements', {}), indent=2)}
        
        Format the report with Markdown, including appropriate headings, lists, and emphasis.
        Start with "Document format: Report" followed by a blank line and then "Title: {params.get('title', 'Untitled Report')}".
        """
        
        # Generate report
        response = self.llm.invoke([
            SystemMessage(content=system_message),
            {"role": "user", "content": f"Generate a formal report about {params.get('topic', '')}"}
        ])
        
        return response.content
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state using the report writer agent.
        
        Args:
            state: Current state of the system
            
        Returns:
            Dict[str, Any]: Updated state after report generation
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else ""
        
        # Parse document request
        params = self._parse_document_request(query)
        params["document_type"] = "report"
        
        # Add default sections for reports if not specified
        if not params.get("sections"):
            params["sections"] = ["Executive Summary", "Introduction", "Findings", "Recommendations", "Conclusion"]
        
        # Generate report
        report_content = self._generate_document(params)
        
        # Count words
        word_count = self._count_words(report_content)
        
        # Prepare metadata
        metadata = {
            "title": params.get("title", "Untitled Report"),
            "sections": params.get("sections", []),
            "formality_level": self.config.formality_level,
            "include_executive_summary": self.config.include_executive_summary,
            "include_recommendations": self.config.include_recommendations,
            "include_charts": self.config.include_charts,
            "word_count": word_count
        }
        
        # Save report if enabled
        file_path = self._save_document(report_content, metadata)
        
        # Update metadata with file path
        if file_path:
            metadata["file_path"] = file_path
        
        # Update state with report generation output
        state["agent_outputs"]["report_generation"] = {
            "content": report_content,
            "word_count": word_count,
            "format": params,
            "file_path": file_path
        }
        
        # Add a message with the report generation result
        message = f"I've generated a formal report titled '{params.get('title', 'Untitled Report')}' with {word_count} words."
        if file_path:
            message += f" The report has been saved to {file_path}."
        
        state["messages"].append({"role": "assistant", "content": message})
        
        return state


class BlogWriterAgentConfig(BaseDocumentAgentConfig):
    """Configuration for the blog writer agent."""
    documents_dir: str = "./generated_documents/blogs"
    metadata_dir: str = "./generated_documents/blogs/metadata"
    tone: Literal["casual", "professional", "technical", "humorous"] = "casual"
    target_audience: str = "general"
    include_images: bool = True
    include_seo_keywords: bool = True


class BlogWriterAgent(BaseDocumentAgent):
    """
    Blog writer agent that can create blog posts and articles with different tones and styles.
    """
    
    def __init__(self, config=None):
        """
        Initialize the blog writer agent.
        
        Args:
            config: Configuration for the blog writer agent
        """
        super().__init__(config or BlogWriterAgentConfig())
    
    def _generate_document(self, params):
        """
        Generate a blog post based on parameters.
        
        Args:
            params: Blog post parameters
            
        Returns:
            str: Generated blog post
        """
        # Create prompt for blog post generation
        system_message = f"""
        You are a professional blog writer. Create a blog post with the following specifications:
        
        Title: {params.get('title', 'Untitled Blog Post')}
        Topic: {params.get('topic', '')}
        Sections: {', '.join(params.get('sections', ['Introduction', 'Main Points', 'Conclusion']))}
        Tone: {self.config.tone}
        Target Audience: {self.config.target_audience}
        
        Additional requirements:
        - Include images: {self.config.include_images}
        - Include SEO keywords: {self.config.include_seo_keywords}
        {json.dumps(params.get('requirements', {}), indent=2)}
        
        Format the blog post with Markdown, including appropriate headings, lists, and emphasis.
        Start with "Document format: Blog Post" followed by a blank line and then "Title: {params.get('title', 'Untitled Blog Post')}".
        """
        
        # Generate blog post
        response = self.llm.invoke([
            SystemMessage(content=system_message),
            {"role": "user", "content": f"Write a blog post about {params.get('topic', '')}"}
        ])
        
        return response.content
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state using the blog writer agent.
        
        Args:
            state: Current state of the system
            
        Returns:
            Dict[str, Any]: Updated state after blog post generation
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else ""
        
        # Parse document request
        params = self._parse_document_request(query)
        params["document_type"] = "blog post"
        
        # Add default sections for blog posts if not specified
        if not params.get("sections"):
            params["sections"] = ["Introduction", "Main Points", "Conclusion"]
        
        # Generate blog post
        blog_content = self._generate_document(params)
        
        # Count words
        word_count = self._count_words(blog_content)
        
        # Calculate reading time (average reading speed: 200-250 words per minute)
        reading_time_minutes = max(1, round(word_count / 225))
        
        # Prepare metadata
        metadata = {
            "title": params.get("title", "Untitled Blog Post"),
            "sections": params.get("sections", []),
            "tone": self.config.tone,
            "target_audience": self.config.target_audience,
            "reading_time_minutes": reading_time_minutes,
            "word_count": word_count
        }
        
        # Save blog post if enabled
        file_path = self._save_document(blog_content, metadata)
        
        # Update metadata with file path
        if file_path:
            metadata["file_path"] = file_path
        
        # Update state with blog post generation output
        state["agent_outputs"]["blog_generation"] = {
            "content": blog_content,
            "word_count": word_count,
            "format": {
                **params,
                "reading_time_minutes": reading_time_minutes,
                "tone": self.config.tone,
                "target_audience": self.config.target_audience
            },
            "file_path": file_path
        }
        
        # Add a message with the blog post generation result
        message = f"I've written a blog post titled '{params.get('title', 'Untitled Blog Post')}' with {word_count} words (about {reading_time_minutes} minute read)."
        if file_path:
            message += f" The blog post has been saved to {file_path}."
        
        state["messages"].append({"role": "assistant", "content": message})
        
        return state
