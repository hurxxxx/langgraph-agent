"""
Document Generation Agents for Multi-Agent System

This package provides specialized document generation agents for different types of documents:
- Base document agent: Generic document generation
- Report writer agent: Formal reports
- Blog writer agent: Blog posts and articles
- Academic writer agent: Academic papers
- Proposal writer agent: Business proposals
"""

from .base_document_agent import (
    BaseDocumentAgent,
    BaseDocumentAgentConfig,
    DocumentFormat,
    DocumentGenerationResult
)

from .report_writer_agent import (
    ReportWriterAgent,
    ReportWriterAgentConfig,
    ReportFormat,
    ReportGenerationResult
)

from .blog_writer_agent import (
    BlogWriterAgent,
    BlogWriterAgentConfig,
    BlogFormat,
    BlogGenerationResult
)

from .academic_writer_agent import (
    AcademicWriterAgent,
    AcademicWriterAgentConfig,
    AcademicFormat,
    AcademicGenerationResult,
    CitationStyle
)

from .proposal_writer_agent import (
    ProposalWriterAgent,
    ProposalWriterAgentConfig,
    ProposalFormat,
    ProposalGenerationResult
)

__all__ = [
    # Base document agent
    'BaseDocumentAgent',
    'BaseDocumentAgentConfig',
    'DocumentFormat',
    'DocumentGenerationResult',
    
    # Report writer agent
    'ReportWriterAgent',
    'ReportWriterAgentConfig',
    'ReportFormat',
    'ReportGenerationResult',
    
    # Blog writer agent
    'BlogWriterAgent',
    'BlogWriterAgentConfig',
    'BlogFormat',
    'BlogGenerationResult',
    
    # Academic writer agent
    'AcademicWriterAgent',
    'AcademicWriterAgentConfig',
    'AcademicFormat',
    'AcademicGenerationResult',
    'CitationStyle',
    
    # Proposal writer agent
    'ProposalWriterAgent',
    'ProposalWriterAgentConfig',
    'ProposalFormat',
    'ProposalGenerationResult'
]
