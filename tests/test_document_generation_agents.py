"""
Test Document Generation Agents

This script tests the document generation agents' ability to generate different types of documents.
"""

import os
import sys
import json
import time
import shutil
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import document generation agents
from src.agents.document_generation import (
    BaseDocumentAgent,
    ReportWriterAgent,
    BlogWriterAgent,
    AcademicWriterAgent,
    ProposalWriterAgent
)


def test_base_document_agent():
    """
    Test the base document generation agent.
    """
    print("Testing base document generation agent...")
    
    # Initialize the base document agent
    document_agent = BaseDocumentAgent()
    
    # Test with a document generation request
    state = {
        "messages": [{"role": "user", "content": "Generate a document about artificial intelligence with sections: Introduction, History, Current Applications, Future Trends, Conclusion"}],
        "agent_outputs": {}
    }
    
    # Process the request
    updated_state = document_agent(state)
    
    # Print the response
    print("\nAgent Response:")
    print(updated_state["messages"][-1]["content"])
    
    # Check if the document was generated
    if "document_generation" in updated_state["agent_outputs"]:
        doc_output = updated_state["agent_outputs"]["document_generation"]
        
        print("\nDocument Generation Output:")
        print(f"Title: {doc_output.get('format', {}).get('title')}")
        print(f"Word Count: {doc_output.get('word_count')}")
        
        # Check if the document was saved
        file_path = doc_output.get("file_path")
        if file_path and os.path.exists(file_path):
            print(f"\nVerified: Document file exists at {file_path}")
            
            # Print the first few lines of the document
            with open(file_path, "r") as f:
                content = f.read(500)  # Read first 500 characters
                print("\nDocument Preview:")
                print(content + "...")
        else:
            print("\nWarning: Document file does not exist or could not be verified")
    else:
        print("\nError: Document generation output not found in agent outputs")
    
    return True


def test_report_writer_agent():
    """
    Test the report writer agent.
    """
    print("\nTesting report writer agent...")
    
    # Initialize the report writer agent
    report_agent = ReportWriterAgent()
    
    # Test with a report generation request
    state = {
        "messages": [{"role": "user", "content": "Generate a formal report about renewable energy trends with charts and recommendations"}],
        "agent_outputs": {}
    }
    
    # Process the request
    updated_state = report_agent(state)
    
    # Print the response
    print("\nAgent Response:")
    print(updated_state["messages"][-1]["content"])
    
    # Check if the report was generated
    if "report_generation" in updated_state["agent_outputs"]:
        report_output = updated_state["agent_outputs"]["report_generation"]
        
        print("\nReport Generation Output:")
        print(f"Title: {report_output.get('format', {}).get('title')}")
        print(f"Formality Level: {report_output.get('format', {}).get('formality_level')}")
        print(f"Word Count: {report_output.get('word_count')}")
        
        # Check if the report was saved
        file_path = report_output.get("file_path")
        if file_path and os.path.exists(file_path):
            print(f"\nVerified: Report file exists at {file_path}")
            
            # Print the first few lines of the report
            with open(file_path, "r") as f:
                content = f.read(500)  # Read first 500 characters
                print("\nReport Preview:")
                print(content + "...")
        else:
            print("\nWarning: Report file does not exist or could not be verified")
    else:
        print("\nError: Report generation output not found in agent outputs")
    
    return True


def test_blog_writer_agent():
    """
    Test the blog writer agent.
    """
    print("\nTesting blog writer agent...")
    
    # Initialize the blog writer agent
    blog_agent = BlogWriterAgent()
    
    # Test with a blog generation request
    state = {
        "messages": [{"role": "user", "content": "Write a blog post about machine learning for beginners with a casual tone. Keywords: AI, machine learning, beginners guide"}],
        "agent_outputs": {}
    }
    
    # Process the request
    updated_state = blog_agent(state)
    
    # Print the response
    print("\nAgent Response:")
    print(updated_state["messages"][-1]["content"])
    
    # Check if the blog post was generated
    if "blog_generation" in updated_state["agent_outputs"]:
        blog_output = updated_state["agent_outputs"]["blog_generation"]
        
        print("\nBlog Generation Output:")
        print(f"Title: {blog_output.get('format', {}).get('title')}")
        print(f"Target Audience: {blog_output.get('format', {}).get('target_audience')}")
        print(f"Tone: {blog_output.get('format', {}).get('tone')}")
        print(f"Word Count: {blog_output.get('word_count')}")
        print(f"Reading Time: {blog_output.get('format', {}).get('reading_time_minutes')} minutes")
        
        # Check if the blog post was saved
        file_path = blog_output.get("file_path")
        if file_path and os.path.exists(file_path):
            print(f"\nVerified: Blog post file exists at {file_path}")
            
            # Print the first few lines of the blog post
            with open(file_path, "r") as f:
                content = f.read(500)  # Read first 500 characters
                print("\nBlog Post Preview:")
                print(content + "...")
        else:
            print("\nWarning: Blog post file does not exist or could not be verified")
    else:
        print("\nError: Blog generation output not found in agent outputs")
    
    return True


def test_academic_writer_agent():
    """
    Test the academic writer agent.
    """
    print("\nTesting academic writer agent...")
    
    # Initialize the academic writer agent
    academic_agent = AcademicWriterAgent()
    
    # Test with an academic paper generation request
    state = {
        "messages": [{"role": "user", "content": "Write an academic paper about quantum computing with APA citation style. Include figures and equations."}],
        "agent_outputs": {}
    }
    
    # Process the request
    updated_state = academic_agent(state)
    
    # Print the response
    print("\nAgent Response:")
    print(updated_state["messages"][-1]["content"])
    
    # Check if the academic paper was generated
    if "academic_generation" in updated_state["agent_outputs"]:
        academic_output = updated_state["agent_outputs"]["academic_generation"]
        
        print("\nAcademic Paper Generation Output:")
        print(f"Title: {academic_output.get('format', {}).get('title')}")
        print(f"Field of Study: {academic_output.get('format', {}).get('field_of_study')}")
        print(f"Citation Style: {academic_output.get('format', {}).get('citation_style')}")
        print(f"Word Count: {academic_output.get('word_count')}")
        
        # Check if the academic paper was saved
        file_path = academic_output.get("file_path")
        if file_path and os.path.exists(file_path):
            print(f"\nVerified: Academic paper file exists at {file_path}")
            
            # Print the first few lines of the academic paper
            with open(file_path, "r") as f:
                content = f.read(500)  # Read first 500 characters
                print("\nAcademic Paper Preview:")
                print(content + "...")
        else:
            print("\nWarning: Academic paper file does not exist or could not be verified")
    else:
        print("\nError: Academic paper generation output not found in agent outputs")
    
    return True


def test_proposal_writer_agent():
    """
    Test the proposal writer agent.
    """
    print("\nTesting proposal writer agent...")
    
    # Initialize the proposal writer agent
    proposal_agent = ProposalWriterAgent()
    
    # Test with a proposal generation request
    state = {
        "messages": [{"role": "user", "content": "Write a business proposal for a new software product for investors. Include budget and timeline."}],
        "agent_outputs": {}
    }
    
    # Process the request
    updated_state = proposal_agent(state)
    
    # Print the response
    print("\nAgent Response:")
    print(updated_state["messages"][-1]["content"])
    
    # Check if the proposal was generated
    if "proposal_generation" in updated_state["agent_outputs"]:
        proposal_output = updated_state["agent_outputs"]["proposal_generation"]
        
        print("\nProposal Generation Output:")
        print(f"Title: {proposal_output.get('format', {}).get('title')}")
        print(f"Proposal Type: {proposal_output.get('format', {}).get('proposal_type')}")
        print(f"Target Audience: {proposal_output.get('format', {}).get('target_audience')}")
        print(f"Word Count: {proposal_output.get('word_count')}")
        
        # Check if the proposal was saved
        file_path = proposal_output.get("file_path")
        if file_path and os.path.exists(file_path):
            print(f"\nVerified: Proposal file exists at {file_path}")
            
            # Print the first few lines of the proposal
            with open(file_path, "r") as f:
                content = f.read(500)  # Read first 500 characters
                print("\nProposal Preview:")
                print(content + "...")
        else:
            print("\nWarning: Proposal file does not exist or could not be verified")
    else:
        print("\nError: Proposal generation output not found in agent outputs")
    
    return True


def main():
    """Main function to run the tests."""
    # Load environment variables
    load_dotenv()
    
    # Create test directories
    os.makedirs("./generated_documents", exist_ok=True)
    os.makedirs("./generated_documents/reports", exist_ok=True)
    os.makedirs("./generated_documents/blogs", exist_ok=True)
    os.makedirs("./generated_documents/academic", exist_ok=True)
    os.makedirs("./generated_documents/proposals", exist_ok=True)
    
    # Run the tests
    test_base_document_agent()
    test_report_writer_agent()
    test_blog_writer_agent()
    test_academic_writer_agent()
    test_proposal_writer_agent()
    
    print("\nAll document generation agent tests completed!")


if __name__ == "__main__":
    main()
