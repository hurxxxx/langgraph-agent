"""
Test Planning Document Agent

This script tests the planning document agent's ability to generate project plans and specifications.
"""

import os
import sys
import json
import time
import shutil
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import planning document agent
from src.agents.planning_document_agent import PlanningDocumentAgent, PlanningDocumentAgentConfig


def test_planning_document_agent():
    """
    Test the planning document agent.
    """
    print("Testing planning document agent...")
    
    # Initialize the planning document agent
    planning_agent = PlanningDocumentAgent(
        config=PlanningDocumentAgentConfig(
            plan_type="project",
            include_gantt_chart=True,
            include_resource_allocation=True,
            include_risk_assessment=True,
            include_success_metrics=True
        )
    )
    
    # Test with a planning document generation request
    state = {
        "messages": [{"role": "user", "content": "Create a project plan for developing a mobile app with user authentication, database integration, and push notifications. Include timeline, resource allocation, and risk assessment."}],
        "agent_outputs": {}
    }
    
    # Process the request
    updated_state = planning_agent(state)
    
    # Print the response
    print("\nAgent Response:")
    print(updated_state["messages"][-1]["content"])
    
    # Check if the planning document was generated
    if "planning_generation" in updated_state["agent_outputs"]:
        plan_output = updated_state["agent_outputs"]["planning_generation"]
        
        print("\nPlanning Document Generation Output:")
        print(f"Title: {plan_output.get('format', {}).get('title')}")
        print(f"Plan Type: {plan_output.get('format', {}).get('plan_type')}")
        print(f"Word Count: {plan_output.get('word_count')}")
        
        # Check if the planning document was saved
        file_path = plan_output.get("file_path")
        if file_path and os.path.exists(file_path):
            print(f"\nVerified: Planning document file exists at {file_path}")
            
            # Print the first few lines of the planning document
            with open(file_path, "r") as f:
                content = f.read(500)  # Read first 500 characters
                print("\nPlanning Document Preview:")
                print(content + "...")
        else:
            print("\nWarning: Planning document file does not exist or could not be verified")
    else:
        print("\nError: Planning document generation output not found in agent outputs")
    
    return True


def test_strategic_plan():
    """
    Test the planning document agent with a strategic plan.
    """
    print("\nTesting strategic plan generation...")
    
    # Initialize the planning document agent
    planning_agent = PlanningDocumentAgent(
        config=PlanningDocumentAgentConfig(
            plan_type="strategic",
            include_gantt_chart=False,
            include_resource_allocation=True,
            include_risk_assessment=True,
            include_success_metrics=True
        )
    )
    
    # Test with a strategic plan generation request
    state = {
        "messages": [{"role": "user", "content": "Create a 5-year strategic plan for a software company looking to expand into AI services. Include market analysis, resource needs, and success metrics."}],
        "agent_outputs": {}
    }
    
    # Process the request
    updated_state = planning_agent(state)
    
    # Print the response
    print("\nAgent Response:")
    print(updated_state["messages"][-1]["content"])
    
    # Check if the strategic plan was generated
    if "planning_generation" in updated_state["agent_outputs"]:
        plan_output = updated_state["agent_outputs"]["planning_generation"]
        
        print("\nStrategic Plan Generation Output:")
        print(f"Title: {plan_output.get('format', {}).get('title')}")
        print(f"Plan Type: {plan_output.get('format', {}).get('plan_type')}")
        print(f"Word Count: {plan_output.get('word_count')}")
        
        # Check if the strategic plan was saved
        file_path = plan_output.get("file_path")
        if file_path and os.path.exists(file_path):
            print(f"\nVerified: Strategic plan file exists at {file_path}")
            
            # Print the first few lines of the strategic plan
            with open(file_path, "r") as f:
                content = f.read(500)  # Read first 500 characters
                print("\nStrategic Plan Preview:")
                print(content + "...")
        else:
            print("\nWarning: Strategic plan file does not exist or could not be verified")
    else:
        print("\nError: Strategic plan generation output not found in agent outputs")
    
    return True


def test_implementation_plan():
    """
    Test the planning document agent with an implementation plan.
    """
    print("\nTesting implementation plan generation...")
    
    # Initialize the planning document agent
    planning_agent = PlanningDocumentAgent(
        config=PlanningDocumentAgentConfig(
            plan_type="implementation",
            include_gantt_chart=True,
            include_resource_allocation=True,
            include_risk_assessment=True,
            include_success_metrics=True
        )
    )
    
    # Test with an implementation plan generation request
    state = {
        "messages": [{"role": "user", "content": "Create an implementation plan for migrating a legacy system to a cloud-based architecture. Include timeline, resource allocation, and risk mitigation strategies."}],
        "agent_outputs": {}
    }
    
    # Process the request
    updated_state = planning_agent(state)
    
    # Print the response
    print("\nAgent Response:")
    print(updated_state["messages"][-1]["content"])
    
    # Check if the implementation plan was generated
    if "planning_generation" in updated_state["agent_outputs"]:
        plan_output = updated_state["agent_outputs"]["planning_generation"]
        
        print("\nImplementation Plan Generation Output:")
        print(f"Title: {plan_output.get('format', {}).get('title')}")
        print(f"Plan Type: {plan_output.get('format', {}).get('plan_type')}")
        print(f"Word Count: {plan_output.get('word_count')}")
        
        # Check if the implementation plan was saved
        file_path = plan_output.get("file_path")
        if file_path and os.path.exists(file_path):
            print(f"\nVerified: Implementation plan file exists at {file_path}")
            
            # Print the first few lines of the implementation plan
            with open(file_path, "r") as f:
                content = f.read(500)  # Read first 500 characters
                print("\nImplementation Plan Preview:")
                print(content + "...")
        else:
            print("\nWarning: Implementation plan file does not exist or could not be verified")
    else:
        print("\nError: Implementation plan generation output not found in agent outputs")
    
    return True


def main():
    """Main function to run the tests."""
    # Load environment variables
    load_dotenv()
    
    # Create test directories
    os.makedirs("./generated_documents", exist_ok=True)
    os.makedirs("./generated_documents/plans", exist_ok=True)
    os.makedirs("./generated_documents/plans/metadata", exist_ok=True)
    
    # Run the tests
    test_planning_document_agent()
    test_strategic_plan()
    test_implementation_plan()
    
    print("\nAll planning document agent tests completed!")


if __name__ == "__main__":
    main()
