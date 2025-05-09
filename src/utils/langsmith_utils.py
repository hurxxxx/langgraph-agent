"""
LangSmith Integration Utilities

This module provides utilities for integrating LangSmith tracing and monitoring
with the multi-agent supervisor system.
"""

import os
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
import time
import json

# Import LangSmith components
try:
    from langsmith import Client
    from langsmith.run_trees import RunTree
    LANGSMITH_AVAILABLE = True
except ImportError:
    print("Warning: LangSmith components not available. Using mock implementations.")
    LANGSMITH_AVAILABLE = False
    # Mock implementations for testing
    class Client:
        def __init__(self, api_key=None, project_name=None):
            self.api_key = api_key
            self.project_name = project_name
            
        def create_run(self, name, run_type, inputs, **kwargs):
            print(f"[MOCK] Creating run: {name}, type: {run_type}")
            return MockRunTree(name=name, run_type=run_type, inputs=inputs)
            
        def update_run(self, run_id, **kwargs):
            print(f"[MOCK] Updating run: {run_id}, kwargs: {kwargs}")
            
        def create_feedback(self, run_id, key, score=None, value=None, comment=None):
            print(f"[MOCK] Creating feedback for run: {run_id}, key: {key}")
    
    class MockRunTree:
        def __init__(self, name, run_type, inputs):
            self.name = name
            self.run_type = run_type
            self.inputs = inputs
            self.id = f"mock-run-{int(time.time())}"
            
        def end(self, outputs=None, error=None):
            print(f"[MOCK] Ending run: {self.name}, outputs: {outputs}, error: {error}")
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                self.end(error=str(exc_val))
            else:
                self.end()


class LangSmithTracer:
    """
    Utility class for tracing agent execution with LangSmith.
    """
    
    def __init__(self, project_name=None):
        """
        Initialize the LangSmith tracer.
        
        Args:
            project_name: Name of the LangSmith project (defaults to LANGSMITH_PROJECT env var)
        """
        self.project_name = project_name or os.getenv("LANGSMITH_PROJECT", "langgraph-agent")
        
        if LANGSMITH_AVAILABLE:
            try:
                self.client = Client(
                    api_key=os.getenv("LANGSMITH_API_KEY"),
                    project_name=self.project_name
                )
                print(f"LangSmith tracer initialized for project: {self.project_name}")
            except Exception as e:
                print(f"Error initializing LangSmith client: {str(e)}")
                self.client = None
        else:
            print("Using mock LangSmith client")
            self.client = Client(
                api_key="mock-api-key",
                project_name=self.project_name
            )
    
    def trace_agent(self, agent_name: str):
        """
        Decorator for tracing agent execution.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(state, *args, **kwargs):
                if not self.client:
                    return func(state, *args, **kwargs)
                
                # Extract query from state
                query = state["messages"][-1]["content"] if state.get("messages") else "No query"
                
                # Create run
                with self.client.create_run(
                    name=f"{agent_name}",
                    run_type="agent",
                    inputs={
                        "query": query,
                        "state": json.dumps(state, default=str)
                    },
                    tags=[agent_name, "langgraph-agent"]
                ) as run:
                    try:
                        # Execute agent
                        result = func(state, *args, **kwargs)
                        
                        # Record output
                        if isinstance(result, dict):
                            run.end(
                                outputs={
                                    "result": json.dumps(result, default=str)
                                }
                            )
                        else:
                            run.end(
                                outputs={
                                    "result": str(result)
                                }
                            )
                        
                        return result
                    except Exception as e:
                        run.end(error=str(e))
                        raise
            
            return wrapper
        
        return decorator
    
    def trace_supervisor(self, supervisor_name: str):
        """
        Decorator for tracing supervisor execution.
        
        Args:
            supervisor_name: Name of the supervisor
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(query, *args, **kwargs):
                if not self.client:
                    return func(query, *args, **kwargs)
                
                # Create run
                with self.client.create_run(
                    name=f"{supervisor_name}",
                    run_type="chain",
                    inputs={
                        "query": query
                    },
                    tags=[supervisor_name, "langgraph-agent"]
                ) as run:
                    try:
                        # Execute supervisor
                        result = func(query, *args, **kwargs)
                        
                        # Record output
                        if isinstance(result, dict):
                            run.end(
                                outputs={
                                    "result": json.dumps(result, default=str)
                                }
                            )
                        else:
                            run.end(
                                outputs={
                                    "result": str(result)
                                }
                            )
                        
                        return result
                    except Exception as e:
                        run.end(error=str(e))
                        raise
            
            return wrapper
        
        return decorator


# Create a global instance for convenience
tracer = LangSmithTracer()
