"""
Multi-Agent Supervisor System

This is the main application file that brings together all components of the
multi-agent supervisor system. It initializes the supervisor and specialized agents,
and provides an API for interacting with the system.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Iterator
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Load environment variables at the beginning
from dotenv import load_dotenv
load_dotenv()

# Import LangSmith utilities
from utils.langsmith_utils import tracer

# Import supervisor and agents
from supervisor.supervisor import Supervisor, SupervisorConfig
from supervisor.parallel_supervisor import ParallelSupervisor, ParallelSupervisorConfig
from agents.search_agent import SearchAgent, SearchAgentConfig
from agents.vector_storage_agent import VectorStorageAgent, VectorStorageAgentConfig
from agents.image_generation_agent import ImageGenerationAgent, ImageGenerationAgentConfig
from agents.quality_agent import QualityAgent, QualityAgentConfig
from agents.sql_rag_agent import SQLRAGAgent, SQLRAGAgentConfig
from agents.vector_retrieval_agent import VectorRetrievalAgent, VectorRetrievalAgentConfig
from agents.reranking_agent import RerankingAgent, RerankingAgentConfig

# Import document generation agents
from agents.document_generation import (
    BaseDocumentAgent, BaseDocumentAgentConfig,
    ReportWriterAgent, ReportWriterAgentConfig,
    BlogWriterAgent, BlogWriterAgentConfig
)
from agents.document_generation_part2 import (
    AcademicWriterAgent, AcademicWriterAgentConfig,
    ProposalWriterAgent, ProposalWriterAgentConfig
)
from agents.planning_document_agent import (
    PlanningDocumentAgent, PlanningDocumentAgentConfig
)


# Define API models
class QueryRequest(BaseModel):
    """Request model for querying the multi-agent system."""
    query: str
    stream: bool = False
    context: Optional[Dict[str, Any]] = None
    use_parallel: bool = True  # Whether to use the parallel supervisor


class QueryResponse(BaseModel):
    """Response model for the multi-agent system."""
    response: str
    agent_trace: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Supervisor API",
    description="API for interacting with a multi-agent system built with LangGraph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add static files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Mount static files directory if it exists
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "ui")
if os.path.exists(static_dir):
    app.mount("/ui", StaticFiles(directory=static_dir), name="ui")


# Initialize agents
def initialize_agents():
    """
    Initialize all specialized agents.

    Returns:
        Dict: Dictionary of agent functions keyed by agent name
    """
    # Initialize search agent with Serper
    search_agent = SearchAgent(
        config=SearchAgentConfig(
            provider="serper",  # Use Serper as the default search provider
            max_results=5
        )
    )

    # Initialize vector storage agent
    vector_storage_agent = VectorStorageAgent(
        config=VectorStorageAgentConfig(
            store_type="chroma",
            collection_name="default_collection",
            persist_directory="./vector_db"
        )
    )

    # Initialize image generation agent
    image_generation_agent = ImageGenerationAgent(
        config=ImageGenerationAgentConfig(
            provider="dalle",
            dalle_model="dall-e-3",
            image_size="1024x1024",
            save_images=True,
            images_dir="./generated_images",
            metadata_dir="./generated_images/metadata"
        )
    )

    # Initialize quality agent
    quality_agent = QualityAgent(
        config=QualityAgentConfig()
    )

    # Initialize SQL RAG agent
    sql_rag_agent = SQLRAGAgent(
        config=SQLRAGAgentConfig(
            db_type="postgresql",
            connection_string="postgresql://postgres:102938@localhost:5432/langgraph_agent_db",
            use_cache=True,
            cache_ttl=3600
        )
    )

    # Initialize vector retrieval agent
    vector_retrieval_agent = VectorRetrievalAgent(
        config=VectorRetrievalAgentConfig(
            store_type="chroma",
            collection_name="default_collection",
            persist_directory="./vector_db",
            embedding_model="text-embedding-3-small",
            use_cache=True,
            cache_ttl=3600
        )
    )

    # Initialize document generation agents
    report_writer_agent = ReportWriterAgent(
        config=ReportWriterAgentConfig(
            documents_dir="./generated_documents/reports",
            metadata_dir="./generated_documents/reports/metadata"
        )
    )

    blog_writer_agent = BlogWriterAgent(
        config=BlogWriterAgentConfig(
            documents_dir="./generated_documents/blogs",
            metadata_dir="./generated_documents/blogs/metadata"
        )
    )

    academic_writer_agent = AcademicWriterAgent(
        config=AcademicWriterAgentConfig(
            documents_dir="./generated_documents/academic",
            metadata_dir="./generated_documents/academic/metadata"
        )
    )

    proposal_writer_agent = ProposalWriterAgent(
        config=ProposalWriterAgentConfig(
            documents_dir="./generated_documents/proposals",
            metadata_dir="./generated_documents/proposals/metadata"
        )
    )

    # Initialize reranking agent
    reranking_agent = RerankingAgent(
        config=RerankingAgentConfig(
            provider="cohere",  # Use Cohere as the default reranking provider
            cohere_model="rerank-english-v3.0",
            cohere_top_n=5,
            use_cache=True,
            cache_ttl=3600
        )
    )

    # Initialize planning document agent
    planning_document_agent = PlanningDocumentAgent(
        config=PlanningDocumentAgentConfig(
            documents_dir="./generated_documents/plans",
            metadata_dir="./generated_documents/plans/metadata"
        )
    )

    # Return dictionary of agents
    return {
        "search_agent": search_agent,
        "vector_storage_agent": vector_storage_agent,
        "image_generation_agent": image_generation_agent,
        "quality_agent": quality_agent,
        "sql_rag_agent": sql_rag_agent,
        "vector_retrieval_agent": vector_retrieval_agent,
        "reranking_agent": reranking_agent,
        "report_writer_agent": report_writer_agent,
        "blog_writer_agent": blog_writer_agent,
        "academic_writer_agent": academic_writer_agent,
        "proposal_writer_agent": proposal_writer_agent,
        "planning_document_agent": planning_document_agent
    }


# Initialize agents
agents = initialize_agents()

# Initialize standard supervisor
standard_supervisor = Supervisor(
    config=SupervisorConfig(
        llm_provider="openai",
        openai_model="gpt-4o",
        streaming=True,
        system_message="""
        You are a supervisor agent that coordinates multiple specialized agents to solve complex tasks.
        Your job is to:
        1. Understand the user's request
        2. Determine which specialized agent(s) should handle the request
        3. Coordinate the flow of information between agents
        4. Synthesize a final response for the user

        You have access to the following specialized agents:
        - search_agent: For web searches and information retrieval using Serper (Google Search API)
        - vector_storage_agent: For storing documents in vector databases
        - vector_retrieval_agent: For retrieving documents from vector databases based on semantic similarity
        - reranking_agent: For reranking search results or vector search results based on relevance
        - image_generation_agent: For creating images based on descriptions
        - quality_agent: For evaluating the quality of responses
        - sql_rag_agent: For querying databases and providing insights from data
        - report_writer_agent: For generating formal reports with executive summaries and recommendations
        - blog_writer_agent: For creating blog posts and articles with different tones and styles
        - academic_writer_agent: For writing academic papers with proper citations and formatting
        - proposal_writer_agent: For creating business proposals with budgets and timelines
        - planning_document_agent: For creating project plans and specifications with timelines and resource allocation

        Always think carefully about which agent(s) would be most appropriate for the task.
        You can use multiple agents in sequence or in parallel if needed.
        """
    ),
    agents=agents
)

# Initialize parallel supervisor
parallel_supervisor = ParallelSupervisor(
    config=ParallelSupervisorConfig(
        llm_provider="openai",
        openai_model="gpt-4o",
        streaming=True,
        system_message="""
        You are a supervisor agent that coordinates multiple specialized agents to solve complex tasks.
        Your job is to:
        1. Understand the user's request
        2. Break down the request into subtasks
        3. Determine which specialized agent(s) should handle each subtask
        4. Coordinate the flow of information between agents
        5. Synthesize a final response for the user

        You have access to the following specialized agents:
        - search_agent: For web searches and information retrieval using Serper (Google Search API)
        - vector_storage_agent: For storing documents in vector databases
        - vector_retrieval_agent: For retrieving documents from vector databases based on semantic similarity
        - reranking_agent: For reranking search results or vector search results based on relevance
        - image_generation_agent: For creating images based on descriptions
        - quality_agent: For evaluating the quality of responses
        - sql_rag_agent: For querying databases and providing insights from data
        - report_writer_agent: For generating formal reports with executive summaries and recommendations
        - blog_writer_agent: For creating blog posts and articles with different tones and styles
        - academic_writer_agent: For writing academic papers with proper citations and formatting
        - proposal_writer_agent: For creating business proposals with budgets and timelines
        - planning_document_agent: For creating project plans and specifications with timelines and resource allocation

        You can run agents in parallel when appropriate to save time.
        """
    ),
    agents=agents
)

# Use parallel supervisor as the default
supervisor = parallel_supervisor


# Define API endpoints
@app.get("/")
async def get_monitor_ui():
    """
    Serve the agent monitor UI.

    Returns:
        FileResponse: The agent monitor HTML file
    """
    monitor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "ui", "agent_monitor.html")
    if os.path.exists(monitor_path):
        return FileResponse(monitor_path)
    else:
        return {"error": "Monitor UI not found"}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query using the multi-agent system.

    Args:
        request: Query request

    Returns:
        QueryResponse: Response from the multi-agent system
    """
    try:
        # Select the appropriate supervisor
        selected_supervisor = parallel_supervisor if request.use_parallel else standard_supervisor

        if request.stream:
            # Return a streaming response
            return StreamingResponse(
                stream_response(request),
                media_type="text/event-stream"
            )
        else:
            # Return a standard response
            result = selected_supervisor.invoke(
                query=request.query,
                stream=False
            )

            # Extract the final response
            if "final_response" in result and result["final_response"]:
                final_response = result["final_response"]
            elif "messages" in result and result["messages"]:
                final_response = result["messages"][-1]["content"]
            else:
                final_response = "No response generated."

            # Add context from the request if provided
            if request.context:
                result["context"] = request.context

            # Get LangSmith run ID if available
            langsmith_metadata = {
                "stream": False,
                "query": request.query,
                "supervisor_type": "parallel" if request.use_parallel else "standard"
            }

            if hasattr(tracer, "client") and tracer.client:
                langsmith_metadata.update({
                    "langsmith_project": tracer.project_name,
                    "langsmith_run_id": "latest"  # In a real implementation, we would capture the actual run ID
                })

            return QueryResponse(
                response=final_response,
                agent_trace=result.get("agent_outputs", {}),
                metadata=langsmith_metadata
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(request: QueryRequest) -> Iterator[str]:
    """
    Stream a response from the multi-agent system.

    Args:
        request: Query request

    Yields:
        str: Chunks of the response in SSE format
    """
    try:
        # Select the appropriate supervisor
        selected_supervisor = parallel_supervisor if request.use_parallel else standard_supervisor

        # For our simplified supervisor, we'll simulate streaming
        # by yielding partial results
        state = {
            "messages": [{"role": "user", "content": request.query}],
            "agent_outputs": {},
            "human_input": None,
            "stream": True
        }

        # For standard supervisor
        if not request.use_parallel:
            # Determine which agent to use
            agent_name = selected_supervisor._determine_next_agent(request.query)

            # Yield initial chunk
            yield f"data: {json.dumps({'messages': state['messages'], 'next_agent': agent_name, 'supervisor_type': 'standard'})}\n\n"

            # If no agent is appropriate, generate a response directly
            if agent_name is None:
                # Simulate thinking
                time.sleep(1)

                # Generate response
                response = selected_supervisor.llm.invoke([
                    {"role": "system", "content": selected_supervisor.config.system_message},
                    {"role": "user", "content": request.query}
                ])

                # Update state
                state["messages"].append({"role": "assistant", "content": response.content})

                # Yield final chunk
                yield f"data: {json.dumps(state)}\n\n"
            else:
                # Call the agent
                agent = selected_supervisor.agents[agent_name]

                # Simulate streaming by yielding intermediate state
                yield f"data: {json.dumps({'messages': state['messages'], 'next_agent': agent_name, 'thinking': True})}\n\n"

                # Process with agent
                updated_state = agent(state)

                # Yield final chunk
                yield f"data: {json.dumps(updated_state)}\n\n"
        else:
            # For parallel supervisor
            # Initialize state for parallel supervisor
            parallel_state = {
                "messages": [{"role": "user", "content": request.query}],
                "subtasks": [],
                "agent_outputs": {},
                "final_response": None
            }

            # Yield initial state
            yield f"data: {json.dumps({'messages': parallel_state['messages'], 'supervisor_type': 'parallel'})}\n\n"

            # Plan tasks
            subtasks = selected_supervisor._plan_tasks(request.query)
            parallel_state["subtasks"] = subtasks

            # Yield subtasks
            yield f"data: {json.dumps({'messages': parallel_state['messages'], 'subtasks': subtasks})}\n\n"

            # Process each subtask sequentially for streaming
            for subtask in subtasks:
                # Yield current subtask
                yield f"data: {json.dumps({'messages': parallel_state['messages'], 'current_subtask': subtask})}\n\n"

                # Process with agent
                agent_name = subtask["agent"]
                if agent_name in selected_supervisor.agents:
                    # Create a copy of the state for this agent
                    agent_state = {
                        "messages": parallel_state["messages"].copy(),
                        "agent_outputs": {},
                        "current_subtask": subtask,
                        "subtasks": []
                    }

                    # Execute the agent
                    agent = selected_supervisor.agents[agent_name]
                    updated_state = agent(agent_state)

                    # Extract the agent's output
                    if "agent_outputs" in updated_state:
                        parallel_state["agent_outputs"][agent_name] = updated_state["agent_outputs"].get(agent_name, {})

                    # Add any messages from the agent
                    if "messages" in updated_state and len(updated_state["messages"]) > len(parallel_state["messages"]):
                        for i in range(len(parallel_state["messages"]), len(updated_state["messages"])):
                            parallel_state["messages"].append(updated_state["messages"][i])

                # Yield updated state
                yield f"data: {json.dumps({'messages': parallel_state['messages'], 'agent_outputs': parallel_state['agent_outputs']})}\n\n"

            # Synthesize final response
            final_response = selected_supervisor._synthesize_response(request.query, parallel_state)
            parallel_state["final_response"] = final_response
            parallel_state["messages"].append({"role": "assistant", "content": final_response})

            # Add LangSmith metadata if available
            if hasattr(tracer, "client") and tracer.client:
                parallel_state["metadata"] = {
                    "langsmith_project": tracer.project_name,
                    "langsmith_run_id": "latest"  # In a real implementation, we would capture the actual run ID
                }

            # Yield final state
            yield f"data: {json.dumps(parallel_state)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Dict: Health status
    """
    return {"status": "healthy"}


@app.get("/agents")
async def list_agents():
    """
    List all available agents and supervisors.

    Returns:
        Dict: List of available agents and supervisors
    """
    return {
        "agents": list(agents.keys()),
        "supervisors": ["standard", "parallel"],
        "default_supervisor": "parallel"
    }


# Run the application
if __name__ == "__main__":
    import uvicorn

    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
