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

# Import supervisor and agents
from supervisor.supervisor import Supervisor, SupervisorConfig
from agents.search_agent import SearchAgent, SearchAgentConfig
from agents.vector_storage_agent import VectorStorageAgent, VectorStorageAgentConfig
from agents.image_generation_agent import ImageGenerationAgent, ImageGenerationAgentConfig
from agents.quality_agent import QualityAgent, QualityAgentConfig


# Define API models
class QueryRequest(BaseModel):
    """Request model for querying the multi-agent system."""
    query: str
    stream: bool = False
    context: Optional[Dict[str, Any]] = None


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
            image_size="1024x1024"
        )
    )

    # Initialize quality agent
    quality_agent = QualityAgent(
        config=QualityAgentConfig()
    )

    # Return dictionary of agents
    return {
        "search_agent": search_agent,
        "vector_storage_agent": vector_storage_agent,
        "image_generation_agent": image_generation_agent,
        "quality_agent": quality_agent
    }


# Initialize agents
agents = initialize_agents()

# Initialize supervisor
supervisor = Supervisor(
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
        - image_generation_agent: For creating images based on descriptions
        - quality_agent: For evaluating the quality of responses

        Always think carefully about which agent(s) would be most appropriate for the task.
        You can use multiple agents in sequence or in parallel if needed.
        """
    ),
    agents=agents
)


# Define API endpoints
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
        if request.stream:
            # Return a streaming response
            return StreamingResponse(
                stream_response(request),
                media_type="text/event-stream"
            )
        else:
            # Return a standard response
            result = supervisor.invoke(
                query=request.query,
                stream=False
            )

            # Extract the final response
            final_message = result["messages"][-1]["content"] if result["messages"] else ""

            # Add context from the request if provided
            if request.context:
                result["context"] = request.context

            return QueryResponse(
                response=final_message,
                agent_trace=result.get("agent_outputs", {}),
                metadata={"stream": False, "query": request.query}
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
        # For our simplified supervisor, we'll simulate streaming
        # by yielding partial results
        state = {
            "messages": [{"role": "user", "content": request.query}],
            "agent_outputs": {},
            "human_input": None,
            "stream": True
        }

        # Determine which agent to use
        agent_name = supervisor._determine_next_agent(request.query)

        # Yield initial chunk
        yield f"data: {json.dumps({'messages': state['messages'], 'next_agent': agent_name})}\n\n"

        # If no agent is appropriate, generate a response directly
        if agent_name is None:
            # Simulate thinking
            time.sleep(1)

            # Generate response
            response = supervisor.llm.invoke([
                {"role": "system", "content": supervisor.config.system_message},
                {"role": "user", "content": request.query}
            ])

            # Update state
            state["messages"].append({"role": "assistant", "content": response.content})

            # Yield final chunk
            yield f"data: {json.dumps(state)}\n\n"
        else:
            # Call the agent
            agent = supervisor.agents[agent_name]

            # Simulate streaming by yielding intermediate state
            yield f"data: {json.dumps({'messages': state['messages'], 'next_agent': agent_name, 'thinking': True})}\n\n"

            # Process with agent
            updated_state = agent(state)

            # Yield final chunk
            yield f"data: {json.dumps(updated_state)}\n\n"
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
    List all available agents.

    Returns:
        Dict: List of available agents
    """
    return {"agents": list(agents.keys())}


# Run the application
if __name__ == "__main__":
    import uvicorn

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
