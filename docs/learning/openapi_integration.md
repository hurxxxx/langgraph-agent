# OpenAPI Integration with LangGraph (as of May 2025)

This document outlines how to integrate OpenAPI specifications with LangGraph to create a robust multi-agent system with well-defined APIs.

## Overview

OpenAPI (formerly known as Swagger) is a specification for machine-readable interface files for describing, producing, consuming, and visualizing RESTful web services. Integrating OpenAPI with LangGraph allows for:

1. Standardized API definitions for agent interactions
2. Automatic validation of inputs and outputs
3. Interactive API documentation
4. Client code generation for various languages
5. Easier testing and debugging

## FastAPI Integration

FastAPI is a modern, fast web framework for building APIs with Python that natively supports OpenAPI. It's an excellent choice for exposing LangGraph agents as web services.

### Basic Setup

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph

# Create FastAPI app
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

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    stream: bool = False
    additional_context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    response: str
    agent_trace: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# Create LangGraph application (defined elsewhere)
graph = StateGraph(...)
graph_app = graph.compile()
graph_app_streaming = graph.compile(streaming=True)

# Define API endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        if request.stream:
            # Return a streaming response
            return StreamingResponse(
                stream_response(request),
                media_type="text/event-stream"
            )
        else:
            # Return a standard response
            result = graph_app.invoke({
                "messages": [{"role": "user", "content": request.query}],
                "additional_context": request.additional_context or {}
            })
            
            return QueryResponse(
                response=result["messages"][-1]["content"],
                agent_trace=result.get("agent_trace", []),
                metadata=result.get("metadata", {})
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streaming response handler
async def stream_response(request: QueryRequest):
    try:
        for chunk in graph_app_streaming.stream({
            "messages": [{"role": "user", "content": request.query}],
            "additional_context": request.additional_context or {}
        }):
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"
```

## Agent API Definitions

Each specialized agent should have a well-defined API using Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

# Base agent input/output
class AgentInput(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class AgentOutput(BaseModel):
    response: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = {}

# Search agent
class SearchAgentInput(AgentInput):
    search_provider: str = "google"
    num_results: int = 5
    
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    
class SearchAgentOutput(AgentOutput):
    results: List[SearchResult]

# SQL RAG agent
class SQLRAGAgentInput(AgentInput):
    database_type: str
    query_type: str = "natural_language"
    tables: Optional[List[str]] = None
    
class SQLRAGAgentOutput(AgentOutput):
    sql_query: str
    results: List[Dict[str, Any]]
    
# Vector storage agent
class VectorStorageAgentInput(AgentInput):
    vector_store: str
    operation: str  # "store", "update", "delete"
    documents: Optional[List[Dict[str, Any]]] = None
    document_ids: Optional[List[str]] = None
    
class VectorStorageAgentOutput(AgentOutput):
    operation_status: str
    document_ids: List[str]
```

## Supervisor API

The supervisor API should be designed to handle task delegation and orchestration:

```python
class SupervisorInput(BaseModel):
    query: str
    stream: bool = False
    context: Optional[Dict[str, Any]] = None
    agents_to_use: Optional[List[str]] = None  # If None, supervisor decides

class AgentTask(BaseModel):
    agent_id: str
    input: Dict[str, Any]
    status: str  # "pending", "in_progress", "completed", "failed"
    output: Optional[Dict[str, Any]] = None
    
class SupervisorState(BaseModel):
    original_query: str
    tasks: List[AgentTask]
    current_agent: Optional[str] = None
    final_response: Optional[str] = None
    
class SupervisorOutput(BaseModel):
    response: str
    tasks: List[AgentTask]
    metadata: Dict[str, Any] = {}
```

## OpenAPI Schema Generation

FastAPI automatically generates OpenAPI schemas, which can be accessed at `/docs` (Swagger UI) or `/redoc` (ReDoc) endpoints. You can also export the schema to a file:

```python
import json

with open("openapi_schema.json", "w") as f:
    json.dump(app.openapi(), f)
```

## Client Generation

Using the OpenAPI schema, you can generate client libraries in various languages:

```bash
# Generate Python client
openapi-generator generate -i openapi_schema.json -g python -o ./clients/python

# Generate JavaScript client
openapi-generator generate -i openapi_schema.json -g javascript -o ./clients/javascript

# Generate Java client
openapi-generator generate -i openapi_schema.json -g java -o ./clients/java
```

## Testing with OpenAPI

The OpenAPI schema can be used to validate requests and responses during testing:

```python
from openapi_core import OpenAPISpec
from openapi_core.validation.request.validators import RequestValidator
from openapi_core.validation.response.validators import ResponseValidator
import yaml

# Load OpenAPI spec
with open("openapi_schema.yaml", "r") as f:
    spec_dict = yaml.safe_load(f)
    
spec = OpenAPISpec.from_dict(spec_dict)
request_validator = RequestValidator(spec)
response_validator = ResponseValidator(spec)

# Validate request
request = Request("POST", "/query", {"Content-Type": "application/json"}, json.dumps({"query": "test"}))
result = request_validator.validate(request)
assert result.errors == []

# Validate response
response = Response(200, {"Content-Type": "application/json"}, json.dumps({"response": "test response"}))
result = response_validator.validate(request, response)
assert result.errors == []
```

## Best Practices

1. **Versioning**: Include version information in your API paths (e.g., `/v1/query`)
2. **Documentation**: Add detailed descriptions to all models and endpoints
3. **Validation**: Use Pydantic validators for complex validation logic
4. **Error Handling**: Define standard error responses and codes
5. **Rate Limiting**: Implement rate limiting for production APIs
6. **Authentication**: Add proper authentication mechanisms
7. **Monitoring**: Include endpoints for health checks and monitoring

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAPI Specification](https://spec.openapis.org/oas/latest.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAPI Generator](https://openapi-generator.tech/)
