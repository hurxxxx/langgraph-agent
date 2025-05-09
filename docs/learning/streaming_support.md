# Streaming Support in LangGraph (as of May 2025)

This document outlines how to implement streaming responses in a multi-agent system built with LangGraph, covering both the LangGraph streaming capabilities and how to expose them through an API.

## Overview

Streaming responses are essential for providing real-time feedback to users, especially when working with LLMs that may take time to generate complete responses. LangGraph provides built-in support for streaming, allowing for:

1. Token-by-token streaming from LLMs
2. Incremental updates from agent actions
3. Real-time visibility into the agent's thought process
4. Improved user experience for long-running tasks

## LangGraph Streaming Basics

### Enabling Streaming

To enable streaming in LangGraph, you need to compile your graph with the `streaming=True` parameter:

```python
from langgraph.graph import StateGraph

# Define your graph
graph = StateGraph(...)

# Add nodes and edges
# ...

# Compile with streaming enabled
app = graph.compile(streaming=True)
```

### Consuming Streamed Output

Once streaming is enabled, you can consume the streamed output using the `stream` method:

```python
# Stream the output
for chunk in app.stream({"messages": [user_message]}):
    print(chunk)  # Process each chunk as it arrives
```

## Streaming with Different LLM Providers

### OpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    streaming=True  # Enable streaming
)
```

### Anthropic

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0,
    streaming=True  # Enable streaming
)
```

## Streaming in Multi-Agent Systems

In a multi-agent system, you can stream the outputs from each agent as they work:

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_agent: str

# Create graph
graph = StateGraph(AgentState)

# Add nodes for each agent
graph.add_node("agent_1", agent_1_function)
graph.add_node("agent_2", agent_2_function)

# Add edges
graph.add_conditional_edges(
    "agent_1",
    lambda state: "agent_2" if condition(state) else "output"
)
graph.add_edge("agent_2", "output")

# Set entry point
graph.set_entry_point("agent_1")

# Compile with streaming
app = graph.compile(streaming=True)

# Stream the output
for chunk in app.stream({"messages": [user_message], "current_agent": "agent_1"}):
    # The chunk will contain information about which agent is currently active
    # and the partial output from that agent
    print(f"Agent: {chunk['current_agent']}")
    print(f"Output: {chunk['messages'][-1]['content']}")
```

## Implementing Streaming in FastAPI

To expose streaming responses through a FastAPI API:

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    stream: bool = False
    context: Optional[Dict[str, Any]] = None

@app.post("/query")
async def process_query(request: QueryRequest):
    if request.stream:
        return StreamingResponse(
            stream_response(request),
            media_type="text/event-stream"
        )
    else:
        # Handle non-streaming request
        result = graph_app.invoke({
            "messages": [{"role": "user", "content": request.query}],
            "context": request.context or {}
        })
        return result

async def stream_response(request: QueryRequest):
    try:
        for chunk in graph_app_streaming.stream({
            "messages": [{"role": "user", "content": request.query}],
            "context": request.context or {}
        }):
            # Format as Server-Sent Events (SSE)
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"
```

## Client-Side Handling of Streamed Responses

### JavaScript Example

```javascript
const fetchStream = async (query) => {
  const response = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query,
      stream: true
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n\n');
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') {
          console.log('Stream complete');
          break;
        }
        
        try {
          const parsedData = JSON.parse(data);
          // Process the streamed chunk
          console.log('Received chunk:', parsedData);
          
          // Update UI with the latest content
          updateUI(parsedData);
        } catch (e) {
          console.error('Error parsing JSON:', e);
        }
      }
    }
  }
};

const updateUI = (chunk) => {
  // Example: Update a div with the latest content
  const outputDiv = document.getElementById('output');
  
  if (chunk.messages && chunk.messages.length > 0) {
    const latestMessage = chunk.messages[chunk.messages.length - 1];
    outputDiv.textContent = latestMessage.content;
  }
  
  // You could also update UI to show which agent is currently active
  if (chunk.current_agent) {
    const agentDiv = document.getElementById('current-agent');
    agentDiv.textContent = `Current agent: ${chunk.current_agent}`;
  }
};
```

### Python Example

```python
import requests
import json
import sseclient

def stream_query(query):
    response = requests.post(
        "http://localhost:8000/query",
        json={"query": query, "stream": True},
        stream=True,
        headers={"Accept": "text/event-stream"}
    )
    
    client = sseclient.SSEClient(response)
    
    for event in client.events():
        if event.data == "[DONE]":
            print("Stream complete")
            break
            
        try:
            data = json.loads(event.data)
            # Process the streamed chunk
            print(f"Received chunk: {data}")
            
            # Extract the latest message content
            if "messages" in data and data["messages"]:
                latest_message = data["messages"][-1]
                print(f"Latest content: {latest_message['content']}")
                
            # Show which agent is currently active
            if "current_agent" in data:
                print(f"Current agent: {data['current_agent']}")
                
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {event.data}")
```

## Advanced Streaming Techniques

### Progress Tracking

You can include progress information in your streamed responses:

```python
def agent_function(state):
    # ... agent logic ...
    
    # Include progress information in the state
    state["progress"] = {
        "percentage": 50,
        "status": "Processing data",
        "steps_completed": 2,
        "total_steps": 4
    }
    
    return state
```

### Partial Results

For long-running tasks, you can stream partial results as they become available:

```python
def search_agent(state):
    query = state["messages"][-1]["content"]
    results = []
    
    # Stream results as they come in
    for i, result in enumerate(search_api.search(query)):
        results.append(result)
        
        # Update state with partial results
        state["partial_results"] = results.copy()
        
        # Yield the updated state to stream it
        yield state
    
    # Final state update
    state["results"] = results
    return state
```

## Best Practices for Streaming

1. **Chunked Processing**: Break down tasks into smaller chunks that can be processed and streamed incrementally
2. **Progress Indicators**: Include progress information in streamed responses
3. **Error Handling**: Properly handle errors in both the server and client
4. **Timeouts**: Implement appropriate timeouts for long-running operations
5. **Backpressure**: Handle backpressure when the client can't process chunks as fast as they're generated
6. **Graceful Degradation**: Provide a non-streaming fallback for clients that don't support streaming
7. **Testing**: Test streaming endpoints with various network conditions and client implementations

## Resources

- [LangGraph Streaming Documentation](https://langchain-ai.github.io/langgraph/how-tos/streaming/)
- [FastAPI Streaming Response](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [Server-Sent Events (SSE) Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [JavaScript Fetch API with Streams](https://developer.mozilla.org/en-US/docs/Web/API/Streams_API/Using_readable_streams)
