# LangGraph Agent API

This document describes the API endpoints available in the LangGraph Agent system.

## API Overview

The API is implemented using FastAPI and provides endpoints for interacting with the multi-agent system.

## Base URL

The API is available at `http://localhost:8000` by default.

## Authentication

The API does not currently require authentication. However, it does require API keys for the various services used by the agents (OpenAI, Serper, etc.) to be set in the `.env` file.

## Endpoints

### Query Endpoint

The query endpoint processes a user query using the multi-agent system.

**URL**: `/query`

**Method**: `POST`

**Request Body**:
```json
{
  "query": "Tell me about climate change",
  "stream": false,
  "context": {}
}
```

**Parameters**:
- `query` (string, required): The user's query
- `stream` (boolean, optional): Whether to stream the response (default: false)
- `context` (object, optional): Additional context for the query

**Response**:
```json
{
  "response": "Climate change refers to long-term shifts in temperatures and weather patterns...",
  "agent_outputs": {
    "search_agent": {
      "result": {...},
      "query": "Tell me about climate change"
    }
  },
  "messages": [
    {"role": "user", "content": "Tell me about climate change"},
    {"role": "assistant", "content": "Climate change refers to long-term shifts in temperatures and weather patterns..."}
  ]
}
```

**Streaming Response**:

If `stream` is set to `true`, the response will be streamed as Server-Sent Events (SSE). Each event will contain a chunk of the response.

```
event: message
data: {"messages": [{"role": "assistant", "content": "Climate"}]}

event: message
data: {"messages": [{"role": "assistant", "content": "Climate change"}]}

event: message
data: {"messages": [{"role": "assistant", "content": "Climate change refers"}]}

...
```

### Health Check Endpoint

The health check endpoint returns the health status of the API.

**URL**: `/health`

**Method**: `GET`

**Response**:
```json
{
  "status": "healthy"
}
```

### Agents Endpoint

The agents endpoint returns a list of available agents.

**URL**: `/agents`

**Method**: `GET`

**Response**:
```json
{
  "agents": ["search_agent", "image_agent", "report_agent", "sql_agent", "vector_agent"],
  "supervisors": ["standard", "parallel"],
  "default_supervisor": "parallel"
}
```

### Direct Agent Endpoint

The direct agent endpoint allows direct interaction with a specific agent.

**URL**: `/agent/{agent_name}`

**Method**: `POST`

**URL Parameters**:
- `agent_name` (string, required): The name of the agent to use

**Request Body**:
```json
{
  "query": "What are the latest developments in AI?",
  "stream": false,
  "context": {}
}
```

**Parameters**:
- `query` (string, required): The user's query
- `stream` (boolean, optional): Whether to stream the response (default: false)
- `context` (object, optional): Additional context for the query

**Response**:
```json
{
  "response": "The latest developments in AI include...",
  "agent_outputs": {
    "search_agent": {
      "result": {...},
      "query": "What are the latest developments in AI?"
    }
  },
  "messages": [
    {"role": "user", "content": "What are the latest developments in AI?"},
    {"role": "assistant", "content": "The latest developments in AI include..."}
  ]
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages for different error scenarios:

- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include a message describing the error:

```json
{
  "detail": "Error message"
}
```

## Examples

### Basic Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about climate change", "stream": false}'
```

### Streaming Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "Tell me about climate change", "stream": true}'
```

### Direct Agent Query

```bash
curl -X POST "http://localhost:8000/agent/search_agent" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in AI?", "stream": false}'
```

## Client Libraries

### Python Client

```python
import requests

def query_api(query, stream=False):
    url = "http://localhost:8000/query"
    payload = {
        "query": query,
        "stream": stream
    }
    
    if not stream:
        response = requests.post(url, json=payload)
        return response.json()
    else:
        response = requests.post(url, json=payload, stream=True, headers={"Accept": "text/event-stream"})
        for line in response.iter_lines():
            if line:
                yield line.decode("utf-8")

# Example usage
result = query_api("Tell me about climate change")
print(result["response"])

# Streaming example
for chunk in query_api("Tell me about climate change", stream=True):
    print(chunk)
```

### JavaScript Client

```javascript
async function queryApi(query, stream = false) {
  const url = "http://localhost:8000/query";
  const payload = {
    query,
    stream
  };
  
  if (!stream) {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    return await response.json();
  } else {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
      },
      body: JSON.stringify(payload)
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      console.log(chunk);
    }
  }
}

// Example usage
queryApi("Tell me about climate change")
  .then(result => console.log(result.response));

// Streaming example
queryApi("Tell me about climate change", true);
```
