# Multi-Agent Supervisor System

A flexible, extensible multi-agent system built using LangGraph and OpenAPI. The system orchestrates various specialized agents to perform complex tasks, with a supervisor managing communication flow, task delegation, and decision-making.

## Features

- **Supervisor Architecture**: Central agent controls all communication and delegates tasks with task descriptions
- **Streaming Support**: Real-time updates from agents as they work
- **Specialized Agents**:
  - Search Agent (Serper, Tavily, Google, DuckDuckGo)
  - SQL RAG Agent (PostgreSQL, SQLite)
  - Vector Storage Agent (Chroma, Qdrant, Milvus, pgvector, Meilisearch)
  - Vector Retrieval Agent
  - Reranking Agent (Cohere, Pinecone)
  - Image Generation Agent (DALL-E, GPT-4o)
  - Writer Agent
  - MCP (Master Control Program) Agent
  - Quality Measurement Agent
- **Human-in-the-Loop**: Incorporates human feedback into the agent workflow
- **OpenAPI Integration**: Well-defined APIs for all components

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/langgraph-agent.git
   cd langgraph-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Usage

### Running the API Server

```bash
python src/app.py
```

This will start the FastAPI server on http://localhost:8000.

### API Endpoints

- **POST /query**: Process a query using the multi-agent system
  ```json
  {
    "query": "Tell me about climate change",
    "stream": false,
    "context": {}
  }
  ```

- **GET /health**: Health check endpoint
- **GET /agents**: List all available agents

### Streaming Example

```javascript
// Client-side JavaScript
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
          console.log('Received chunk:', parsedData);
        } catch (e) {
          console.error('Error parsing JSON:', e);
        }
      }
    }
  }
};

fetchStream("Tell me about climate change");
```

## Project Structure

```
langgraph-agent/
├── docs/
│   ├── learning/           # Documentation of APIs and libraries
│   │   ├── langgraph_latest.md
│   │   ├── specialized_agents.md
│   │   ├── openapi_integration.md
│   │   └── streaming_support.md
│   └── PRD.md              # Product Requirements Document
├── src/
│   ├── agents/             # Specialized agents
│   │   ├── search_agent.py
│   │   ├── vector_storage_agent.py
│   │   ├── image_generation_agent.py
│   │   └── quality_agent.py
│   ├── supervisor/         # Supervisor agent
│   │   └── supervisor.py
│   ├── utils/              # Utility functions
│   └── app.py              # Main application
├── .env                    # Environment variables
└── requirements.txt        # Dependencies
```

## Extending the System

### Adding a New Agent

1. Create a new agent file in `src/agents/`:
   ```python
   class NewAgentConfig(BaseModel):
       # Agent configuration
       pass

   class NewAgent:
       def __init__(self, config=NewAgentConfig()):
           self.config = config
           # Initialize agent

       def __call__(self, state):
           # Process state
           return updated_state
   ```

2. Register the agent in `src/app.py`:
   ```python
   # Initialize agents
   def initialize_agents():
       # ...
       new_agent = NewAgent()

       return {
           # ...
           "new_agent": new_agent
       }
   ```

## Documentation

For more detailed information, see the documentation in the `docs/` directory:

- [LangGraph Latest Information](docs/learning/langgraph_latest.md)
- [Specialized Agents](docs/learning/specialized_agents.md)
- [OpenAPI Integration](docs/learning/openapi_integration.md)
- [Streaming Support](docs/learning/streaming_support.md)
- [Serper API Integration](docs/learning/serper_integration.md)

## License

MIT
