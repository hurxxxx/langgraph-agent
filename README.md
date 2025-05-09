# LangGraph Agent

A multi-agent system built with LangGraph and LangChain that orchestrates specialized agents to solve complex tasks.

## Features

- **Multi-Agent Architecture**: Coordinate multiple specialized agents to solve complex tasks
- **LangGraph Integration**: Built on top of LangGraph for reactive agent workflows
- **Specialized Agents**: Search, image generation, report writing, SQL querying, and vector storage
- **Streaming Support**: Real-time streaming of agent responses
- **Parallel Processing**: Execute multiple agents in parallel when appropriate
- **API Integration**: FastAPI endpoints for easy integration
- **UI Interface**: Streamlit interface for testing and monitoring

## Architecture

The system follows a three-tier architecture:

1. **Tools**: Specific capabilities that perform discrete tasks
2. **Agents**: LangGraph constructs that use tools to solve domain-specific problems
3. **Supervisor**: A LangGraph that coordinates agents to solve complex tasks

For more details, see the [architecture documentation](docs/architecture.md).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hurxxxx/langgraph-agent.git
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

For more details, see the [API documentation](docs/api.md).

### Streaming Example

```python
import requests
import json
import sseclient

def stream_query(query):
    url = "http://localhost:8000/query"
    payload = {
        "query": query,
        "stream": True
    }

    response = requests.post(
        url,
        json=payload,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )

    client = sseclient.SSEClient(response)
    for event in client.events():
        chunk = json.loads(event.data)
        if "messages" in chunk and chunk["messages"]:
            message = chunk["messages"][-1]
            if "content" in message:
                print(message["content"], end="", flush=True)

# Example usage
stream_query("Tell me about climate change")
```

## Available Agents

### Search Agent

The Search Agent searches the web for information using search tools like Serper and Tavily.

```python
from src.agents.search_agent import SearchAgent, SearchAgentConfig

agent = SearchAgent(
    config=SearchAgentConfig(
        providers=["serper", "tavily"],
        max_results=5,
        time_period="1d",
        news_only=True
    )
)
```

### Image Generation Agent

The Image Generation Agent generates images from text descriptions using image generation tools like DALL-E and GPT-Image.

```python
from src.agents.image_agent import ImageGenerationAgent, ImageGenerationAgentConfig

agent = ImageGenerationAgent(
    config=ImageGenerationAgentConfig(
        provider="gpt-image",
        gpt_image_model="gpt-image-1",
        image_size="1024x1024",
        image_quality="high"
    )
)
```

For more details on available agents, see the [agents documentation](docs/agents.md).

## Supervisor

The supervisor coordinates agents to solve complex tasks. It follows the LangGraph tutorial on multi-agent systems and implements a reactive workflow.

```python
from src.supervisor.supervisor import Supervisor, SupervisorConfig
from src.agents.search_agent import SearchAgent
from src.agents.image_agent import ImageGenerationAgent

# Create agents
search_agent = SearchAgent()
image_agent = ImageGenerationAgent()

# Create supervisor
supervisor = Supervisor(
    config=SupervisorConfig(
        parallel_execution=True
    ),
    agents={
        "search_agent": search_agent,
        "image_agent": image_agent
    }
)

# Use the supervisor
result = supervisor.invoke("Search for information about climate change and generate an image of a sustainable city")
```

For more details on the supervisor, see the [supervisor documentation](docs/supervisor.md).

## Tools

The system includes various tools that agents can use to accomplish their tasks:

- **Search Tools**: Serper, Tavily
- **Image Generation Tools**: DALL-E, GPT-Image
- **Vector Storage Tools**: Chroma, PostgreSQL
- **SQL Database Tools**: PostgreSQL
- **Document Generation Tools**: Report, Blog Post, Proposal

For more details on available tools, see the [tools documentation](docs/tools.md).

## Configuration

The system is highly configurable through environment variables and configuration objects:

- **Tool Configurations**: API keys, parameters
- **Agent Configurations**: LLM models, system messages
- **Supervisor Configurations**: Execution mode, streaming

Example `.env` file:

```
# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key

# Search providers
SERPER_API_KEY=your-serper-api-key
TAVILY_API_KEY=your-tavily-api-key

# Vector storage
PGVECTOR_CONNECTION_STRING=postgresql://user:password@localhost:5432/vector_db

# SQL database
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/db

# LangSmith (optional)
LANGCHAIN_API_KEY=your-langchain-api-key
LANGCHAIN_PROJECT=your-langchain-project
```

## Development

### Project Structure

```
langgraph-agent/
├── docs/                      # Documentation
├── src/
│   ├── tools/                 # Tool implementations
│   ├── agents/                # Agent implementations
│   ├── supervisor/            # Supervisor implementation
│   ├── utils/                 # Utility functions
│   └── app.py                 # Main application

├── ui/                        # UI components
├── .env                       # Environment variables
└── requirements.txt           # Dependencies
```



### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the LangChain framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for the LangGraph framework
- [FastAPI](https://github.com/tiangolo/fastapi) for the API framework
- [Streamlit](https://github.com/streamlit/streamlit) for the UI framework
