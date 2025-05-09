# LangGraph Supervisor

This document describes the supervisor implementation in the LangGraph Agent system. The supervisor is a LangGraph that coordinates agents to solve complex tasks.

## Supervisor Architecture

The supervisor follows the LangGraph tutorial on multi-agent systems and implements a reactive workflow:

1. **Agent Selection**: Determine which agent(s) should handle the request
2. **Agent Execution**: Execute the selected agent(s)
3. **Result Processing**: Process the results from the agent(s)
4. **Response Synthesis**: Synthesize a final response for the user

The supervisor is implemented as a LangGraph with the following nodes:

```
┌─────────────────────────────────────────────────────────────┐
│                        Supervisor                           │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  Agent  │    │  Agent  │    │ Result  │    │Synthesis│  │
│  │Selection│───▶│Execution│───▶│ Process │───▶│  Node   │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Supervisor Configuration

The supervisor can be configured with the following options:

- `llm_provider`: LLM provider to use (default: "openai")
- `openai_model`: OpenAI model to use (default: "gpt-4o")
- `anthropic_model`: Anthropic model to use (default: "claude-3-7-sonnet-20250219")
- `temperature`: Temperature for the LLM (default: 0)
- `streaming`: Whether to stream the response (default: true)
- `system_message`: System message for the supervisor
- `parallel_execution`: Whether to execute agents in parallel (default: false)

## Supervisor Implementation

The supervisor is implemented using LangGraph's `StateGraph` class. It defines a state schema and a set of nodes that operate on the state.

### State Schema

The supervisor state schema includes:

- `messages`: List of messages in the conversation
- `agent_outputs`: Outputs from the agents
- `next_agent`: The next agent to execute
- `final_response`: The final response to the user

### Nodes

The supervisor has the following nodes:

#### Agent Selection Node

The agent selection node determines which agent(s) should handle the request. It analyzes the user's query and selects the most appropriate agent(s) based on the query content.

```python
def agent_selection(state):
    """
    Determine which agent to use based on the query.
    """
    query = state["messages"][-1]["content"]
    
    # Use LLM to determine the appropriate agent
    response = llm.invoke([
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Determine which agent should handle this query: {query}"}
    ])
    
    agent_name = extract_agent_name(response.content)
    state["next_agent"] = agent_name
    
    return state
```

#### Agent Execution Node

The agent execution node executes the selected agent(s). It calls the agent with the current state and updates the state with the agent's output.

```python
def agent_execution(state):
    """
    Execute the selected agent.
    """
    agent_name = state["next_agent"]
    agent = agents[agent_name]
    
    # Execute the agent
    updated_state = agent(state)
    
    # Update the state with the agent's output
    state["agent_outputs"][agent_name] = updated_state["agent_outputs"].get(agent_name, {})
    
    return state
```

#### Result Processing Node

The result processing node processes the results from the agent(s). It analyzes the agent outputs and determines if additional processing is needed.

```python
def result_processing(state):
    """
    Process the results from the agent(s).
    """
    # Check if we need to execute another agent
    if "next_agent" in state and state["next_agent"] is not None:
        return {"next": "agent_execution"}
    
    # Otherwise, proceed to synthesis
    return {"next": "synthesis"}
```

#### Synthesis Node

The synthesis node synthesizes a final response for the user based on the agent outputs.

```python
def synthesis(state):
    """
    Synthesize a final response for the user.
    """
    # Use LLM to synthesize a response
    response = llm.invoke([
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Synthesize a response based on the agent outputs: {state['agent_outputs']}"}
    ])
    
    state["final_response"] = response.content
    state["messages"].append({"role": "assistant", "content": response.content})
    
    return state
```

## Parallel Execution

The supervisor supports parallel execution of agents when appropriate. This is implemented using LangGraph's parallel execution capabilities.

```python
def parallel_agent_execution(state):
    """
    Execute multiple agents in parallel.
    """
    # Get the list of agents to execute
    agent_names = state["parallel_agents"]
    
    # Execute the agents in parallel
    results = {}
    for agent_name in agent_names:
        agent = agents[agent_name]
        agent_state = copy.deepcopy(state)
        results[agent_name] = agent(agent_state)
    
    # Merge the results
    for agent_name, result in results.items():
        state["agent_outputs"][agent_name] = result["agent_outputs"].get(agent_name, {})
    
    return state
```

## Streaming Support

The supervisor supports streaming responses at all levels. This is implemented using LangGraph's streaming capabilities.

```python
async def astream(self, query):
    """
    Process a user query using the multi-agent system with streaming.
    """
    # Initialize state
    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {},
        "next_agent": None,
        "final_response": None
    }
    
    # Stream the execution
    async for chunk in self.graph.astream(state):
        yield chunk
```

## Usage

The supervisor can be used as follows:

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

# Or with streaming
for chunk in supervisor.stream("Search for information about climate change and generate an image of a sustainable city"):
    print(chunk)
```
