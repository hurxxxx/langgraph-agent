# LangGraph Latest Information (as of May 2025)

## Overview
LangGraph is a library for building stateful, multi-actor applications with LLMs. It provides a framework for creating complex workflows and agent systems with clear control flow and state management.

## Latest Version
As of May 2025, the latest version of LangGraph is 0.3.x, which introduced prebuilt agents and enhanced supervisor capabilities.

## Key Features

- **Functional API**: More intuitive workflow definitions with better type checking
- **Streaming Support**: Real-time updates from agents and progressive rendering
- **Multi-Agent Architectures**: Supervisor, Tool-Calling, Peer-to-Peer, and Human-in-the-Loop

## Core Concepts

### Graphs
Graphs define the flow of information between nodes (agents or functions) in a LangGraph application.

```python
from langgraph.graph import StateGraph
from typing import TypedDict

# Define state
class State(TypedDict):
    messages: list[dict]
    current_agent: str

# Create graph
graph = StateGraph(State)
graph.add_node("agent_1", agent_1_function)
graph.add_node("agent_2", agent_2_function)
graph.add_edge("agent_1", "agent_2")
app = graph.compile()
```

### State Management
- Typed state definitions
- Immutable state updates
- State history tracking

### Prebuilt Agents
- Assistant agents
- Tool-using agents
- Supervisor agents

## Integration with LangChain
- LangChain tools can be used directly in LangGraph nodes
- LangChain chains can be incorporated as nodes
- LangChain retrievers can be used for RAG applications

## Best Practices

### Supervisor Implementation

```python
from langgraph.graph import StateGraph
from langgraph.prebuilt import SupervisorAgent
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    next_agent: str

# Create supervisor
supervisor = SupervisorAgent.from_llm(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    agents={"search": search_agent, "writer": writer_agent}
)

# Create graph
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor)
graph.add_node("search", search_agent)
graph.add_node("writer", writer_agent)

# Add conditional edges
graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"]
)

# Add edges back to supervisor
graph.add_edge("search", "supervisor")
graph.add_edge("writer", "supervisor")

# Set entry point and compile
graph.set_entry_point("supervisor")
app = graph.compile()
```

### Streaming Implementation

```python
# Create a streaming app
app_streaming = graph.compile(streaming=True)

# Use the app with streaming
for chunk in app_streaming.stream({"messages": [user_message]}):
    print(chunk)
```

## Resources
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [GitHub Repository](https://github.com/langchain-ai/langgraph)
