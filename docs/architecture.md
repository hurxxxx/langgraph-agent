# LangGraph Agent Architecture

## Overview

This project implements a multi-agent system using LangGraph and LangChain. The architecture follows a three-tier structure:

1. **Tools**: Specific capabilities that perform discrete tasks
2. **Agents**: LangGraph constructs that use tools to solve domain-specific problems
3. **Supervisor**: A LangGraph that coordinates agents to solve complex tasks

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Supervisor                           │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  Agent  │    │  Agent  │    │  Agent  │    │  Agent  │  │
│  │Selection│    │Execution│    │ Result  │    │Synthesis│  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                         Agents                              │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Search │  │  Image  │  │ Report  │  │   SQL   │  ...   │
│  │  Agent  │  │  Agent  │  │  Agent  │  │  Agent  │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                         Tools                               │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Search │  │  Image  │  │Document │  │Vector DB│  ...   │
│  │  Tools  │  │  Tools  │  │  Tools  │  │  Tools  │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Tools

Tools are specific capabilities that perform discrete tasks. They are the building blocks of the system and are used by agents to accomplish their goals.

Examples of tools:
- Search tools (Serper, Tavily)
- Vector storage tools
- Image generation tools
- SQL database tools
- Document generation tools

Tools follow the LangChain Tool interface and can be used with LangGraph's `create_react_agent` function.

### Agents

Agents are LangGraph constructs that use tools to solve domain-specific problems. They are implemented using LangGraph's `create_react_agent` function, which creates a ReAct agent that can use tools to accomplish tasks.

Examples of agents:
- Search agent
- Image generation agent
- Report generation agent
- SQL query agent
- Vector storage agent

Each agent is a LangGraph that follows this pattern:
1. Analyze the user's request
2. Plan a sequence of tool calls
3. Execute the tools
4. Synthesize a response

### Supervisor

The supervisor is a LangGraph that coordinates agents to solve complex tasks. It follows the LangGraph tutorial on multi-agent systems and implements a reactive workflow:

1. **Agent Selection**: Determine which agent(s) should handle the request
2. **Agent Execution**: Execute the selected agent(s)
3. **Result Processing**: Process the results from the agent(s)
4. **Response Synthesis**: Synthesize a final response for the user

The supervisor can operate in two modes:
- **Sequential**: Execute agents one after another
- **Parallel**: Execute multiple agents in parallel when appropriate

## Data Flow

1. User submits a query to the API
2. The supervisor analyzes the query and selects appropriate agent(s)
3. The selected agent(s) use tools to accomplish their tasks
4. The supervisor synthesizes a final response based on the agent outputs
5. The API returns the response to the user

## Streaming Support

The system supports streaming responses at all levels:
- Tools can stream their outputs
- Agents can stream their reasoning and outputs
- The supervisor can stream the entire process

This allows for a responsive user experience with real-time updates.

## Configuration

The system is highly configurable through environment variables and configuration objects:
- Tool configurations (API keys, parameters)
- Agent configurations (LLM models, system messages)
- Supervisor configurations (execution mode, streaming)

## Extension

The system is designed to be easily extensible:
1. Add new tools by implementing the Tool interface
2. Add new agents by creating a new agent using `create_react_agent`
3. Update the supervisor to recognize and use the new agents
