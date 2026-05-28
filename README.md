# Sample LangGraph Support Agent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A multi-node customer support chatbot built with [LangGraph](https://github.com/langchain-ai/langgraph). Serves as a canonical example of LangGraph features and as a test fixture for the AI Workflow Visualizer.

## Workflow

```
START → [Guardrail] → [Router] ──billing──→ [Billing Agent] ──→ [Response Formatter] → END
                               ──technical→ [Tech Agent]    ──→ [Response Formatter] → END
                               ──complaint→ [Complaint Agent]──→ [Response Formatter] → END
                               ──escalate→ [Human Node (interrupt)] → [Response Formatter] → END
```

The **Guardrail** node rejects harmful or off-topic input before anything else runs. Each agent uses one or more `@tool`-decorated functions in a ReAct loop (agent → tools → agent) before handing off to the formatter.

## Features Demonstrated

| LangGraph Feature | Location |
|---|---|
| `StateGraph` + `TypedDict` state | `src/state.py` |
| `add_node` / `add_edge` | `src/graph.py` |
| `add_conditional_edges` | `src/graph.py` |
| `START` / `END` constants | `src/graph.py` |
| `@tool` decorated functions | `src/tools/` |
| `bind_tools` on `ChatOpenAI` | `src/nodes/*_agent.py` |
| `ToolNode` (ReAct loops) | `src/graph.py` |
| `MemorySaver` checkpointer | `src/graph.py` |
| `interrupt()` human-in-the-loop | `src/nodes/human_node.py` |
| Guardrail pattern | `src/nodes/guardrail.py` |

## Quickstart

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set credentials
cp .env.example .env
# Fill in OPENAI_API_KEY and TAVILY_API_KEY

# 3. Run a query
python - <<'EOF'
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from src.graph import graph

config = {"configurable": {"thread_id": "demo-1"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="Why was I charged twice this month?")]},
    config=config,
)
print(result["messages"][-1].content)
EOF
```

## Human-in-the-Loop Escalation

When the router classifies intent as `escalate`, the graph pauses at `human_node`:

```python
from langgraph.types import Command

# First call — graph pauses
graph.invoke({"messages": [HumanMessage(content="I need to speak to a human")]}, config=config)

# Human provides input — graph resumes
graph.invoke(Command(resume="I understand your frustration. Let me help you directly."), config=config)
```

## Tests

```bash
pytest tests/ -v
```

## Project Structure

```
src/
├── graph.py          # StateGraph assembly — the file the visualizer parses
├── state.py          # SupportState TypedDict
├── nodes/            # One file per graph node
└── tools/            # @tool-decorated functions
tests/
notebooks/
└── demo.ipynb        # Interactive walkthrough
```
