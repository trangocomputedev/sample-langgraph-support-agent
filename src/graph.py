from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.state import SupportState
from src.nodes import (
    guardrail_node,
    router_node,
    billing_agent,
    tech_agent,
    complaint_agent,
    human_node,
    response_formatter,
)
from src.tools import account_lookup, knowledge_base_search, create_ticket, TavilySearchTool

# ---------------------------------------------------------------------------
# Tool nodes — one per agent so the visualizer sees distinct tool boundaries
# ---------------------------------------------------------------------------
billing_tools = ToolNode([account_lookup])
tech_tools = ToolNode([knowledge_base_search, TavilySearchTool])
complaint_tools = ToolNode([create_ticket])

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
builder = StateGraph(SupportState)

builder.add_node("guardrail", guardrail_node)
builder.add_node("router", router_node)
builder.add_node("billing_agent", billing_agent)
builder.add_node("billing_tools", billing_tools)
builder.add_node("tech_agent", tech_agent)
builder.add_node("tech_tools", tech_tools)
builder.add_node("complaint_agent", complaint_agent)
builder.add_node("complaint_tools", complaint_tools)
builder.add_node("human_node", human_node)
builder.add_node("response_formatter", response_formatter)

# Entry point
builder.add_edge(START, "guardrail")

# Guardrail gate
builder.add_conditional_edges(
    "guardrail",
    lambda s: "router" if s["guardrail_passed"] else END,
)

# Intent routing
builder.add_conditional_edges(
    "router",
    lambda s: s["intent"],
    {
        "billing":   "billing_agent",
        "technical": "tech_agent",
        "complaint": "complaint_agent",
        "escalate":  "human_node",
    },
)

# Tool loops: agent → tools → agent (ReAct pattern)
builder.add_conditional_edges(
    "billing_agent",
    lambda s: "billing_tools" if s["messages"][-1].tool_calls else "response_formatter",
)
builder.add_edge("billing_tools", "billing_agent")

builder.add_conditional_edges(
    "tech_agent",
    lambda s: "tech_tools" if s["messages"][-1].tool_calls else "response_formatter",
)
builder.add_edge("tech_tools", "tech_agent")

builder.add_conditional_edges(
    "complaint_agent",
    lambda s: "complaint_tools" if s["messages"][-1].tool_calls else "response_formatter",
)
builder.add_edge("complaint_tools", "complaint_agent")

builder.add_edge("human_node", "response_formatter")
builder.add_edge("response_formatter", END)

# ---------------------------------------------------------------------------
# Compile with checkpointing and human-in-the-loop interrupt
# ---------------------------------------------------------------------------
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["human_node"])
