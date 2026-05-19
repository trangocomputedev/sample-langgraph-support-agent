from langgraph.types import interrupt
from src.state import SupportState


def human_node(state: SupportState) -> dict:
    """Pause execution and hand off to a human agent.

    Resume by calling: graph.invoke(Command(resume=human_response), config=config)
    """
    human_response = interrupt({"reason": state.get("escalation_reason", "Customer requested human assistance")})
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=human_response)]}
