from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from src.state import SupportState
from src.tools import create_ticket

_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([create_ticket])

_SYSTEM = (
    "You are a complaint-handling specialist. "
    "Acknowledge the customer's frustration empathetically, then use create_ticket to file a support ticket. "
    "Always communicate the ticket ID back to the customer."
)


def complaint_agent(state: SupportState) -> dict:
    response = _llm.invoke([SystemMessage(content=_SYSTEM)] + state["messages"])

    # Extract ticket ID from tool call result if present
    ticket_id = None
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc.get("name") == "create_ticket":
                ticket_id = tc.get("id")

    updates: dict = {"messages": [response]}
    if ticket_id:
        updates["ticket_id"] = ticket_id
    return updates
