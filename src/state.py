from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str | None
    customer_id: str | None
    ticket_id: str | None
    guardrail_passed: bool
    escalation_reason: str | None
