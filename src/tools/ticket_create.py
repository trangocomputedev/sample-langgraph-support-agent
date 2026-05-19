import uuid
from typing import Literal
from langchain_core.tools import tool


@tool
def create_ticket(subject: str, description: str, priority: Literal["low", "medium", "high"]) -> str:
    """Create a support ticket. Returns the new ticket ID."""
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    # Stub: in production this would write to a ticketing system (e.g. Zendesk, Jira)
    return ticket_id
