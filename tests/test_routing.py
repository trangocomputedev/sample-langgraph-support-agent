"""Tests for the router node's intent classification."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from src.nodes.router import router_node


def _state(content: str) -> dict:
    return {
        "messages": [HumanMessage(content=content)],
        "intent": None,
        "customer_id": None,
        "ticket_id": None,
        "guardrail_passed": True,
        "escalation_reason": None,
    }


@pytest.mark.parametrize("intent", ["billing", "technical", "complaint", "escalate"])
@patch("src.nodes.router._llm")
def test_router_returns_valid_intent(mock_llm, intent):
    mock_llm.invoke.return_value = MagicMock(intent=intent, reasoning="test")
    result = router_node(_state("some message"))
    assert result["intent"] == intent


@patch("src.nodes.router._llm")
def test_billing_query_routed(mock_llm):
    mock_llm.invoke.return_value = MagicMock(intent="billing", reasoning="invoice question")
    result = router_node(_state("Why was I charged twice this month?"))
    assert result["intent"] == "billing"


@patch("src.nodes.router._llm")
def test_technical_query_routed(mock_llm):
    mock_llm.invoke.return_value = MagicMock(intent="technical", reasoning="how-to question")
    result = router_node(_state("How do I configure the API integration?"))
    assert result["intent"] == "technical"
