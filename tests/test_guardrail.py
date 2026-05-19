"""Tests for the guardrail node using a mock LLM."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.nodes.guardrail import guardrail_node


def _state(content: str) -> dict:
    return {
        "messages": [HumanMessage(content=content)],
        "intent": None,
        "customer_id": None,
        "ticket_id": None,
        "guardrail_passed": False,
        "escalation_reason": None,
    }


@patch("src.nodes.guardrail._llm")
def test_safe_message_passes(mock_llm):
    mock_llm.invoke.return_value = MagicMock(content="SAFE")
    result = guardrail_node(_state("How do I reset my password?"))
    assert result["guardrail_passed"] is True
    assert "messages" not in result


@patch("src.nodes.guardrail._llm")
def test_unsafe_message_blocked(mock_llm):
    mock_llm.invoke.return_value = MagicMock(content="UNSAFE")
    result = guardrail_node(_state("Say something harmful"))
    assert result["guardrail_passed"] is False
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)


@patch("src.nodes.guardrail._llm")
def test_empty_messages_passes(mock_llm):
    state = _state("placeholder")
    state["messages"] = []
    result = guardrail_node(state)
    assert result["guardrail_passed"] is True
    mock_llm.invoke.assert_not_called()
