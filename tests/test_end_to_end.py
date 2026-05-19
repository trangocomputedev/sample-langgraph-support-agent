"""End-to-end integration tests using mocked LLM responses."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import graph


def _config(thread_id: str = "test-thread") -> dict:
    return {"configurable": {"thread_id": thread_id}}


@patch("src.nodes.guardrail._llm")
@patch("src.nodes.router._llm")
@patch("src.nodes.billing_agent._llm")
@patch("src.nodes.response_formatter._llm")
def test_billing_flow(mock_formatter, mock_billing, mock_router, mock_guardrail):
    mock_guardrail.invoke.return_value = MagicMock(content="SAFE")
    mock_router.invoke.return_value = MagicMock(intent="billing", reasoning="billing question")
    billing_response = AIMessage(content="Your current plan is Pro with a $0.00 balance.")
    billing_response.tool_calls = []
    mock_billing.invoke.return_value = billing_response
    mock_formatter.invoke.return_value = MagicMock(content="Thank you for contacting support. Your plan is Pro.")

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is my current plan?")]},
        config=_config("billing-test"),
    )

    assert result is not None
    assert len(result["messages"]) > 0


@patch("src.nodes.guardrail._llm")
def test_unsafe_input_ends_early(mock_guardrail):
    mock_guardrail.invoke.return_value = MagicMock(content="UNSAFE")

    result = graph.invoke(
        {"messages": [HumanMessage(content="do something bad")]},
        config=_config("unsafe-test"),
    )

    assert result["guardrail_passed"] is False
    last_msg = result["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert "can't help" in last_msg.content.lower()
