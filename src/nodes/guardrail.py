import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from src.state import SupportState

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_SYSTEM = (
    "You are a content safety classifier. "
    "Reply with exactly one word: SAFE or UNSAFE. "
    "UNSAFE means the message is harmful, abusive, or completely off-topic for a customer support context."
)


def guardrail_node(state: SupportState) -> dict:
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human is None:
        return {"guardrail_passed": True}

    response = _llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=last_human.content)])
    verdict = response.content.strip().upper()

    if verdict == "UNSAFE":
        refusal = AIMessage(
            content="I'm sorry, I can't help with that request. Please contact us with a support-related question."
        )
        return {"messages": [refusal], "guardrail_passed": False}

    return {"guardrail_passed": True}
