from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from src.state import SupportState

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

_SYSTEM = (
    "You are a customer support writer. "
    "Rewrite the last assistant message as a polite, professional support response. "
    "Keep the same facts. Be warm but concise."
)


def response_formatter(state: SupportState) -> dict:
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai is None:
        return {}

    formatted = _llm.invoke([SystemMessage(content=_SYSTEM), last_ai])
    return {"messages": [AIMessage(content=formatted.content)]}
