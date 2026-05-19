from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from src.state import SupportState
from src.tools import account_lookup

_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([account_lookup])

_SYSTEM = (
    "You are a billing support specialist. "
    "Use the account_lookup tool when you need account details. "
    "Answer billing questions clearly and concisely."
)


def billing_agent(state: SupportState) -> dict:
    response = _llm.invoke([SystemMessage(content=_SYSTEM)] + state["messages"])
    return {"messages": [response]}
