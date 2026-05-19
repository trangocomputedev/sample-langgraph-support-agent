from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from src.state import SupportState
from src.tools import knowledge_base_search, TavilySearchTool

_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([knowledge_base_search, TavilySearchTool])

_SYSTEM = (
    "You are a technical support specialist. "
    "Always try knowledge_base_search first. "
    "Fall back to TavilySearchResults only if the knowledge base has no answer. "
    "Provide step-by-step guidance."
)


def tech_agent(state: SupportState) -> dict:
    response = _llm.invoke([SystemMessage(content=_SYSTEM)] + state["messages"])
    return {"messages": [response]}
