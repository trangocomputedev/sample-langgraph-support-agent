from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from pydantic import BaseModel
from src.state import SupportState

_INTENTS = Literal["billing", "technical", "complaint", "escalate"]


class IntentOutput(BaseModel):
    intent: _INTENTS
    reasoning: str


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(IntentOutput)

_SYSTEM = (
    "Classify the customer's support request into exactly one intent:\n"
    "- billing: questions about invoices, payments, plans, or account charges\n"
    "- technical: product how-to questions, bugs, or configuration issues\n"
    "- complaint: expressing dissatisfaction; requires a ticket to be filed\n"
    "- escalate: requests to speak to a human or situations too complex to handle automatically\n"
    "Reply with the intent and a one-sentence reasoning."
)


def router_node(state: SupportState) -> dict:
    result: IntentOutput = _llm.invoke([SystemMessage(content=_SYSTEM)] + state["messages"])
    return {"intent": result.intent}
