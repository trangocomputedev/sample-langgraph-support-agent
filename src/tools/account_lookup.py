from langchain_core.tools import tool


@tool
def account_lookup(customer_id: str) -> dict:
    """Look up account details by customer ID."""
    # Stub: in production this would hit a SQL/CRM query
    return {"name": "Jane Doe", "plan": "Pro", "balance": 0.0}
