from langchain_core.tools import tool

_FAQ = {
    "reset password": "Go to Settings → Security → Reset Password and follow the email link.",
    "cancel subscription": "Go to Billing → Subscription → Cancel. Changes take effect at period end.",
    "download invoice": "Go to Billing → Invoices and click Download PDF next to any past invoice.",
    "two factor": "Go to Settings → Security → Two-Factor Authentication to enable TOTP or SMS.",
    "api key": "Go to Settings → API → Generate Key. Keep your key secret; it grants full account access.",
}


@tool
def knowledge_base_search(query: str) -> str:
    """Search internal knowledge base for support articles."""
    q = query.lower()
    for keyword, answer in _FAQ.items():
        if keyword in q:
            return answer
    return "No matching article found. Escalating to a human agent may be needed."
