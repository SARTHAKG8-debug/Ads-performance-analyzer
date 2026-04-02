"""
guardrails.py — Validate and sanitize user input before sending to the LLM.

Prevents:
  • Empty or too-short queries
  • Overly long queries (prompt injection surface)
  • Queries unrelated to ad/marketing data
  • Potentially harmful instructions
"""

import re

# Keywords that indicate a marketing / ad analytics question
VALID_TOPIC_KEYWORDS = {
    "campaign", "ad", "ads", "click", "clicks", "impression", "impressions",
    "ctr", "cost", "spend", "spending", "conversion", "conversions",
    "sale", "sales", "revenue", "roi", "device", "mobile", "desktop",
    "tablet", "keyword", "lead", "leads", "date", "time", "trend",
    "performance", "best", "worst", "highest", "lowest", "average",
    "total", "compare", "comparison", "which", "what", "how", "show",
    "list", "top", "bottom", "most", "least", "budget", "cpc",
    "data", "analytics", "hyderabad", "location",
}

# Patterns that suggest prompt injection or off-topic abuse
BLOCKED_PATTERNS = [
    r"ignore\s+(all\s+)?previous",
    r"forget\s+(all\s+)?instructions",
    r"you\s+are\s+now",
    r"act\s+as",
    r"pretend\s+to\s+be",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+prompt",
    r"write\s+(me\s+)?a?\s*(poem|story|essay|song|code)",
    r"translate\s+to",
]


def validate_query(question: str) -> tuple[bool, str]:
    """
    Check whether a query is valid for the ad analytics system.

    Parameters
    ----------
    question : str
        Raw user input.

    Returns
    -------
    (is_valid, message) : tuple[bool, str]
        is_valid  — True if the query can be forwarded to the LLM.
        message   — If invalid, a user-friendly explanation of why.
    """
    # Strip and basic length checks
    q = question.strip()

    if len(q) < 5:
        return False, "❌ Your question is too short. Please ask a more detailed question about your ad campaigns."

    if len(q) > 1000:
        return False, "❌ Your question is too long. Please keep it under 1,000 characters."

    # Check for prompt injection patterns
    q_lower = q.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "🚫 That query appears to contain instructions that aren't related to ad data analysis. Please ask a question about your campaigns."

    # Check topic relevance (at least one keyword should match)
    words = set(re.findall(r'\b\w+\b', q_lower))
    if not words.intersection(VALID_TOPIC_KEYWORDS):
        return False, (
            "🤔 That doesn't seem related to your Google Ads data. "
            "Try questions like:\n"
            '  • "Which campaigns have the highest CTR?"\n'
            '  • "What is the trend in spend over time?"\n'
            '  • "Compare device performance by conversions"'
        )

    return True, ""
