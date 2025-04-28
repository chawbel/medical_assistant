from typing import TypedDict

# output schema for the router LLM
class RoutingDecision(TypedDict):
    """Output structure for the main routing LLM"""

    supervisor: str  # name of the supervisor to route to
    reasoning: str  # Explanation of the decision
    confidence: float  # decision confidence