"""Request-local call budgets, shared by vision, text and MCP adapters."""
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RequestBudget:
    seconds: float = 60
    max_model: int = 6
    max_mcp: int = 3
    started: float = field(default_factory=monotonic)
    model_calls: int = 0
    mcp_calls: int = 0

    def remaining(self):
        seconds = self.seconds - (monotonic() - self.started)
        if seconds <= 0:
            raise BudgetExceeded('Le délai de traitement est dépassé. Réessaie une demande plus courte.')
        return seconds

    def consume(self, kind):
        self.remaining()
        count = getattr(self, kind + '_calls')
        if count >= getattr(self, 'max_' + kind):
            raise BudgetExceeded('Le nombre maximal d’appels est atteint pour cette demande.')
        setattr(self, kind + '_calls', count + 1)


current_budget = ContextVar('narrativelens_budget', default=None)


def configured_budget():
    return RequestBudget(seconds=max(1, float(os.getenv('REQUEST_TIMEOUT_SECONDS', '60'))),
        max_model=max(1, int(os.getenv('MAX_MODEL_CALLS_PER_REQUEST', '6'))),
        max_mcp=max(1, int(os.getenv('MAX_MCP_CALLS_PER_REQUEST', '3'))))


def consume(kind):
    budget = current_budget.get()
    if budget:
        budget.consume(kind)
    return budget
