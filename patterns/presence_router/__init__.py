from .questions import build_questions
from .router import DeciderClient, PresenceRouterCore, Route, Thresholds, Tier, route

__all__ = ["DeciderClient", "PresenceRouter", "PresenceRouterCore", "Route", "Thresholds", "Tier", "build_questions", "route"]


def __getattr__(name: str):
    if name == "PresenceRouter":
        from .handler import PresenceRouter

        return PresenceRouter
    raise AttributeError(name)
