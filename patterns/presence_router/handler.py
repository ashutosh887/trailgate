"""PresenceRouter: a Strands intervention that asks Decider whether an image is worth a model's time.

Before a guarded tool runs, the image argument is routed with one Decider call:
  skip      Deny: the tool never runs, the model is told why
  small     Transform: input[tier_arg] = "small"
  large     Transform: input[tier_arg] = "large"
  escalate  Transform: input[tier_arg] = "large" and input[escalate_arg] = True

Fails open: if Decider raises, the call is transformed to the large tier and the frame is
recorded in `routes` with `decider_error` set. An image rejected by `precheck` is denied and
recorded as a skip, so every guarded call shows up in `routes`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from strands.hooks.events import BeforeToolCallEvent
from strands.interventions import Deny, InterventionHandler, Proceed, Transform

from .router import PresenceRouterCore, Route, Tier


class PresenceRouter(InterventionHandler):
    """Routes image tool calls to skip, a small model, a large model, or a person."""

    name = "presence-router"
    on_error = "proceed"

    def __init__(
        self,
        core: PresenceRouterCore | None = None,
        tools: Iterable[str] = ("label_species", "describe_image"),
        image_arg: str = "image_path",
        tier_arg: str = "model_tier",
        escalate_arg: str = "escalate",
        tiers: dict[Tier, str] | None = None,
        precheck: Callable[[Path], str | None] | None = None,
        skip_reason: str = "empty frame, skipped by Decider (p={p:.2f})",
    ):
        self.core = core or PresenceRouterCore()
        self.tools = frozenset(tools)
        self.image_arg = image_arg
        self.tier_arg = tier_arg
        self.escalate_arg = escalate_arg
        self.tiers = tiers or {Tier.SMALL: "small", Tier.LARGE: "large", Tier.ESCALATE: "large"}
        self.precheck = precheck
        self.skip_reason = skip_reason
        self.routes: dict[str, Route] = {}
        self.last_route: Route | None = None

    def route(self, image: str | Path) -> Route:
        key = str(Path(image).resolve())
        cached = self.routes.get(key)
        if cached is None or cached.decider_error is not None:
            self.routes[key] = self.core.try_route(image)
        self.last_route = self.routes[key]
        return self.last_route

    def reject(self, image: str | Path, problem: str) -> Route:
        reason = f"code check: {problem}; Decider not asked"
        self.last_route = self.routes[str(Path(image).resolve())] = Route(Tier.SKIP, 0.0, None, "n/a", 0.0, reason, 0.0)
        return self.last_route

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any) -> Proceed | Deny | Transform:
        if event.tool_use["name"] not in self.tools:
            return Proceed()
        image = (event.tool_use.get("input") or {}).get(self.image_arg)
        if not image:
            return Proceed(reason=f"no {self.image_arg} argument")
        if self.precheck and (problem := self.precheck(Path(image))):
            self.reject(image, problem)
            return Deny(reason=f"unusable image, skipped before Decider: {problem}")
        route = self.route(image)
        if route.tier is Tier.SKIP:
            return Deny(reason=self.skip_reason.format(p=route.present))
        return Transform(apply=lambda e: self._apply(e, route), reason=route.reason)

    def _apply(self, event: BeforeToolCallEvent, route: Route) -> None:
        tool_input = event.tool_use.setdefault("input", {})
        if self._accepts(event, self.tier_arg):
            tool_input[self.tier_arg] = self.tiers[route.tier]
        if route.tier is Tier.ESCALATE and self._accepts(event, self.escalate_arg):
            tool_input[self.escalate_arg] = True

    @staticmethod
    def _accepts(event: BeforeToolCallEvent, arg: str) -> bool:
        if event.selected_tool is None:
            return False
        schema = event.selected_tool.tool_spec.get("inputSchema", {}).get("json", {})
        properties = schema.get("properties")
        return properties is None or arg in properties
