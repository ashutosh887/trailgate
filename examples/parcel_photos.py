"""PresenceRouter in a second agent: delivery photos without a parcel never reach the vision model.

  python -m examples.parcel_photos path/to/doorstep.jpg

The tool is called directly so no LLM is needed; a model-chosen call such as
agent("Check the delivery photo at ...") passes through the same router.
"""

from __future__ import annotations

import sys

from strands import Agent, tool

from patterns.presence_router import PresenceRouterCore
from patterns.presence_router.handler import PresenceRouter

router = PresenceRouter(
    PresenceRouterCore(target="a parcel or package", escalate=None, state="A proof-of-delivery photo taken at a doorstep."),
    tools={"describe_image"},
    skip_reason="no parcel in the photo, skipped by Decider (p={p:.2f})",
)


@tool
def describe_image(image_path: str, model_tier: str = "large") -> str:
    """Describe the delivered parcel and where it was left.

    Args:
        image_path: Path to the delivery photo.
        model_tier: "small" or "large"; set by the router.
    """
    return f"would describe {image_path} with the {model_tier} model"


if __name__ == "__main__":
    agent = Agent(tools=[describe_image], interventions=[router], callback_handler=None)
    result = agent.tool.describe_image(image_path=sys.argv[1], record_direct_tool_call=False)
    route = router.last_route
    print(result["status"], result["content"][0]["text"])
    if route:
        print(f"tier={route.tier.value} parcel p={route.present:.2f} difficulty={route.difficulty} ({route.latency_ms:.0f} ms)")
