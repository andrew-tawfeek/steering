from __future__ import annotations

from .state import SteerItem, SteeringState


def describe_strength(strength: float) -> str:
    magnitude = abs(strength)
    direction = "toward" if strength >= 0 else "away from"

    if magnitude == 0:
        level = "neutral"
    elif magnitude < 10:
        level = "subtle"
    elif magnitude < 50:
        level = "noticeable"
    else:
        level = "strong"

    return f"{level} {direction}"


def item_to_directive(item: SteerItem) -> str:
    label = item.label or f"SAE feature {item.feature_id}"
    layers = ",".join(str(layer) for layer in item.layers)
    return (
        f"- {describe_strength(item.strength)} {label}; "
        f"feature_id={item.feature_id}, strength={item.strength:g}, layers={layers}"
    )


def build_steering_system_prompt(state: SteeringState) -> str:
    if state.is_empty:
        return ""

    directives = "\n".join(item_to_directive(item) for item in state.items)
    return (
        "You are running with an external steering state. "
        "Ollama does not expose residual-stream hooks, so interpret these as "
        "soft behavioral steering directives for the next response. Keep the "
        "answer coherent and obey the user's prompt.\n"
        f"{directives}"
    )


def apply_steering_to_prompt(prompt: str, state: SteeringState) -> str:
    system_prompt = build_steering_system_prompt(state)
    if not system_prompt:
        return prompt
    return f"{system_prompt}\n\nUser prompt:\n{prompt}"
