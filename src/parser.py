# src/parser.py
'''
Parse model predictions for metaphorical vs. literal classification.
Code was fixed with CODEX, Chat: https://chatgpt.com/s/cx_6a8ed8033ab88191b216f6771dcd9b40
'''

import re

_EXPLICIT_PREDICTION = re.compile(
    r"(?:answer|classification)\s*:\s*([01])\b|^([01])\s*[-=:]"
)
_STANDALONE_PREDICTION = re.compile(r"\b([01])\b")


def parse_prediction(response: str) -> int | None:
    """Return 1 for metaphorical, 0 for literal, or None if ambiguous/invalid."""
    if not isinstance(response, str):
        return None

    response = response.strip().lower()
    if response in {"0", "1"}:
        return int(response)

    explicit = _EXPLICIT_PREDICTION.findall(response)
    values = {next(value for value in match if value) for match in explicit}
    if len(values) == 1:
        return int(values.pop())
    if len(values) > 1:
        return None

    # Only use a bare label if it is the sole standalone 0/1 in the response.
    standalone = set(_STANDALONE_PREDICTION.findall(response))
    return int(standalone.pop()) if len(standalone) == 1 else None