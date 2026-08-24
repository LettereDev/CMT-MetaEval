# src/parser.py

import re


def parse_prediction(response):
    """
    Parse the final classification from an LLM response.

    Returns:
        1    = metaphorical
        0    = literal
        None = invalid/unparseable
    """

    if not response:
        return None

    response = response.strip().lower()

    # --------------------------------
    # Exact response
    # --------------------------------

    if response == "1":
        return 1

    if response == "0":
        return 0

    # --------------------------------
    # Look for explicit answer formats
    # --------------------------------

    patterns = [
        (r"answer\s*:\s*1\b", 1),
        (r"answer\s*:\s*0\b", 0),

        (r"classification\s*:\s*1\b", 1),
        (r"classification\s*:\s*0\b", 0),

        (r"^1\s*[-=:]", 1),
        (r"^0\s*[-=:]", 0),
    ]

    for pattern, prediction in patterns:
        if re.search(pattern, response):
            return prediction

    # --------------------------------
    # Last resort:
    # look at the first standalone 0/1
    # --------------------------------

    match = re.search(r"\b([01])\b", response)

    if match:
        return int(match.group(1))

    # --------------------------------
    # Invalid response
    # --------------------------------

    return None