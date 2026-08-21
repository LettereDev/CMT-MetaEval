

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(prompt, statement):
    return prompt.replace("{statement}", statement)