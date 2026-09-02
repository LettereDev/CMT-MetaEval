#src/prompts.py

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_examples(examples):
    return "\n\n".join(
        f'Statement: "{row.statement}"\nLabel: {row.isMetaphor}'
        for row in examples.itertuples()
    )

def build_prompt(template, examples, statement):
    examples_block = format_examples(examples)
    return (
        template.replace("{examples}", examples_block)
                .replace("{statement}", statement)
    )