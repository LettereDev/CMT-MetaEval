DATASETS = {
    "COMETA": "data/processed/cometa.tsv",
    "MIST": "data/processed/mist.tsv",
    "CoMeta": "data/processed/cometa_es_matched.tsv",
}

PROMPTS = {
    "P1": {
        "zero_shot": "prompts/zero_shot/PA_BasicPrompt.txt",
        "few_shot": "prompts/multi_shot/PA_BasicPrompt.txt",
    },
    "P2": {
        "zero_shot": "prompts/zero_shot/PB_DefinitionPrompt.txt",
        "few_shot": "prompts/multi_shot/PB_DefinitionPrompt.txt",
    },
    "P3": {
        "zero_shot": "prompts/zero_shot/PC_CMT_Basic.txt",
        "few_shot": "prompts/multi_shot/PC_CMT_Basic.txt",
    },
}

MODELS = {
    "Gemma-3-4B": "google/gemma-3-4b-it",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
}

GENERATION_CONFIG = {
    "max_new_tokens": 15,
    "do_sample": False,
}

SPLIT_CONFIG = {
    "support_fraction": 0.20,
    "split_seed": 42,
}

SHOT_COUNTS = [0, 2, 4]

# Seeds used to deterministically pick the fixed cross-lingual few-shot
# examples once (see inference.build_fewshot_examples). Two separate
# seeds keep the 4-shot examples independent of the 2-shot pairs, rather
# than being a combination of them.
FEWSHOT_2SHOT_SEED = 42
FEWSHOT_4SHOT_SEED = 43

LANGUAGES = {
    "DE": ["COMETA", "MIST"],
    "ES": ["CoMeta"],
}