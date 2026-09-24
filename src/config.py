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

#Changed model back to reduced ones due to hardware limitations.
MODELS = {
    "Gemma-3-1B": "google/gemma-3-1b-it",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
}

GENERATION_CONFIG = {
    "max_new_tokens": 5, #Execution reduced to 5 tokens, with a slight modification to PC to view if models improve
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