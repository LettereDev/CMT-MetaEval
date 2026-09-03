DATASETS = {
    "COMETA": "data/processed/cometa.tsv",
    "MIST": "data/processed/mist.tsv",
    "CoMeta": "data/processed/cometa_es_matched.tsv", #adjusted to match the size of COMETA + MIST
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

#Switched to smaller models due to hardware limitations
MODELS = {
    "Gemma-3-1B": "google/gemma-3-1b-it", 
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Phi-3.5": "microsoft/Phi-3.5-mini-instruct", #replaced mistral with phi-3.5-mini-instruct due to hardware limitations and lack of a stable release for mistralai/Ministral-3-3B-Instruct-2512
}

GENERATION_CONFIG = {
    "max_new_tokens": 5, #modified from 15 to improve performance
    "do_sample": False,
}

SPLIT_CONFIG = {
    "support_fraction": 0.20,
    "split_seed": 42,
}

SHOT_COUNTS = [0, 2, 4]
SUPPORT_SEEDS = [1, 2, 3]

LANGUAGES = {
    "DE": ["COMETA", "MIST"],
    "ES": ["CoMeta"],
}
