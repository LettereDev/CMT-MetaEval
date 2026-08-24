DATASETS = {
    "COMETA": "data/processed/cometa.tsv",
    "MIST": "data/processed/mist.tsv",
    "CoMeta": "data/processed/cometa_es.tsv",
}

PROMPTS = {
    "P1": "prompts/PA_BasicPrompt.txt",
    "P2": "prompts/PB_DefinitionPrompt.txt",
    "P3": "prompts/PC_CMT_Basic.txt",
    "P4": "prompts/PD_CMT_Few-Shot.txt",
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

EXPERIMENTS = {
    "COMETA": ["COMETA"],
    "MIST": ["MIST"],
    "CoMeta": ["CoMeta"],
    
    "COMETA_MIST": [
        "COMETA", 
        "MIST"
    ],
    
    "COMETA_CoMeta": [
        "COMETA",
        "CoMeta"
    ],
    
    "ALL": [
        "COMETA",
        "MIST",
        "CoMeta"
    ],
}