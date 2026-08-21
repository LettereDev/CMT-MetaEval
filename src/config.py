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
    "M1_gemma": "gemma",
    "M2_quen": "quen",
    "M3_llama": "llama3.1 instruct",
    "M4_mistral": "mistral-7b-instruct-v0.1",
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