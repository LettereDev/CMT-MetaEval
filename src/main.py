# src/main.py
'''
Main entry point for the inference and evaluation pipeline.
Sanity checks and evaluations were done with CODEX
Chat: https://chatgpt.com/s/cx_6a8eb8418d608191afd65b57b169e0f3
'''

from .config import MODELS, PROMPTS, LANGUAGES, SHOT_COUNTS, SUPPORT_SEEDS, SPLIT_CONFIG
from .models import load_model, unload_model
from .inference import run_experiment
from .evaluation import evaluate_all_predictions
from .prompts import load_prompt


def inference():

    # PROMPTS is nested: {"P1": {"zero_shot": path, "few_shot": path}, ...}
    # Load both templates for every prompt condition up front.
    prompts = {
        prompt_name: {
            shot_type: load_prompt(path)
            for shot_type, path in shot_paths.items()
        }
        for prompt_name, shot_paths in PROMPTS.items()
    }

    for model_name, model_id in MODELS.items():

        print("=" * 60)
        print(f"CURRENT MODEL: {model_name}")
        print("=" * 60)
        # Load ONLY this model
        tokenizer, model = load_model(model_id)
        try:

            for prompt_name, shot_templates in prompts.items():
                for lang in LANGUAGES:
                    for shot_count in SHOT_COUNTS:

                        shot_type = "zero_shot" if shot_count == 0 else "few_shot"
                        prompt_template = shot_templates[shot_type]

                        seeds = SUPPORT_SEEDS if shot_count > 0 else SUPPORT_SEEDS[:1]

                        for support_seed in seeds:
                            run_experiment(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt_template,
                                prompt_name=prompt_name,
                                model_name=model_name,
                                test_lang=lang,
                                shot_count=shot_count,
                                split_seed=SPLIT_CONFIG["split_seed"],
                                support_seed=support_seed,
                                output_dir="results/predictions"
                            )

        finally:
            # Make absolutely sure model is unloaded
            unload_model(
                model,
                tokenizer
            )


def main():
    inference()
    evaluate_all_predictions()


if __name__ == "__main__":
    main()