# src/main.py
'''
Main entry point for the inference and evaluation pipeline.
Sanity checks and evaluations were done with CODEX
Chat: https://chatgpt.com/s/cx_6a8eb8418d608191afd65b57b169e0f3
'''

import os
import sys
from datetime import datetime

from .config import MODELS, PROMPTS, LANGUAGES, SHOT_COUNTS, SUPPORT_SEEDS
from .models import load_model, unload_model
from .inference import run_experiment
from .evaluation import evaluate_all_predictions
from .prompts import load_prompt


def inference(run_id):
    """
    Run every model / prompt / language / shot_count / support_seed
    combination, writing predictions under results/predictions/<run_id>/
    
    Individual conditions whose result file already exists under this
    run_id are skipped by run_experiment (see inference.run_experiment),
    so re-running with the same run_id resumes an interrupted run.
    """

    output_dir = os.path.join("results/predictions", run_id)

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

                        # Zero-shot draws no examples, so looping every
                        # support seed would just repeat an identical run.
                        # Run it once, using the first seed as a label.
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
                                support_seed=support_seed,
                                output_dir=output_dir
                            )

        finally:

            # Make absolutely sure model is unloaded
            unload_model(
                model,
                tokenizer
            )


def main(run_id=None):
    """
    run_id identifies one full pipeline execution. Leave it unset to start
    a fresh, timestamped run. Pass the run_id printed by a previous
    (interrupted) run to resume it in place instead of starting over.
    """

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Run ID: {run_id}")

    inference(run_id)

    evaluate_all_predictions(
        predictions_dir=os.path.join("results/predictions", run_id),
        output_path=os.path.join("results/metrics", run_id, "all_metrics.tsv"),
    )


if __name__ == "__main__":
    # Optionally resume a specific run: python -m src.main 20260902_153000
    cli_run_id = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_run_id)