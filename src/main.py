'''
Main entry point for the inference and evaluation pipeline.
Sanity checks and evaluations were done with CODEX
Chat: https://chatgpt.com/s/cx_6a8eb8418d608191afd65b57b169e0f3
'''

from .config import MODELS, PROMPTS, EXPERIMENTS
from .models import load_model, unload_model
from .inference import run_experiment
from .evaluation import evaluate_all_predictions
from .prompts import load_prompt

def inference_main():
    
    prompts = {
        prompt_name: load_prompt(path)
        for prompt_name, path in PROMPTS.items()
    }

    for model_name, model_id in MODELS.items():

        print("=" * 60)
        print(f"CURRENT MODEL: {model_name}")
        print("=" * 60)

        # Load ONLY this model
        tokenizer, model = load_model(model_id)

        try:

            # Run every configured experiment with every prompt
            for experiment_name in EXPERIMENTS:

                for prompt_name, prompt in prompts.items():

                    run_experiment(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        prompt_name=prompt_name,
                        model_name=model_name,
                        experiment_name=experiment_name,
                        output_dir="results/predictions"
                    )

        finally:

            # Make absolutely sure model is unloaded
            unload_model(
                model,
                tokenizer
            )

def main():
    inference_main()
    evaluate_all_predictions()


if __name__ == "__main__":
    main()
