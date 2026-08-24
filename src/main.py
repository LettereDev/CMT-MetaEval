from config import MODELS, PROMPTS, DATASETS, EXPERIMENTS
from models import load_model, unload_model
from inference import run_experiment
from evaluation import evaluate_all_predictions


def main():

    for model_name, model_id in MODELS.items():

        print("=" * 60)
        print(f"CURRENT MODEL: {model_name}")
        print("=" * 60)

        # Load ONLY this model
        tokenizer, model = load_model(model_id)

        try:

            # Run only the datasets configured for each experiment.
            for experiment_name, dataset_names in EXPERIMENTS.items():
                for dataset_name in dataset_names:
                    dataset = DATASETS[dataset_name]
                    for prompt_name, prompt in PROMPTS.items():
                        run_experiment(
                            model=model,
                            tokenizer=tokenizer,
                            dataset=dataset,
                            dataset_name=dataset_name,
                            prompt=prompt,
                            prompt_name=prompt_name,
                            model_name=model_name,
                            experiment_name=experiment_name,
                        )

        finally:

            # Make absolutely sure model is unloaded
            unload_model(
                model,
                tokenizer
            )


if __name__ == "__main__":

    main()

    evaluate_all_predictions()