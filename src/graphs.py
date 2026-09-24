# src/graphs.py
"""
Builds the summary charts from a run's all_metrics.tsv (produced by
evaluation.evaluate_all_predictions):

  1. barplot_apr        -- Accuracy/Precision/Recall bar chart, grouped
                            by model, for one prompt AND one specific
                            experiment (language + shot condition).
  2. model_prompt_heatmap -- one metric (default F1) as a Model x Prompt
                            heatmap, for one specific experiment
                            (language + shot condition).
  3. invalid_responses_per_model_and_prompt -- vertical grouped bar
                            chart of invalid-response counts, summed
                            across every language/condition (not
                            split per-condition like the two above).

main() fixes one language and produces one heatmap + one APR chart for
EVERY shot condition found for that language (e.g. 0shot, 2shot_DE_ES,
2shot_ES_DE, 4shot -- 4 of each, not averaged together).

Usage (from the project root):
    python -m src.graphs results/metrics/<run_id>/all_metrics.tsv
    python -m src.graphs results/metrics/<run_id>/all_metrics.tsv P1
    python -m src.graphs results/metrics/<run_id>/all_metrics.tsv P1 DE

Arguments: metrics path, which prompt the APR charts use (defaults to
the first prompt found alphabetically), which language to fix on
(defaults to the first language found alphabetically).

Note: order matters, simply imputting the language (DE or ES) creates erroneous graphs
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(path="results/metrics/all_metrics.tsv"):
    """
    Loads all_metrics.tsv and splits the "experiment" column (e.g.
    "DE_2shot_DE_ES") into "language" and "condition", since the rest of
    this module treats those as separate dimensions.
    """
    metrics_df = pd.read_csv(path, sep="\t")
    split = metrics_df["experiment"].str.split("_", n=1, expand=True)
    metrics_df["language"] = split[0]
    metrics_df["condition"] = split[1]
    return metrics_df

#Added by Claude
def build_heatmap_matrix(metrics_df, metric="f1", experiment=None):
    """
    Builds a (models x prompts) matrix of one metric.

    If `experiment` is given (e.g. "DE_0shot"), only that exact
    condition is used. Otherwise, every language/shot-condition row for
    a given (model, prompt) is averaged together (skipping any rows
    with no valid predictions) -- a (model, prompt) cell only ends up
    NaN if every one of its experiments had zero valid predictions.
    """
    df = metrics_df if experiment is None else metrics_df[metrics_df["experiment"] == experiment]

    models = sorted(df["model"].unique())
    prompts = sorted(df["prompt"].unique())

    matrix = np.full((len(models), len(prompts)), np.nan)

    for i, model in enumerate(models):
        for j, prompt in enumerate(prompts):
            subset = df[(df["model"] == model) & (df["prompt"] == prompt)]
            if len(subset) == 0:
                continue
            # .mean(skipna=True) on an all-NaN subset correctly yields
            # NaN, which model_prompt_heatmap renders as "N/A".
            matrix[i, j] = subset[metric].mean(skipna=True)

    return matrix, models, prompts

#Added by Claude
def build_apr_by_model_language(metrics_df, prompt_name, experiment=None):
    """
    Builds the {group_label: {"accuracy":.., "precision":.., "recall":..}}
    dict that barplot_apr expects, for one prompt, with one group per
    (model, language) pair. If `experiment` is given, only that exact
    condition is used -- which also fixes the language, since experiment
    strings are "<language>_<condition>" -- and the "(language)" suffix
    is dropped from labels since it would be identical, and therefore
    redundant, on every bar. Without `experiment`, every shot-condition
    row for that (model, language, prompt) is averaged (skipping invalid
    rows), the same aggregation rule as build_heatmap_matrix, and labels
    keep the "(language)" suffix since multiple languages may be mixed.
    """
    df = metrics_df[metrics_df["prompt"] == prompt_name]
    if experiment is not None:
        df = df[df["experiment"] == experiment]

    show_language = df["language"].nunique() > 1

    apr_by_group = {}

    for (model, language), subset in df.groupby(["model", "language"]):
        group_label = f"{model} ({language})" if show_language else model
        apr_by_group[group_label] = {
            "accuracy": subset["accuracy"].mean(skipna=True),
            "precision": subset["precision"].mean(skipna=True),
            "recall": subset["recall"].mean(skipna=True),
        }

    return apr_by_group

#Added by Claude
def build_invalid_counts(metrics_df):
    """
    Builds the {model: {prompt: total_invalid_count}} dict that
    invalid_responses_per_model_and_prompt expects, summing
    invalid_count across every language/shot-condition for that
    (model, prompt) pair.
    """
    invalid_counts = {}

    for (model, prompt), subset in metrics_df.groupby(["model", "prompt"]):
        invalid_counts.setdefault(model, {})[prompt] = subset["invalid_count"].sum()

    return invalid_counts

#Fixed with Claude
def barplot_apr(apr_by_group, title='Accuracy, Precision, Recall', prompt_name="", ax=None):
    """
    Grouped bar chart of Accuracy / Precision / Recall.

    Parameters:
    - apr_by_group: dict mapping a group label (e.g. "Gemma-3-1B (DE)")
      to a dict with keys "accuracy", "precision", "recall". A missing
      or NaN value (no valid predictions for that group) is drawn as a
      zero-height bar labeled "N/A" -- so it reads as "no data
      produced", not as "the model scored zero".
    - title, prompt_name: chart title pieces.
    - ax: if given, draws into this existing axis instead of creating a
      new figure (used by apr_grid to combine several of these into one
      image) -- in that case no legend is added here and nothing is
      shown/returned; the caller handles both, using the returned bar
      handles to build one shared legend.
    """
    groups = list(apr_by_group.keys())
    metrics = ["accuracy", "precision", "recall"]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.6), 6))

    bar_width = 0.25
    index = np.arange(len(groups))
    bars_by_metric = {}

    for metric_index, metric in enumerate(metrics):
        values = [apr_by_group[group].get(metric) for group in groups]
        is_invalid = [pd.isna(value) for value in values]
        heights = [0.0 if invalid else value for value, invalid in zip(values, is_invalid)]

        bars = ax.bar(
            index + metric_index * bar_width,
            heights,
            bar_width,
            label=metric.capitalize(),
        )
        bars_by_metric[metric] = bars

        for bar, invalid in zip(bars, is_invalid):
            if invalid:
                # A hatch pattern is invisible on a zero-height bar --
                # there's no area to draw it on. Label it directly
                # instead, so "no valid predictions" is still visibly
                # distinct from "scored zero".
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.015,
                    "N/A",
                    ha='center', va='bottom',
                    color='dimgray', fontsize=7, rotation=90,
                )

    xlabel = 'Model (Language)' if any('(' in group for group in groups) else 'Model'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_title(f"{title} for: {prompt_name}" if prompt_name else title)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(groups, rotation=45, ha='right')

    if standalone:
        ax.legend()
        fig.tight_layout() #type: ignore[no-untyped-call]
        plt.show()
        return fig, ax #type: ignore[no-untyped-call]

    return bars_by_metric

#Fixed with Claude
def model_prompt_heatmap(scores, models, prompts, metric_name='F1', title=None, ax=None):
    """
    Heatmap of one metric across Model (rows) x Prompt (columns).

    Parameters:
    - scores: 2D array-like, shape (len(models), len(prompts)). Use
      None/np.nan for a cell with no valid predictions at all.
    - models, prompts: axis labels, in row/column order matching scores.
    - metric_name: used in the title/colorbar label.
    - title: overrides the default title if given.
    - ax: if given, draws into this existing axis instead of creating a
      new figure (used by heatmap_grid to combine several of these into
      one image) -- in that case no colorbar is added here and nothing
      is shown/returned; the caller handles both, using the returned
      image handle to build one shared colorbar.

    A cell with no valid predictions is rendered in light gray with an
    "N/A" label, instead of as a colored zero. Plotting it as 0 on a
    sequential colormap would make it visually indistinguishable from
    "the model scored genuinely badly" -- but those are different
    things (no output at all, vs. output that was simply wrong), and
    conflating them is misleading to read at a glance.
    """
    data = np.array(scores, dtype=float)
    masked = np.ma.masked_invalid(data)

    models = list(models)
    prompts = list(prompts)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=(max(6, len(prompts) * 1.6), max(4, len(models) * 0.9))
        )

    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad(color="lightgray")

    image = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    if standalone:
        fig.colorbar(image, ax=ax, label=f'{metric_name} Score') #type: ignore[no-untyped-call]

    ax.set_xticks(np.arange(len(prompts)), labels=prompts)
    ax.set_yticks(np.arange(len(models)), labels=models)

    for model_index in range(len(models)):
        for prompt_index in range(len(prompts)):
            value = data[model_index, prompt_index]
            if np.isnan(value):
                ax.text(
                    prompt_index, model_index, "N/A",
                    ha='center', va='center', color='dimgray', fontsize=9,
                )
            else:
                # White text reads better on dark (high-score) cells,
                # black on light (low-score) cells -- a fixed color
                # would go illegible at one end of the scale.
                text_color = "white" if value > 0.6 else "black"
                ax.text(
                    prompt_index, model_index, f"{value:.2f}",
                    ha='center', va='center', color=text_color,
                )

    ax.set_xlabel('Prompt')
    ax.set_ylabel('Model')
    ax.set_title(title or f"{metric_name} Heatmap: Model x Prompt")

    if standalone:
        fig.tight_layout() #type: ignore[no-untyped-call]
        plt.show()
        return fig, ax #type: ignore[no-untyped-call]

    return image


def invalid_responses_per_model_and_prompt(invalid_counts, models, prompts, title='Invalid Responses per Model and Prompt'):
    """
    Horizontal grouped bar chart of invalid-response counts, on a
    symmetric-log ("symlog") x-axis.

    Parameters:
    - invalid_counts: dict of dicts, invalid_counts[model][prompt] -> count
    - models, prompts: axis order / legend order.

    Symlog rather than plain linear: when one model/prompt combination
    (e.g. Llama on P3) has a far larger invalid count than everything
    else, a linear axis compresses every other bar down to a sliver
    next to it. Symlog (linear near zero, log beyond a small threshold)
    keeps small counts readable while still fitting a much larger one
    on the same chart -- and unlike plain log, it handles a count of
    exactly 0 without erroring or silently dropping the bar.
    """
    counts = np.zeros((len(models), len(prompts)))
    for i, model in enumerate(models):
        for j, prompt in enumerate(prompts):
            counts[i, j] = invalid_counts.get(model, {}).get(prompt, 0)

    fig, ax = plt.subplots(figsize=(9, max(4, len(models) * 1.2)))
    bar_height = 0.8 / max(len(prompts), 1)
    index = np.arange(len(models))

    for j, prompt in enumerate(prompts):
        ax.barh(index + j * bar_height, counts[:, j], bar_height, label=prompt)

    ax.set_xscale('symlog')
    ax.set_xlabel('Invalid Responses Count (symlog scale)')
    ax.set_ylabel('Models')
    ax.set_title(title)
    ax.set_yticks(index + bar_height * (len(prompts) - 1) / 2)
    ax.set_yticklabels(models)
    ax.invert_yaxis()  # first model reads top-to-bottom, not bottom-up
    ax.grid(axis='x', which='both', alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plt.show()
    return fig, ax

#Added by Claude
def _grid_shape(n):
    """2 columns for anything but a single plot, enough rows to fit n."""
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def heatmap_grid(metrics_df, language, metric="f1", metric_name=None, conditions=None):
    """
    Combines the Model x Prompt heatmap for every shot condition of one
    language into a single image -- one subplot per condition, sharing
    a single colorbar since every subplot is on the same 0-1 scale.
    """
    metric_name = metric_name or metric.upper()
    lang_df = metrics_df[metrics_df["language"] == language]
    conditions = list(conditions) if conditions is not None else sorted(lang_df["condition"].unique())

    nrows, ncols = _grid_shape(len(conditions))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False, constrained_layout=True)

    last_image = None
    for index, condition in enumerate(conditions):
        ax = axes[index // ncols][index % ncols]
        experiment = f"{language}_{condition}"

        matrix, models, prompts = build_heatmap_matrix(metrics_df, metric=metric, experiment=experiment)
        last_image = model_prompt_heatmap(
            matrix, models, prompts, metric_name=metric_name,
            title=experiment, ax=ax,
        )

    # Hide any unused grid cells (e.g. 3 conditions in a 2x2 grid)
    for index in range(len(conditions), nrows * ncols):
        axes[index // ncols][index % ncols].axis('off')

    if last_image is not None:
        fig.colorbar(last_image, ax=list(axes.flat), label=f'{metric_name} Score', shrink=0.8) #type: ignore[no-untyped-call]

    fig.suptitle(f"{metric_name} Heatmaps: Model x Prompt ({language})")
    plt.show()
    return fig, axes


def apr_grid(metrics_df, prompt_name, language, conditions=None):
    """
    Combines the Accuracy/Precision/Recall bar chart for every shot
    condition of one (prompt, language) pair into a single image -- one
    subplot per condition, sharing a single legend.
    """
    lang_df = metrics_df[metrics_df["language"] == language]
    conditions = list(conditions) if conditions is not None else sorted(lang_df["condition"].unique())

    nrows, ncols = _grid_shape(len(conditions))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5.5), squeeze=False, constrained_layout=True)

    last_bars = None
    for index, condition in enumerate(conditions):
        ax = axes[index // ncols][index % ncols]
        experiment = f"{language}_{condition}"

        apr_by_group = build_apr_by_model_language(metrics_df, prompt_name, experiment=experiment)
        last_bars = barplot_apr(apr_by_group, prompt_name=experiment, ax=ax)

    for index in range(len(conditions), nrows * ncols):
        axes[index // ncols][index % ncols].axis('off')

    if last_bars is not None:
        fig.legend(
            handles=list(last_bars.values()), #type: ignore[no-untyped-call]
            labels=[metric.capitalize() for metric in last_bars.keys()], #type: ignore[no-untyped-call]
            title=f"Accuracy, Precision, Recall ({prompt_name}, {language})",
            loc='outside upper center',
            ncol=3, 
        )

    plt.show()
    return fig, axes

def main(metrics_path="results/metrics/all_metrics.tsv", apr_prompt=None, language=None):
    """
    For one chosen `language` (DE or ES; defaults to whichever sorts
    first): produces ONE combined heatmap image (every shot condition
    found for that language as a subplot, e.g. 0shot, 2shot_DE_ES,
    2shot_ES_DE, 4shot), and ONE combined APR bar-chart image for
    `apr_prompt` (same per-condition subplots) -- each condition fixed
    exactly, not averaged together.

    The invalid-responses chart is unaffected by any of this -- it
    stays a single chart summed across every language/condition, since
    it's about overall response validity per model/prompt, not this
    per-condition comparison.
    """
    metrics_df = load_metrics(metrics_path)

    prompts = sorted(metrics_df["prompt"].unique())
    apr_prompt = apr_prompt or prompts[0]

    language = language or sorted(metrics_df["language"].unique())[0]
    lang_df = metrics_df[metrics_df["language"] == language]
    conditions = sorted(lang_df["condition"].unique())

    print(f"Language: {language} -- conditions found: {conditions}")

    # 1. All heatmaps for this language, combined into one image
    heatmap_grid(metrics_df, language, metric="f1", conditions=conditions)

    # 2. All APR bar charts for this (prompt, language), combined into one image
    apr_grid(metrics_df, apr_prompt, language, conditions=conditions)

    # Invalid responses per model and prompt -- stays a single,
    # standalone chart; not part of the per-condition grouping above.
    models = sorted(metrics_df["model"].unique())
    invalid_counts = build_invalid_counts(metrics_df)
    invalid_responses_per_model_and_prompt(invalid_counts, models, prompts)


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else "results/metrics/all_metrics.tsv"
    prompt_arg = sys.argv[2] if len(sys.argv) > 2 else None
    language_arg = sys.argv[3] if len(sys.argv) > 3 else None
    main(path_arg, prompt_arg, language_arg)