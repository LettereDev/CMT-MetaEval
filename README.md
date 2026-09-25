# CMT-MetaEval
Testing whether explicit conceptual metaphor knowledge, injected via linguistically-informed prompting, improves the data efficiency of small open-weight LLMs at metaphor detection in German and Spanish.

## Repository layout

```
data/
  raw/            Raw input data (CoNLL-tagged CoMeta splits, COMETA/MIST CSVs)
  processed/      Everything preprocess.py and split.py produce (see below)
prompts/
  zero_shot/      PA_BasicPrompt.txt, PB_DefinitionPrompt.txt, PC_CMT_Basic.txt
  multi_shot/     Same three prompts, few-shot variants (contain {examples})
src/
  config.py       All experiment configuration in one place (see "Configuration" below)
  preprocess.py   Raw data -> data/processed/*.tsv
  split.py        Per-language support/query split -> data/processed/{DE,ES}_{support,query}.tsv
  datasets.py     Dataset loading/combining helpers, used by preprocess.py/split.py
  prompts.py      Prompt template loading and {examples}/{statement} substitution
  parser.py       Extracts a 0/1 label from a model's raw text response
  models.py       Model loading (4-bit quantized) / unloading
  inference.py    Runs one model+prompt+condition against the query set, writes predictions
  main.py         Orchestrates the full inference run, then calls evaluation
  evaluation.py   Predictions -> per-condition metrics -> all_metrics.tsv (and confusion matrices)
  graphs.py       all_metrics.tsv -> heatmap/bar chart figures 
  significance.py McNemar's test between two conditions, single or batch/table mode
results/
  predictions/<run_id>/<LANG>_<condition>/<prompt>_<model>.tsv   Raw predictions
  metrics/<run_id>/all_metrics.tsv                               Aggregated metrics
```

## Setup

- Python 3.10+, run as a package (`python -m src.<module>`) from the project root — the modules use relative imports, so `python src/main.py` will fail; always use `-m`.
- A CUDA-enabled PyTorch install, plus `transformers`, `bitsandbytes`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`. `bitsandbytes` on Windows in particular is worth double-checking against your installed CUDA/PyTorch versions, a mismatch can silently degrade performance rather than error clearly (see "Gotchas" below).
- Hugging Face access to any gated models in `config.py`'s `MODELS` (Gemma and Llama families require accepting their license on Hugging Face first). Log in via `huggingface-cli login` or set `HF_TOKEN`, before running `main.py`.

## Running the full pipeline

**Run these in order, each step depends on files the previous one wrote.**

### 1. Preprocess the raw data

```
python -m src.preprocess
```

Reads `data/raw/` and writes to `data/processed/`:
- `cometa_es.tsv`: the full sentence-level Spanish CoMeta dataset, built from the raw token-level CoNLL files
- `cometa.tsv`, `mist.tsv`: the German datasets, shuffled with a fixed seed
- `cometa_es_matched.tsv`: a stratified down-sample of `cometa_es.tsv`, sized to exactly match `|COMETA|+|MIST|`, preserving CoMeta's native metaphor/non-metaphor ratio rather than forcing it to the balanced 50/50 split present in the German data. **This is the file `config.py`'s `DATASETS["CoMeta"]` should point to** — the combined data efficiency comparison depends on DE and ES being the same size.

### 2. Build the support/query split

```
python -m src.split
```

For each language in `config.py`'s `LANGUAGES`, loads the full dataset and splits it (fixed seed, `SPLIT_CONFIG["split_seed"]`) into:
- `data/processed/{DE,ES}_support.tsv`:  the pool few-shot examples are drawn from
- `data/processed/{DE,ES}_query.tsv`:  the held-out set every model is actually evaluated on

**This step is required before running inference** `inference.py` reads these static files directly rather than re-splitting on the fly, so every model/prompt/condition is guaranteed to be evaluated against the exact same query set. If you change `SPLIT_CONFIG` or the underlying data, you must rerun this step before rerunning `main.py`, or the old split silently stays in effect.

### 3. Run inference + evaluation

```
python -m src.main
```

It loads each model in `config.py`'s `MODELS` one at a time (4-bit quantized), and for every (model × prompt × language × shot-condition) combination, generates predictions over the query set and writes them to `results/predictions/<run_id>/<LANG>_<condition>/<prompt>_<model>.tsv`. Once all predictions are written, it calls `evaluation.py` to produce `results/metrics/<run_id>/all_metrics.tsv`.

**Shot conditions**, built once per run by `inference.build_fewshot_examples()` and printed at the start of the run so you can see exactly which sentences were selected:
- `0shot` — no examples
- `2shot_DE_ES` / `2shot_ES_DE` — one metaphorical + one non-metaphorical example, cross-lingual, in each order
- `4shot` — a second, independent pair per language (no overlap with the 2-shot examples)

**`run_id`**: every run gets a timestamp (e.g. `20260916_131746`), printed at the start, used as the folder name under both `results/predictions/` and `results/metrics/`. This keeps separate runs from overwriting each other.

**Resuming an interrupted run**: pass the same `run_id` back in —

```
python -m src.main 20260916_131746
```

- Within a still-in-progress condition, a `.tsv.partial` checkpoint file is written every 10 rows; resuming picks up from there instead of redoing the whole condition. These `.partial` files are deleted automatically once their condition finishes, and are ignored by `evaluation.py`.
- Don't resume an old `run_id` after changing `config.py`, the prompt files, or the data. The skip-checks only look at whether a file exists, not whether it matches your current settings. Start a fresh run whenever something upstream has changed.

### 4. Generate charts

```
python -m src.graphs results/metrics/<run_id>/all_metrics.tsv [prompt] [language]
```

Both arguments are optional (defaulting to whichever prompt/language sorts first). For the chosen `language`, produces:
- **One combined heatmap image** — Model × Prompt, one subplot per shot condition found for that language, F1 score, shared colorbar. A model/prompt cell with zero valid predictions renders as gray "N/A", never as a colored zero.
- **One combined bar chart image** — Accuracy/Precision/Recall, grouped by model, one subplot per shot condition, for the chosen `prompt`. Same "N/A, not zero" handling for conditions with no valid predictions.
- **One invalid-responses chart** — horizontal, symlog-scaled bars, summed across every language/condition (not split per-condition like the two above), so one model's outlier invalid rate doesn't visually flatten every other bar to nothing.

Run it once per language to get both languages' figures (`... P1 DE`, then `... P1 ES`).

Order matters when writing the command, writing the language first (i.e. `... ES P1`) yields invalid images.

### 5. Statistical significance (McNemar's test)

Compares two conditions' predictions on the **same** query set (e.g. 0-shot vs. 4-shot) for one model/prompt/language, using only the discordant pairs (items the two conditions disagree on).

**Single comparison:**
```
python -m src.significance results/predictions/<run_id>/DE_0shot/P1_Gemma-3-1B.tsv results/predictions/<run_id>/DE_4shot/P1_Gemma-3-1B.tsv
```

**Batch mode — one table across every model/prompt combination for a fixed condition pair:**
```
python -m src.significance --batch --predictions-dir results/predictions/<run_id> --language DE --condition-a 0shot --condition-b 4shot --models "Gemma-3-1B,Qwen2.5-1.5B,Phi-3.5-mini,Llama-3.2-1B" --prompts "P1,P3" --invalid-as-incorrect --latex-out results/mcnemar_de.tex --tsv-out results/mcnemar_de.tsv
```

- `--invalid-as-incorrect` counts an unparseable response as a wrong answer; omitting it excludes that statement from the comparison instead. Given the invalid-rate findings are part of this project's actual results (some model/prompt combinations fail to answer >90% of the time), this flag was included to be able to evaluate these models with high invalid outputs in this test. For the paper it was decided to keep those responses as invalid for consisteency with the calculation of the other metrics (accuracy, precision, recall)
- A model/prompt combination with too few (or zero) mutually-valid predictions to compare will print a warning and either be skipped (batch mode) or reported with `n` visibly near zero (single mode) — always check `n` before trusting a p-value in this output, since a p-value computed on a handful of paired items (or none) is not meaningful evidence, even though the script will still compute one.
- The LaTeX table is generated with `\toprule`/`\midrule`/`\bottomrule` (`booktabs` style) — add `\usepackage{booktabs}` if your paper doesn't already.

## Output reference

**Prediction file columns** (`results/predictions/<run_id>/<LANG>_<condition>/<prompt>_<model>.tsv`):

| Column | Meaning |
|---|---|
| `statement` | The input sentence |
| `response` | The model's raw text output |
| `prediction` | Parsed 0/1 label, or blank if unparseable |
| `gold_label` | True label |
| `dataset` | Originating dataset (COMETA / MIST / CoMeta) |
| `shot_count` | 0, 2, or 4 |
| `condition` | e.g. `0shot`, `2shot_DE_ES`, `4shot` |

**Metrics file columns** (`results/metrics/<run_id>/all_metrics.tsv`), one row per (experiment, prompt, model): `experiment` (e.g. `DE_2shot_DE_ES`), `prompt`, `model`, `n`, `accuracy`, `precision`, `recall`, `f1`, `valid_count`, `invalid_count`, `invalid_rate`. Accuracy/precision/recall/F1 are computed over **valid predictions only**; a condition with zero valid predictions has these fields blank rather than zero — check `invalid_rate` before interpreting a blank or surprising score.

## Configuration (`src/config.py`)

| Field | Controls |
|---|---|
| `DATASETS` | Name -> processed file path for each underlying dataset |
| `LANGUAGES` | Which `DATASETS` entries make up each language's combined data |
| `PROMPTS` | Prompt name -> `{zero_shot, few_shot}` template file paths |
| `MODELS` | Display name -> Hugging Face model ID. **Verify this against your actual file** — see the note below |
| `GENERATION_CONFIG` | Passed to `model.generate()`; `max_new_tokens` in particular was tuned experimentally (see the paper's methodology/experiments section) |
| `SPLIT_CONFIG` | `support_fraction`, `split_seed` for `split.py` |
| `FEWSHOT_2SHOT_SEED`, `FEWSHOT_4SHOT_SEED` | Fixed seeds for the two independent few-shot draws (see step 3 above) |

> **Note:** over the course of this project, `MODELS` was changed from an initial set of larger models to four smaller ones to fit a 4GB-VRAM GPU, and `GENERATION_CONFIG["max_new_tokens"]` was changed more than once while diagnosing invalid-response rates. Rather than assume, open `config.py` and confirm both against whichever run you intend to treat as canonical before citing results from it.

## Gotchas worth knowing before a long run

- **GPU memory only frees correctly if the caller deletes its own references first** — `models.py`'s `unload_model()` cannot free memory on the caller's behalf (Python's `del` doesn't reach across function scopes); `main.py` already does this correctly (`del model, tokenizer` before calling `unload_model()`), so don't refactor that ordering without keeping this in mind.
- **A 4GB GPU cannot fit 7–8B models even in 4-bit** — if you see `accelerate`/`bitsandbytes` complaining about CPU/disk offload, it's very likely a genuine capacity ceiling, not a bug; check `nvidia-smi` for stray processes first, then consider smaller models.
- **`bitsandbytes` on Windows can silently fall back to slow or incorrect behavior** rather than erroring — if GPU utilization looks near-idle with high CPU usage during inference, check for `bitsandbytes`/CUDA setup warnings in the model-loading console output.
- **OneDrive-synced project folders can cause stale bytecode caches** — if an edited `.py` file doesn't seem to take effect, delete `src/__pycache__` and retry before assuming the edit didn't save.