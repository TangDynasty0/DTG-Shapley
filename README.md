# DTG-Shapley Supplementary Material

This package contains the public implementation, datasets, and complete raw
results associated with the DTG-Shapley paper. It exposes one algorithm,
`DTGShapleyEvaluator`; historical development-version names and unrelated
prototype algorithms have been removed.

## Directory layout

- `code/`: DTG-Shapley, paper baselines, value functions, experiment drivers,
  environment files, the simulated-data reproduction script, and a smoke test.
- `dataset/`: UCI source archives and deterministic generated weak fields.
  Human- and machine-readable simulated raw data are in
  `dataset/simulated_raw/`.
- `result/`: complete experiment outputs and the three manuscript figures.

## Environment

The paper used Python 3.10, NumPy 1.26.4, SciPy 1.15.2, and scikit-learn 1.7.2.
ReportLab 4.4.9 is included only to regenerate the manuscript figures.

```bash
conda env create -f code/environment.yml
conda activate dtg-shapley
```

The existing project environment can also be used:

```bash
conda run -n greedySHAP python code/tests/test_smoke.py
```

## Reproduce simulated raw data

Generate the CSV files used for the simulated cooperative games and Iris weak-
field experiment:

```bash
conda run -n greedySHAP python code/generate_simulated_datasets.py
```

## Reproduce experiments

Run the short end-to-end check first:

```bash
conda run -n greedySHAP python code/run_confirmatory_experiment.py --profile smoke --run-name public_smoke --max-hours 0.1
```

Run the full 259-case confirmatory experiment:

```bash
conda run -n greedySHAP python code/run_confirmatory_experiment.py --profile overnight --run-name dtg_confirmatory --max-hours 5
```

Run the structured-table HNB and MI experiment:

```bash
conda run -n greedySHAP python code/run_structured_text_experiment.py --profile overnight --run-name structured_text --max-hours 9 --value-quantifiers dataset_default mutual_information
```

All commands are resumable. New outputs are written to `reproduced_results/`
and do not overwrite the results supplied in `result/`.

Regenerate the three manuscript figures from the reported summary values:

```bash
conda run -n greedySHAP python code/make_paper_figures.py --output-dir result/paper_figures
```

## Public API

```python
from dtg_shapley import DTGShapleyEvaluator

evaluator = DTGShapleyEvaluator(
    data=members,
    value_function=value_function,
    n_samples=100,
    threshold_mode="relative",
    relative_threshold_factor=0.5,
    minimum_inclusion_probability=0.10,
    random_seed=11,
)
result = evaluator.estimate_shapley()
```

`value_function` must implement `ValueFunction.evaluate(subset)`. Diagnostics
include actual value-function calls, per-member observation counts, dynamic
group history, inclusion probabilities, audit outcomes, and fallback mode.

## License and third-party data

The source code in `code/` is released under the MIT License. UCI datasets are
third-party material distributed under CC BY 4.0 and are not covered by the
code license. See `LICENSES.md` and `dataset/README.md` for attribution, DOI,
and reuse details.
