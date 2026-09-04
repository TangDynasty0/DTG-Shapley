# DTG-Shapley simulated raw data

## Title

Raw simulated datasets used in the DTG-Shapley experiments.

## Description

This directory provides the complete, human- and machine-readable definitions
of the simulated cooperative games and the simulated weak-field augmentations
reported in the DTG-Shapley paper. All tabular data are stored as CSV with a
header row and UTF-8 encoding.

## Files

- `simulated_games_members.csv`: one row per member for every synthetic game,
  including the additive value and analytic Shapley value.
- `simulated_games_interactions.csv`: one row per unanimity interaction,
  including its participating members and bonus value.
- `iris_with_weak_fields_seed11.csv`, `iris_with_weak_fields_seed23.csv`, and
  `iris_with_weak_fields_seed42.csv`: the complete 150-row Iris dataset with
  eight independently generated binary weak fields and the target label.
- `simulated_dataset_manifest.csv`: row counts, SHA-256 checksums, and short
  descriptions for the five raw CSV files.
- `generate_simulated_datasets.py`: a self-contained script used to reproduce
  the files. The repository also keeps the canonical copy under `code/`.

The synthetic-game files cover the paper's L1 contextual-tail games with 8,
16, and 32 low-contribution members; the L2 second- and third-order hidden
interaction games with 16 tail members; and the G1 balanced-interaction game.

## Simulation methodology

Synthetic games are additive cooperative games with explicitly listed
unanimity-interaction bonuses. For a coalition, the value is the sum of the
additive values of its members plus each interaction bonus whose complete
member set is present. The exact Shapley value equals the member's additive
value plus an equal share of every interaction bonus containing that member.

For each Iris augmentation, NumPy's `default_rng` is initialized with seed 11,
23, or 42. It generates eight integer fields with values in `{1, 2}`. The
fields are generated without access to the Iris class label and are therefore
independent of the target by construction. The CSV contains the original four
Iris measurements, all eight generated fields, and numeric and textual target
labels.

## Reproduction

Create the documented environment. From the root of the DTG-Shapley repository,
run:

```bash
conda env create -f code/environment.yml
conda activate dtg-shapley
python code/generate_simulated_datasets.py
```

After extracting the PeerJ supplemental `dataset.zip`, run the bundled copy:

```bash
python generate_simulated_datasets.py
```

The script deterministically overwrites the generated CSV files in their
current directory and writes a manifest containing their SHA-256 checksums. A
different destination may be selected with `--output-dir PATH`.

## Dataset source and citation

The base Iris measurements and labels originate from the UCI Machine Learning
Repository:

Fisher RA. Iris. UCI Machine Learning Repository. 1988.
https://doi.org/10.24432/C56C76

The Iris dataset is distributed by UCI under CC BY 4.0. The generated weak
fields and synthetic-game definitions are supplied as research data associated
with the DTG-Shapley paper. See the repository-level `LICENSES.md` for the
license boundaries.
