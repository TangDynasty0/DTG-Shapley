# CSV datasets and attribution

All data distributed with this reproducibility package are stored under
`dataset/csv/` as uncompressed CSV. These files can be inspected in a text
editor, loaded by spreadsheet software, or parsed by standard CSV libraries.
The public code reads these CSV files directly; no NPZ file or compressed
dataset archive is required.

## Public datasets

| Dataset | CSV representation | Observations | Features | DOI |
| --- | --- | ---: | ---: | --- |
| Iris | [`csv/iris.csv`](csv/iris.csv) | 150 | 4 | 10.24432/C56C76 |
| Wine | [`csv/wine.csv`](csv/wine.csv) | 178 | 13 | 10.24432/C5PC7J |
| CNAE-9 | [`csv/cnae_9.csv`](csv/cnae_9.csv) | 1,080 | 856 | 10.24432/C5SC8V |
| Internet Advertisements | [`csv/internet_advertisements.csv`](csv/internet_advertisements.csv) | 3,279 | 1,558 | 10.24432/C5V011 |
| Spambase | [`csv/spambase.csv`](csv/spambase.csv) | 4,601 | 57 | 10.24432/C53G6X |
| Farm Ads | [`csv/farm_ads_samples.csv`](csv/farm_ads_samples.csv) and [`csv/farm_ads_features.csv`](csv/farm_ads_features.csv) | 4,143 | 54,877 sparse | 10.24432/C5ZC8D |

Dense tables use one row per observation. The first column is `sample_id`, the
last numeric column is `target`, and the intervening columns are features.
Iris and Wine additionally include a human-readable `target_label` column.

Farm Ads is stored in normalized sparse CSV form. `farm_ads_samples.csv`
contains `sample_id,target`; `farm_ads_features.csv` contains
`sample_id,feature_index_one_based,value` for each nonzero entry. This preserves
the complete 54,877-feature representation without creating a very large dense
CSV.

## Simulated datasets

- [`csv/simulated_games_members.csv`](csv/simulated_games_members.csv) contains
  member-level additive values and analytic Shapley values for all deterministic
  simulated games.
- [`csv/simulated_games_interactions.csv`](csv/simulated_games_interactions.csv)
  contains the unanimity interactions used by those games.
- [`csv/iris_with_weak_fields_seed11.csv`](csv/iris_with_weak_fields_seed11.csv),
  [`seed23`](csv/iris_with_weak_fields_seed23.csv), and
  [`seed42`](csv/iris_with_weak_fields_seed42.csv) contain the complete Iris
  table plus eight independently generated fields taking values in `{1, 2}`.
- [`csv/simulated_dataset_manifest.csv`](csv/simulated_dataset_manifest.csv)
  records the generated-file checksums and descriptions.

Run `python code/generate_simulated_datasets.py` from the repository root to
regenerate the simulated-game and Iris weak-field CSV files. The script reads
`csv/iris.csv` and does not require a binary dataset cache.

## Preprocessing represented in the CSV files

- Iris and Wine retain their original continuous measurements and class labels.
- CNAE-9 retains the official integer word-frequency representation.
- Farm Ads retains the official stemmed, stop-word-filtered SVMlight values,
  converted without numeric alteration to sparse long-table CSV.
- The first three Internet Advertisements geometry fields use at most ten
  empirical-quantile bins after finite-column median imputation. Remaining
  binary indicators retain their values; documented missing indicators use
  code 2.
- Every Spambase feature uses at most ten empirical-quantile bins.
- No observations were removed from any dataset.

The CSV matrices are the exact analysis-ready representations consumed by the
revised loader. Deterministic subsets are selected without replacement using
the configured seed and returned in ascending original-column order.

## Integrity and licenses

[`csv/dataset_manifest.csv`](csv/dataset_manifest.csv) records the row count,
column count, provenance, license, and SHA-256 value of each CSV data file.
`checksums.sha256` provides a second package-level integrity list.

The six public datasets are third-party works from the UCI Machine Learning
Repository and remain under CC BY 4.0. The authors' simulated definitions,
generated weak-field values, and experiment outputs are also released under
CC BY 4.0. Source code is separately licensed under MIT; see `LICENSES.md`.
