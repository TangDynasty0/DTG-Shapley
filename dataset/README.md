# Datasets and attribution

`uci/` contains the unmodified UCI archives used by the structured-table
runtime experiment. The UCI Machine Learning Repository distributes these
datasets under CC BY 4.0. Cite the corresponding dataset record when reusing
an archive:

| Dataset | DOI | UCI record |
| --- | --- | --- |
| CNAE-9 | 10.24432/C5SC8V | https://archive.ics.uci.edu/dataset/233/cnae+9 |
| Farm Ads | 10.24432/C5ZC8D | https://archive.ics.uci.edu/dataset/218/farm+ads |
| Internet Advertisements | 10.24432/C5V011 | https://archive.ics.uci.edu/dataset/51/internet+advertisements |
| Spambase | 10.24432/C53G6X | https://archive.ics.uci.edu/dataset/94/spambase |

Iris (DOI: 10.24432/C56C76) and Wine (DOI: 10.24432/C5PC7J) are loaded from
scikit-learn and are not duplicated in `uci/`. They are also UCI datasets
distributed under CC BY 4.0. Synthetic games are generated deterministically
by the experiment code.

`simulated_raw/` contains the human- and machine-readable raw CSV files used
for the simulated-data experiments. It includes the complete synthetic-game
definitions and three complete Iris tables augmented with eight weak fields.
Run `python code/generate_simulated_datasets.py` to reproduce these files.
`generated/` retains the corresponding compact NumPy weak-field matrices used
directly by the experiment runner; each matrix is generated independently of
the class label from integers in {1, 2}.

`checksums.sha256` records SHA-256 hashes for every archived or generated data
file in this directory. See the package-level `LICENSES.md` for the complete
licensing boundary.
