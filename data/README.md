# Data Provenance

This repository includes split metadata and processed graph files used by the
ECABSD training and evaluation scripts.

- `ppi_dataset.csv` is the complex-level manifest used to build the train,
  validation, and test splits.
- `splits.csv` assigns PDB chain pairs to `train`, `val`, and `test`.
  The current split is deterministic by unique PDB ID so the same PDB ID does
  not appear in more than one partition.
- `processed/*.pt` files are PyTorch Geometric residue graphs generated from
  the corresponding PDB chains.

Before using the benchmark numbers in a publication, run:

```bash
python check_leakage.py --mmseqs
```

The Tier-2 MMseqs2 check verifies that no sequence-similar chains cross split
boundaries at the configured identity cutoff.
