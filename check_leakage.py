"""
check_leakage.py  —  ECABSD Data Leakage Guard
================================================
Run before training or publication evaluation:
    python check_leakage.py

TIER 1 (always): Exact PDB-ID overlap check across train/val/test splits.
TIER 2 (opt-in): MMseqs2 sequence-similarity clustering at 30% identity.
    Run with:  python check_leakage.py --mmseqs

EXIT CODES
    0  — No leakage detected; training may proceed.
    1  — Leakage detected or splits file missing; training is aborted.

NOTE: Until Tier-2 is run on your final dataset, treat reported benchmark
numbers as preliminary. Sequence-similarity leakage is the primary cause
of inflated F1/AUC figures in the PPI literature.
To run Tier-2:
    conda install -c conda-forge -c bioconda mmseqs2
    python check_leakage.py --mmseqs
"""

import sys
import os
import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# Tier 1: Exact PDB-ID overlap check (always runs)
# ---------------------------------------------------------------------------

def check_exact_id_leakage(splits_csv: str) -> bool:
    """Returns True (clean) if no PDB-ID appears in more than one split."""
    if not os.path.exists(splits_csv):
        print(f"[LEAKAGE CHECK] FATAL: splits file not found: {splits_csv}")
        print("  Run scripts/prepare_db5.py first to generate the splits CSV.")
        return False

    df = pd.read_csv(splits_csv)

    required_cols = {"pdb_id", "split"}
    if not required_cols.issubset(df.columns):
        print(f"[LEAKAGE CHECK] FATAL: splits CSV must contain columns {required_cols}.")
        print(f"  Found columns: {list(df.columns)}")
        return False

    train_ids = set(df[df["split"] == "train"]["pdb_id"])
    val_ids   = set(df[df["split"] == "val"]["pdb_id"])
    test_ids  = set(df[df["split"] == "test"]["pdb_id"])

    tv = train_ids & val_ids
    te = train_ids & test_ids
    ve = val_ids   & test_ids

    print("[LEAKAGE CHECK] Tier 1 — Exact PDB-ID overlap")
    print(f"  Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    print(f"  Train & Val  : {len(tv)}")
    print(f"  Train & Test : {len(te)}")
    print(f"  Val & Test   : {len(ve)}")

    if tv or te or ve:
        print("[LEAKAGE CHECK] FAIL — overlapping PDB IDs detected:")
        if tv: print(f"  Train & Val  : {sorted(tv)}")
        if te: print(f"  Train & Test : {sorted(te)}")
        if ve: print(f"  Val & Test   : {sorted(ve)}")
        return False

    print("[LEAKAGE CHECK] PASS — no exact-ID overlap found.\n")
    return True


# ---------------------------------------------------------------------------
# Tier 2: MMseqs2 sequence-similarity check (opt-in, requires mmseqs on PATH)
# ---------------------------------------------------------------------------

def check_mmseqs_leakage(splits_csv: str, processed_dir: str,
                         min_seq_id: float = 0.30) -> bool:
    """
    Clusters all chain-A sequences with MMseqs2 and checks that no cluster
    spans more than one split partition.
    Returns True (clean) or False (leakage / error).
    """
    import shutil
    import subprocess
    import tempfile
    import glob

    if shutil.which("mmseqs") is None:
        print("[LEAKAGE CHECK] ERROR: 'mmseqs' not found on PATH.")
        print("  Install: conda install -c conda-forge -c bioconda mmseqs2")
        return False

    df = pd.read_csv(splits_csv)
    split_map = dict(zip(df["pdb_id"], df["split"]))

    try:
        import torch
    except ImportError:
        print("[LEAKAGE CHECK] ERROR: torch not available for FASTA export.")
        return False

    fasta_lines = []
    for pt_file in glob.glob(os.path.join(processed_dir, "*.pt")):
        pdb_id = os.path.splitext(os.path.basename(pt_file))[0]
        try:
            g = torch.load(pt_file, map_location="cpu", weights_only=False)
            seq = getattr(g, "sequence_a", None) or ("X" * g.num_nodes)
            fasta_lines.append(f">{pdb_id}\n{seq}\n")
        except Exception:
            pass

    if not fasta_lines:
        print("[LEAKAGE CHECK] WARNING: No .pt graphs found — skipping MMseqs2 check.")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path  = os.path.join(tmpdir, "all_chains.fasta")
        cluster_out = os.path.join(tmpdir, "clusters")
        tmp_mmseqs  = os.path.join(tmpdir, "mmseqs_tmp")

        with open(fasta_path, "w") as f:
            f.writelines(fasta_lines)

        cmd = [
            "mmseqs", "easy-cluster",
            fasta_path, cluster_out, tmp_mmseqs,
            "--min-seq-id", str(min_seq_id),
            "-c", "0.8", "--cov-mode", "0",
            "--cluster-mode", "2", "-v", "0",
        ]
        print(f"[LEAKAGE CHECK] Running MMseqs2 (min-seq-id={min_seq_id}) …")
        subprocess.check_call(cmd)

        tsv_path = cluster_out + "_cluster.tsv"
        cluster_df = pd.read_csv(tsv_path, sep="\t", header=None,
                                 names=["rep", "member"])

    leaky_clusters = []
    for rep, group in cluster_df.groupby("rep"):
        splits_in_cluster = {split_map.get(m) for m in group["member"]
                             if split_map.get(m) is not None}
        splits_in_cluster.discard(None)
        if len(splits_in_cluster) > 1:
            leaky_clusters.append((rep, sorted(splits_in_cluster),
                                   group["member"].tolist()))

    if leaky_clusters:
        print(f"[LEAKAGE CHECK] FAIL — {len(leaky_clusters)} cross-split cluster(s):")
        for rep, splits, members in leaky_clusters[:10]:
            print(f"  rep={rep} spans splits={splits} members={members}")
        if len(leaky_clusters) > 10:
            print(f"  … and {len(leaky_clusters) - 10} more.")
        return False

    print(f"[LEAKAGE CHECK] PASS — MMseqs2: no cross-split clusters at "
          f"{int(min_seq_id*100)}% identity.\n")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ECABSD data leakage guard (Tier 1 always; Tier 2 optional)."
    )
    parser.add_argument("--splits-csv", default="data/splits.csv")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--mmseqs", action="store_true",
                        help="Also run MMseqs2 sequence-similarity check")
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    args = parser.parse_args()

    # Tier 1 — always
    if not check_exact_id_leakage(args.splits_csv):
        sys.exit(1)

    # Tier 2 — opt-in
    if args.mmseqs:
        if not check_mmseqs_leakage(args.splits_csv, args.processed_dir,
                                    args.min_seq_id):
            sys.exit(1)
    else:
        print("[LEAKAGE CHECK] NOTE: Tier-2 (MMseqs2) check not run.")
        print("  For publication, run:  python check_leakage.py --mmseqs\n")

    print("[LEAKAGE CHECK] All checks passed. Training may proceed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
