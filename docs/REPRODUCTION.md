# Reproduction

This document describes the single public workflow: replay the frozen FPT
source, then run the fixed eight-model survival protocol.

## Inputs

Place the original XJTU-SY CSV files under `data/XJTU-SY/` using the three
condition directories and five bearings per condition. Each numeric file must
remain unchanged and contain the original two-channel record.

The repository includes the derived FPT score traces needed for replay:

- `fpt_detection/reference/labels.csv` is the canonical 75-row label target.
- `fpt_detection/reference/reference.json` binds the score bundle and raw-data
  identity.
- `fpt_detection/reference/scores/*.npz` contains the 75 detector inputs.

The raw CSV corpus and trained model checkpoints are not included.

## FPT replay

Run:

```bash
python src/generate_fpt_labels.py
```

The command reads the bundled scores, reruns the detector, verifies all 75 rows,
and binds the generated artifact to the raw CSV inventory. It writes to
`outputs/fpt_v2_a_coverage/` unless `--output-dir` is supplied.
Use `--data-dir` when the dataset is stored elsewhere.

For survival training, the same dataset must be available at `data/XJTU-SY/`
as a directory or symbolic link. Use a new output directory for each replay;
existing artifacts are never overwritten.

The generated manifest and CSV must match the immutable reference contract.
Changing the detector output, label schema, score files, or raw-data identity
causes the command to fail before publication.

## Survival workflow

The main entry point is:

```bash
python src/run_cross_validation.py \
  --models all --conditions 1 --censoring-levels 0.5 \
  --device cuda --continue-on-metric-mismatch \
  --output-dir outputs/c2_q50
```

The default profile uses condition index `1` (C2), 50% censoring, five
leave-one-bearing-out folds, and these eight models:

```text
otsurv rawcnn coxph coxboost rsf mtlr dsm bnnsurv
```

Use `--device cpu` when CUDA is unavailable. The launcher accepts subsets of
models, conditions, folds, and censoring levels for smoke tests. `--dry-run`
prints the task plan without reading the dataset, checking GPUs, or training.
OTSurv uses two CUDA devices, one per endpoint. The other neural baselines use
one device; classical models and DSM use CPU. Select physical CUDA devices with
`--gpu-ids` when needed. The full condition/censoring/fold/model matrix has 360
tasks.

The coordinator creates hash-bound feature caches from the raw CSV files. Each
worker consumes the same immutable FPT label artifact, writes its fold outputs,
and compares the resulting metrics with `src/paper/reference.json`.

## Outputs

The selected output directory contains:

- `run_manifest.json`: configuration, environment, input hashes, and source
  hashes;
- one directory per model, condition, censoring level, and outer fold;
- `results.csv` and `summary.csv`: observed fold metrics and aggregates;
- `completion.json`: completion state and historical comparison status.

The bundled reference metrics serve as verification targets during a new run.
With `--continue-on-metric-mismatch`, all independent tasks continue and a
completed run returns exit code `2` if numerical values differ. Without that
option, the launcher stops at the first mismatch. Execution errors stop either
mode. Exit code `0` means all historical comparisons passed.

## Reproduction boundary

The FPT endpoint is detector-defined and should not be interpreted as a physical
failure annotation. Raw-data formatting, file names, label contents, and score
traces are part of the frozen input contract. The public release does not claim
that every model and hardware combination reproduces every historical scalar
value bit-for-bit.

The release starts from frozen GroupVAE scores; generating new scores from raw
signals is outside its scope. Historical OTSurv reporting epochs were selected
post hoc using outer-test results and are now fixed. Tabular baselines retain
the historical forward-aligned moving averages. These settings reproduce the
recorded workflow and do not constitute independent nested or strictly online
validation.
