# BearingSurv

Code for **BearingSurv: Censoring-Aware Probabilistic Bearing Remaining
Useful Life Prediction** on the XJTU-SY dataset.

The public release contains the frozen FPT detector replay and the eight-model
survival-analysis workflow used by the paper.

## Installation

Use Linux with the provided Conda environment:

```bash
git clone https://github.com/baicai0320/BearingSurv.git
cd BearingSurv
conda env create --file environment.yml
conda activate surv

export PYTHONPATH="$PWD/src:$PWD"
export R_HOME="$CONDA_PREFIX/lib/R"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Data

Download the [XJTU-SY dataset](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)
and place the original CSV files under `data/XJTU-SY/`:

```text
data/XJTU-SY/
  35Hz12kN/Bearing1_1/1.csv
  37.5Hz11kN/Bearing2_1/1.csv
  40Hz10kN/Bearing3_5/N.csv
```

Keep the original CSV contents and include all five bearings in each condition.
Raw data and trained checkpoints are not bundled. The frozen FPT score traces
and reference manifests are included under `fpt_detection/reference/`.

## Reproduction

Generate the fixed FPT label artifact from the bundled score traces:

```bash
python src/generate_fpt_labels.py
```

Run the default C2, 50% censoring cross-validation profile:

```bash
python src/run_cross_validation.py \
  --models all --conditions 1 --censoring-levels 0.5 \
  --device cuda --continue-on-metric-mismatch \
  --output-dir outputs/c2_q50
```

Use `--device cpu` for CPU execution. Add `--dry-run` to inspect the task plan
without data access or training. Outputs are written to the selected directory,
including `results.csv`, `summary.csv`, and `completion.json`.

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the data contract, output
files, and interpretation of numerical comparison results.

## Scope

This release covers the fixed FPT replay and the main random-censoring
workflow. FPT labels are detector-defined endpoints rather than physical-failure
annotations. Full numerical agreement with every historical paper value has not
been established; any differences are retained in the run artifacts.

## Citation and License

Please cite the BearingSurv manuscript and the XJTU-SY dataset when using this
code. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
