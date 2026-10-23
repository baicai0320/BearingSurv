# Frozen FPTv2-A Coverage Reference

This directory contains the hash-verified frozen FPT source used by the public
survival pipeline. It contains derived detector score traces, not raw vibration
records or model checkpoints.

`scores/*.npz` are detector inputs. `labels.csv` is only the comparison target.
Normal generation must rerun the detector, not copy this CSV into an output
directory. `reference.json` records each trace's training peers, profile,
content hash and reference identity.

The runtime checks the canonical label and reference-manifest SHA-256 values
against constants in `fpt_detection/labels.py`. Official artifacts must also
bind to the registered raw CSV corpus. Do not edit reference values to make a
replay pass. See [the reproduction guide](../../docs/REPRODUCTION.md) for the
public reproduction boundary.
