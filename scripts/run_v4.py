"""v4 runs (docs/protocol_v4.md). From the repo root:

    PYTHONPATH=src python scripts/run_v4.py dry-run [--source mock|gpu] [--model MODEL_ID]
    PYTHONPATH=src python scripts/run_v4.py dry-run-report [--source mock|gpu]

dry-run (step b): every cell of the three arms for every model (or one), 1 seed. Arm 1 on 12 of the
100 current questions (3 per family); Arms 2 and 3 on 5 questions each. Writes the replies, the turn-1
contexts, run_info.jsonl (revision, precision, packages, lock check, settings, GPU), any failures, and
the per-model report: how replies end, the round trip, empty replies and timing. The report shows
nothing by condition or arm.

dry-run-report rebuilds the per-model report from the files on disk, without loading a model.

--source mock runs the same orchestration without a GPU, into results_v4/mock/. --source gpu needs a
CUDA GPU, the v4 environment and the lock file (the load refuses without it).
"""
import argparse

from ds import load_config
from ds.generate_v4 import gpu_parts, mock_parts, run_dry_run, write_report
from ds.items_v4 import make_bank


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["dry-run", "dry-run-report"])
    p.add_argument("--source", choices=["mock", "gpu"], default="mock")
    p.add_argument("--model", help="one model id from the config (default: all)")
    p.add_argument("--config", default="configs/experiment_v4.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.mode == "dry-run-report":
        report = write_report(cfg, args.source)
    else:
        make_parts = gpu_parts if args.source == "gpu" else mock_parts
        report = run_dry_run(cfg, make_bank(cfg), make_parts, args.source, args.model)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
