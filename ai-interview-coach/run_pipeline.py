"""
Master Pipeline Runner for AI Interview Coach.
Executes all evaluation and fairness auditing modules end-to-end.
Usage:
    python run_pipeline.py [--demo] [--config config.yaml]
"""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="AI Interview Coach: End-to-End Evaluation Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run full pipeline on synthetic benchmark demo dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--rescore", action="store_true",
                        help="Study mode: re-score all stored answers with one fixed config before evaluating")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(base_dir))

    mode_label = "DEMO BENCHMARK DATASET" if args.demo else "EMPIRICAL RESEARCH DATASET"
    print("=" * 70)
    print(f"AI INTERVIEW COACH: EVALUATION & FAIRNESS AUDIT PIPELINE")
    print(f"Target: {mode_label}")
    print("=" * 70)

    stages = []
    if not args.demo:
        if args.rescore:
            stages.append(("STAGE 0a: Re-score all answers (one fixed configuration)", [sys.executable, "-m", "eval.score_all"]))
        stages.append(("STAGE 0b: Export database to CSV", [sys.executable, "-m", "eval.export"]))
    stages += [
        ("STAGE 1: Human Ceiling & Agreement Evaluation", [sys.executable, "-m", "eval.agreement"]),
        ("STAGE 2: Reliability & Prompt Sensitivity Audits", [sys.executable, "-m", "eval.reliability"]),
        ("STAGE 3: Algorithmic Fairness & Bias Audits", [sys.executable, "-m", "eval.fairness"]),
        ("STAGE 4: Report Synthesis (results.md)", [sys.executable, "-m", "eval.report"]),
    ]

    for stage_name, cmd in stages:
        print(f"\n>>> Running {stage_name}...")
        full_cmd = list(cmd)
        if args.demo and not stage_name.startswith("STAGE 0"):
            full_cmd.append("--demo")
        full_cmd.extend(["--config", args.config])

        res = subprocess.run(full_cmd, cwd=str(base_dir))
        if res.returncode != 0:
            print(f"[ERROR] Encountered in {stage_name} (exit code: {res.returncode})")
            sys.exit(res.returncode)

    print("\n" + "=" * 70)
    print("[SUCCESS] EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Generated Report: {base_dir / ('results_demo.md' if args.demo else 'results.md')}")
    print(f"Generated Tables & Plots: {base_dir / 'eval' / 'outputs' / ('demo' if args.demo else '')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
