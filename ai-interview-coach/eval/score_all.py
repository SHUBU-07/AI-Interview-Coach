"""
Re-score ALL stored answers with the configured model (B1, B2, B3, S1, S2) and log to the database.

    python -m eval.score_all            # then:  python -m eval.export

Why: answers scored live during practice may have used different model/prompt versions. Re-scoring
everything once, with one fixed configuration, gives a clean, reproducible evaluation. Every call is
logged (model, version, temperature, prompt hash, raw output). Failed LLM outputs are stored but flagged
invalid and excluded from evaluation.
"""

import argparse

from coach.db import CoachDB
from eval.common import build_scorers, load_config, load_questions, resolve, score_answers
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score all stored answers with one fixed configuration")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    db = CoachDB(resolve(cfg["paths"]["database"]))
    answers = pd.DataFrame(db.get_all_answers())
    if answers.empty:
        raise SystemExit("No answers in the database yet.")
    if args.limit:
        answers = answers.head(args.limit)
    scorers = build_scorers(cfg, demo=False, db=db)
    print(f"Scoring {len(answers)} answers with {cfg['llm']['backend']}:{cfg['llm']['model']} "
          f"(temperature {cfg['llm']['temperature']})")
    scores = score_answers(answers, load_questions(cfg), scorers, cfg)
    print("\nValid scores per method:", {c: int(scores[c].notna().sum()) for c in scores.columns if c != "answer_id"})
    print("Next:  python -m eval.export")


if __name__ == "__main__":
    main()
