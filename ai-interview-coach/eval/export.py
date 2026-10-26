"""
Export the study database to CSV files used by the evaluation scripts.

    python -m eval.export

Writes (paths from config.yaml): answers.csv, human_ratings.csv, merged_dataset.csv.
These files contain participant answers: they are git-ignored and must not be published
unless your consent form allows it.
"""

import argparse
import pandas as pd

from coach.db import CoachDB
from eval.common import load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Export study database to CSV")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    db_path = resolve(cfg["paths"]["database"])
    if not db_path.exists():
        raise SystemExit(f"Database not found at {db_path}. Run the app and collect answers first.")
    db = CoachDB(db_path)

    answers = pd.DataFrame(db.get_all_answers())
    ratings = pd.DataFrame(db.get_all_human_ratings())
    answers.to_csv(resolve(cfg["paths"]["answers_csv"]), index=False)
    ratings.to_csv(resolve(cfg["paths"]["human_ratings_csv"]), index=False)
    merged = db.export_merged_dataset(resolve(cfg["paths"]["merged_dataset_csv"]))

    n_rated2 = int((merged.get("human_count", pd.Series(dtype=float)).fillna(0) >= 2).sum()) if len(merged) else 0
    print(f"Answers: {len(answers)} | Human ratings: {len(ratings)} | Raters: "
          f"{ratings['rater_id'].nunique() if len(ratings) else 0}")
    print(f"Answers rated by >= 2 raters: {n_rated2}")
    score_cols = [c for c in merged.columns if c.startswith("score_")]
    print("System scores present:", {c: int(merged[c].notna().sum()) for c in score_cols} if score_cols else "none")
    print(f"Merged dataset: {resolve(cfg['paths']['merged_dataset_csv'])}")


if __name__ == "__main__":
    main()
