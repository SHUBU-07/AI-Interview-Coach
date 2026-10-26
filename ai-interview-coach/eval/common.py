"""
Shared helpers for the evaluation scripts.

Two modes
- STUDY mode (default): reads real data exported from the app database.
- DEMO mode (--demo): uses synthetic files in data/demo/ and the MOCK LLM. This only tests that the
  pipeline runs end to end. Demo outputs go to a separate folder and are labelled DEMO everywhere.
"""

from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
SCORE_COLS = {"B1": "score_B1", "B2": "score_B2", "B3": "score_B3", "S1": "score_S1", "S2": "score_S2"}


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else BASE_DIR / p


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    with open(resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def out_dir(cfg: Dict[str, Any], demo: bool) -> Path:
    d = resolve(cfg["paths"]["eval_output_dir"]) / ("demo" if demo else "")
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV, allowing '#' comment lines (used to label demo files)."""
    return pd.read_csv(resolve(path), comment="#")


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_questions(cfg: Dict[str, Any]) -> pd.DataFrame:
    return read_csv(cfg["paths"]["questions_csv"])


def write_run_meta(cfg: Dict[str, Any], demo: bool, n_answers: Optional[int] = None) -> Dict[str, Any]:
    llm = cfg.get("llm", {})
    meta = {
        "mode": "DEMO" if demo else "STUDY",
        "backend": "mock" if demo else llm.get("backend"),
        "model": "mock-rubric-v1" if demo else llm.get("model"),
        "temperature": llm.get("temperature"),
        "rubric_version": "see rubric.yaml",
        "non_inferiority_margin": cfg["eval"].get("non_inferiority_margin"),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = out_dir(cfg, demo) / "run_meta.json"
    old = load_json(path) or {}
    if n_answers is not None:
        meta["n_answers"] = n_answers
    elif "n_answers" in old:
        meta["n_answers"] = old["n_answers"]
    save_json(path, meta)
    return meta


def human_disagreement(ratings_df: pd.DataFrame) -> Optional[float]:
    """Typical |rater A - rater B| on the overall score, averaged over rater pairs and answers."""
    if ratings_df is None or ratings_df.empty or "rater_id" not in ratings_df.columns:
        return None
    piv = ratings_df.pivot(index="answer_id", columns="rater_id", values="overall_score")
    diffs = []
    for a, b in itertools.combinations(piv.columns, 2):
        both = piv[[a, b]].dropna()
        diffs.extend((both[a] - both[b]).abs().tolist())
    return round(float(np.mean(diffs)), 3) if diffs else None


def build_scorers(cfg: Dict[str, Any], demo: bool = False, db=None) -> Dict[str, Any]:
    from coach.rubric import Rubric
    from coach.llm_backend import MockLLMBackend, get_llm_backend
    from coach.scorer_llm import S1Scorer
    from coach.ensemble import S2EnsembleScorer
    from coach.baselines import B1TfidfScorer, B2EmbeddingScorer, B3UnconstrainedLLMScorer

    rubric = Rubric.load_from_yaml(resolve(cfg["paths"]["rubric"]))
    llm = MockLLMBackend(temperature=0.0) if demo else get_llm_backend(cfg)
    s1 = S1Scorer(rubric=rubric, llm_backend=llm, db=db, max_retries=int(cfg["llm"].get("max_retries", 2)))
    return {
        "rubric": rubric, "llm": llm, "s1": s1,
        "s2": S2EnsembleScorer([s1], aggregation=cfg["ensemble"]["aggregation"], db=db),
        "b1": B1TfidfScorer(db=db, max_features=int(cfg["baselines"]["tfidf_max_features"])),
        "b2": B2EmbeddingScorer(model_name=cfg["baselines"]["embedding_model"], db=db),
        "b3": B3UnconstrainedLLMScorer(llm_backend=llm, db=db),
    }


def score_answers(answers_df: pd.DataFrame, questions_df: pd.DataFrame, scorers: Dict[str, Any],
                  cfg: Dict[str, Any], verbose: bool = True) -> pd.DataFrame:
    """Run B1, B2, B3, S1, S2 on every answer. Failed scores are NaN (never a fallback number)."""
    qmap = questions_df.set_index("question_id")
    has_audio = bool(cfg.get("scoring", {}).get("include_delivery", False))
    num_runs = int(cfg["ensemble"].get("num_runs", 3))
    rows = []
    for i, r in enumerate(answers_df.itertuples(index=False), 1):
        if r.question_id not in qmap.index:
            print(f"[warn] question_id {r.question_id} not in questions.csv; skipping {r.answer_id}")
            continue
        q = qmap.loc[r.question_id]
        qtext, ref, text, aid = q["question_text"], str(q.get("reference_answer", "")), r.transcript, r.answer_id
        out = {"answer_id": aid}
        out["score_B1"] = scorers["b1"].score_answer(aid, text, ref).overall_score
        out["score_B2"] = scorers["b2"].score_answer(aid, text, ref).overall_score
        out["score_B3"] = scorers["b3"].score_answer(aid, qtext, text).overall_score
        s1 = scorers["s1"].score_answer(aid, qtext, text, has_audio=has_audio)
        out["score_S1"] = s1.overall_score if s1.validation_passed else float("nan")
        s2 = scorers["s2"].score_answer(aid, qtext, text, has_audio=has_audio, num_runs=num_runs)
        out["score_S2"] = s2.overall_score if s2.validation_passed else float("nan")
        rows.append(out)
        if verbose:
            print(f"  scored {i}/{len(answers_df)}: {aid}")
    return pd.DataFrame(rows)


def load_dataset(cfg: Dict[str, Any], demo: bool, require_scores: bool = True
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (merged_df, ratings_df, questions_df)."""
    questions_df = load_questions(cfg)
    if demo:
        d = cfg["paths"]["demo_dir"]
        answers_df = read_csv(Path(d) / "demo_answers.csv")
        ratings_df = read_csv(Path(d) / "demo_human_ratings.csv")
        means = (ratings_df.groupby("answer_id")
                 .agg(human_count=("rater_id", "nunique"), human_overall_mean=("overall_score", "mean"))
                 .reset_index())
        merged = answers_df.merge(means, on="answer_id", how="left")
        if require_scores:
            cache = out_dir(cfg, True) / "demo_scored.csv"
            if cache.exists():
                scores = pd.read_csv(cache)
            else:
                print("[DEMO] Scoring demo answers with the MOCK LLM (pipeline test only)...")
                scores = score_answers(answers_df, questions_df, build_scorers(cfg, demo=True), cfg, verbose=False)
                scores.to_csv(cache, index=False)
            merged = merged.merge(scores, on="answer_id", how="left")
        return merged, ratings_df, questions_df

    merged_path, ratings_path = resolve(cfg["paths"]["merged_dataset_csv"]), resolve(cfg["paths"]["human_ratings_csv"])
    if not merged_path.exists() or not ratings_path.exists():
        raise SystemExit(
            f"Missing {merged_path.name} or {ratings_path.name}. Collect data in the app, then run:\n"
            f"    python -m eval.export"
        )
    merged, ratings_df = pd.read_csv(merged_path), pd.read_csv(ratings_path)
    if require_scores and not any(c in merged.columns and merged[c].notna().any() for c in SCORE_COLS.values()):
        raise SystemExit("No system scores in merged dataset. Run:  python -m eval.score_all  then  python -m eval.export")
    return merged, ratings_df, questions_df
