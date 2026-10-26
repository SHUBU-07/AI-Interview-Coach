"""
Report generator. Builds results.md (study) or results_demo.md (demo) ONLY from the artifacts
written by the evaluation stages in eval/outputs/. It contains no hardcoded numbers: if an
artifact is missing or the data are insufficient, the section says so.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from eval.common import load_config, load_json, out_dir, resolve


def df_to_markdown(df: Optional[pd.DataFrame]) -> str:
    if df is None or df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if pd.isna(v):
                cells.append("N/A")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def build_report(out: Path, demo: bool) -> str:
    meta = load_json(out / "run_meta.json") or {}
    ceiling = load_json(out / "human_ceiling.json") or {}
    ni = load_json(out / "non_inferiority.json")
    asr = load_json(out / "transcription_audit.json")
    agreement = _csv(out / "system_human_agreement_table.csv")
    test_retest = _csv(out / "reliability_test_retest.csv")
    prompt_sens = _csv(out / "reliability_prompt_sensitivity.csv")
    counterfactual = _csv(out / "fairness_counterfactual_audit.csv")
    groups = _csv(out / "fairness_group_analysis_first_language.csv")

    L = []
    L.append("# AI Interview Coach: Evaluation Report" + (" (DEMO)" if demo else ""))
    L.append(f"\n*Generated: {meta.get('generated_utc', 'unknown')} UTC | Mode: **{meta.get('mode', '?')}** | "
             f"LLM: `{meta.get('backend')}` / `{meta.get('model')}` (temperature {meta.get('temperature')}) | "
             f"Answers in dataset: {meta.get('n_answers', 'N/A')}*")
    if demo:
        L.append("\n> **DEMO ONLY.** This report was produced from synthetic data and a deterministic MOCK language model "
                 "to test that the pipeline runs. None of these numbers is evidence about the system. "
                 "Do not cite them.")
    L.append("\n> **Transparency notice**: every number below is computed from the data files by the evaluation scripts. "
             "Where data are insufficient the section says so instead of showing a value.\n")

    L.append("## 1. Human-human reliability (the ceiling)")
    if ceiling and "error" not in ceiling:
        L.append(f"- Raters: **{ceiling.get('num_raters')}** | answers rated by all raters: **{ceiling.get('shared_answers')}**")
        L.append(f"- Ceiling = mean pairwise single-rater ICC(2,1): **{ceiling.get('human_ceiling_icc')}**")
        L.append(f"- All-rater ICC(2,1) (information only): {ceiling.get('human_icc_all_raters')}")
        L.append(f"- Mean pairwise quadratic weighted kappa: **{ceiling.get('human_ceiling_qwk')}**")
        L.append(f"- Typical |rater A - rater B| on the overall score: {ceiling.get('typical_abs_disagreement')}")
        dims = ceiling.get("dimension_icc", {})
        if dims:
            L.append("\n| Dimension | All-rater ICC(2,1) |\n| --- | --- |")
            L += [f"| {d} | {v} |" for d, v in dims.items()]
    else:
        L.append(f"> Insufficient data: {ceiling.get('error', 'human ratings not available.')}")

    L.append("\n## 2. System vs human agreement")
    if agreement is not None and not agreement.empty:
        L.append("`ICC_vs_Single_Rater` is the like-for-like comparison with the human ceiling above. "
                 "Other columns compare against the mean of raters, which is more reliable than any single rater "
                 "and therefore flatters the system. Failed or unavailable scores (NaN) are excluded.\n")
        L.append(df_to_markdown(agreement))
    else:
        L.append("> Insufficient data: no system scores could be compared with human ratings.")

    L.append("\n## 3. Non-inferiority of S1 to the human ceiling")
    if ni and "error" not in ni:
        L.append(f"- Answers used: {ni['n_answers']} | raters: {ni['n_raters']} | pre-set margin: {ni['margin_delta']}")
        L.append(f"- Human ceiling ICC: {ni['human_ceiling_icc']} | S1 vs single rater ICC: {ni['system_icc_single_rater']}")
        L.append(f"- Difference (human - S1): {ni['icc_difference']} (bootstrap 95% CI {ni['diff_95_ci'][0]} to {ni['diff_95_ci'][1]})")
        L.append(f"- **Verdict: {ni['conclusion']}**")
    else:
        L.append(f"> Insufficient data: {(ni or {}).get('error', 'not run.')}")

    L.append("\n## 4. Scoring reliability")
    L.append("### A. Test-retest (repeated S1 runs on identical answers)")
    L.append(df_to_markdown(test_retest) or "> Not run or insufficient data.")
    L.append("\n### B. Prompt sensitivity (paraphrased prompts)")
    L.append(df_to_markdown(prompt_sens) or "> Not run or insufficient data.")
    if demo or (meta.get("temperature") == 0.0):
        L.append("\n*Note: at temperature 0 (or with the mock model) repeated runs are near-deterministic, so a high "
                 "test-retest ICC is expected and is weak evidence. Prompt sensitivity is more informative.*")

    L.append("\n## 5. Fairness audits")
    L.append("### A. Counterfactual identity test (only the stated attribute changes)")
    if counterfactual is not None and not counterfactual.empty:
        L.append(df_to_markdown(counterfactual))
        L.append("\n*A shift larger than typical human disagreement suggests the scorer reacts to identity information.*")
    else:
        L.append("> Not run or insufficient data.")
    L.append("\n### B. Transcription (ASR) error vs scoring error")
    if asr and "error" not in asr:
        L.append(f"- Answers with a manually corrected reference: {asr.get('overall_n')}")
        L.append(f"- Pearson r (WER vs absolute scoring error): {asr.get('wer_scoring_error_pearson_r')} (p = {asr.get('p_value')})")
        L.append(f"- {asr.get('conclusion')}")
        gb = pd.DataFrame(asr.get("group_breakdown", []))
        if not gb.empty:
            L.append("\n" + df_to_markdown(gb))
    else:
        L.append(f"> Insufficient data: {(asr or {}).get('error', 'not run.')}")
    L.append("\n### C. Error by first-language group (system minus human)")
    if groups is not None and not groups.empty:
        L.append(df_to_markdown(groups))
        L.append("\n*Groups with fewer than the minimum number of answers are labelled 'Too small for inference'; no conclusion should be drawn from them.*")
    else:
        L.append("> Insufficient data.")

    L.append("\n## 6. Limitations")
    L += [
        "1. **Advisory only.** Scores are practice feedback, not hiring or selection decisions.",
        "2. **Raters are imperfect.** The ceiling reflects this study's rater cohort.",
        "3. **Model drift.** Results apply to the model, version and prompt recorded above.",
        "4. **Sample.** Results are specific to the participants collected and may not generalise to other populations, accents or interview types.",
        "5. **No causal claims.** This evaluation measures agreement and fairness, not whether practice improves real interview outcomes.",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate results.md from evaluation artifacts")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = out_dir(cfg, args.demo)
    text = build_report(out, args.demo)
    target = resolve("results_demo.md" if args.demo else "results.md")
    target.write_text(text, encoding="utf-8")
    print(f"Generated {target.name}")


if __name__ == "__main__":
    main()
