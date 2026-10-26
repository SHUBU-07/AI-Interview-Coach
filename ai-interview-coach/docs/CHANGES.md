# Changes from the uploaded version

This file lists the problems found while reviewing the first version of the code, and what was done about each.
Most matter for the scientific claims of the project, not just for code quality.

## A. Fabricated or placeholder numbers (most important)

| # | Problem in the upload | Fix |
|---|---|---|
| 1 | `eval/report.py` hardcoded the human ceiling (ICC 0.88, QWK 0.86), the S1 ICC (0.84), its confidence interval and the ASR correlation (r = 0.28). The old `results.md` therefore showed numbers nobody had computed. | `report.py` now builds the report only from files written by the evaluation stages. Missing data produces "Insufficient data". |
| 2 | `__main__.py` (agreement CLI) generated the demo "system scores" by formula from the human mean (for example S1 = 0.96 x human + 0.1), so S1 passed non-inferiority by construction. | Demo mode now runs the real scorers (B1/B3/S1/S2) with a mock LLM on synthetic data. The mock is correctly judged on its actual output (in testing it was marked inferior). |
| 3 | `fairness.py` main set `human_overall_mean = 3.5` and `score_S1 = 3.6` for every answer when columns were missing, giving identical "0.1 error" rows. | Removed. Group analysis uses real merged data or reports insufficient data. |
| 4 | Counterfactual "typical human disagreement" was hardcoded to 0.5. | Computed from the human ratings (`eval.common.human_disagreement`). |
| 5 | `reliability.py` and `fairness.py` always used the mock LLM, even on real data. Test-retest ICC = 1.0 in `results.md` came from a deterministic mock. | They use the configured LLM. Demo mode and temperature 0 are flagged in the report as weak evidence. |
| 6 | Failed scoring silently produced numbers: B1/B2 returned 1.0 on error, B3 returned a 5/10 midpoint when it could not parse, S1 returned a 3 for every dimension, and S2 averaged those fallbacks in. | Failures are NaN or flagged `validation_passed = False` and excluded from all evaluation. The UI shows an error instead of a score. |
| 7 | An unrecognised `llm.backend` (and `gemini` was listed in the config but not implemented) silently fell back to the mock. | Raises an error. |

## B. Statistical problems

| # | Problem | Fix |
|---|---|---|
| 8 | System vs. the **mean of all raters** was compared with a human ceiling computed between **single raters**. A mean of raters is more reliable than one rater, which favours the system. | The ceiling is the mean pairwise single-rater ICC; the system is compared with each rater individually (`ICC_vs_Single_Rater`). |
| 9 | Non-inferiority used only the point estimate and bootstrapped one side. | The bootstrap resamples answers for both ceiling and system. Verdict is `non_inferior` / `inconclusive` / `inferior` from the CI upper bound. Needs >= 10 answers. |
| 10 | Evaluation included answers with only one human rating. | Only answers with >= `min_raters_per_answer` ratings are used. |

## C. Code bugs

| # | Problem | Fix |
|---|---|---|
| 11 | `python -m eval.agreement` did nothing (no `__main__` block; CLI was in a separate `__main__.py`). | CLI merged into `agreement.py`. A test now guards this. |
| 12 | Sidebar page "Feedback Faithfulness & SUS" never matched `elif page == "Feedback & SUS"`, so the page never rendered. | Names unified. |
| 13 | SUS and feedback ratings were never saved. SUS sliders were pre-filled with favourable answers. | New DB tables; radios with no default; saved to the database. |
| 14 | Prioritised action in `feedback.py` quoted the **last loop item**, not the weakest dimension. | Fixed and tested. |
| 15 | Spoken audio was re-transcribed on every Streamlit rerun; participants could edit the transcript (hiding ASR error from scoring); raw ASR text was not stored. | Transcribe once; raw ASR stored in `answers.asr_transcript`; editing off by default (`study.allow_transcript_editing`). |
| 16 | Rater ID defaulted to `RATER_A`, so two raters could overwrite each other; sliders pre-set to 3; same answer order for everyone; sliders shared across answers. | Empty default, no pre-selected score, per-rater random order, per-answer widgets, "score every dimension" check. |
| 17 | Model/user text was injected into HTML with `unsafe_allow_html`. | Escaped. |
| 18 | The Delivery dimension could be requested for spoken answers, but S1 only sees the transcript (no acoustic features) and raters cannot hear audio. | Off by default (`scoring.include_delivery: false`) and documented as not validated. |
| 19 | Ollama "model_version" logged a timestamp, not the model. | Logs the model digest from `/api/tags`; a seed is passed. |
| 20 | Extra dimensions returned by the LLM were averaged into the overall score. | Only requested dimensions are used. |

## D. Missing files that were added
`requirements.txt`, `.gitignore`, `LICENSE`, `eval/common.py`, `eval/export.py`, `eval/score_all.py`, `data/questions.csv`,
CSV templates, labelled synthetic demo data, `tests/test_pipeline.py`, `docs/consent_form.md`, `docs/rater_guide.md`.

## E. What has NOT been verified (please test on your own machine)
- Real runs against **Ollama** or an API model (the sandbox used for development has no model server). The prompt, JSON parsing and logging are tested with scripted and mock backends only.
- **faster-whisper** transcription of real audio.
- **sentence-transformers** (B2). It was not installable offline in the test environment, so B2 was exercised only through its "unavailable -> NaN" path.
- Small local models (e.g. llama3 8B) may not follow the JSON format reliably. If many answers fail validation, try a larger model or an API model.

## F. Known limitations that remain
- With temperature 0 and one model, **S2 equals S1** (identical repeated runs). S2 is only meaningful with temperature > 0 or two different models.
- ICC (absolute agreement) penalises a constant offset: a system that is consistently 0.5 points too generous scores a low ICC even with a high correlation. Read it together with the Bland-Altman bias.
- 10 answers is the minimum for the verdict code, not a sensible study size. Aim for 150 or more.
