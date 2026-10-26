# AI Interview Coach

> Interview practice that shows its work. Our scorer is checked against trained human raters and tested for bias, so students get feedback they can trust.

A group mini project at **[College Name]**, Department of **[Department]**, 2026.

## What is this?

Students answer interview questions (typed or spoken). The coach scores each answer on a 1-5 rubric (relevance, depth, structure, language) and gives feedback that **quotes the student's own words**.

Many interview apps give a score and ask you to trust it. This project asks whether the trust is deserved:

1. **Human validation.** Trained raters score the same answers independently. We measure how close the AI gets *relative to how closely humans agree with each other*.
2. **Fairness audit.** We test whether scores change when only a stated name, region or institution changes, and whether speech-to-text makes more errors for some accents or languages.
3. **Honest reporting.** Every number in the report is computed from data. If there is not enough data, the report says so.

> Scores are advisory practice feedback. They are not hiring decisions.

### Why not just use an existing app?
Integrated mock-interview platforms already exist, so the architecture is **not** our novelty. In our literature survey of 24 papers, many prototypes reported no results, undefined "accuracy" figures, or no comparison with human raters, and fairness was rarely tested. Our contribution is a **validated and audited scoring component** with an open method. We did not find a peer-reviewed study doing exactly this in our search; that describes our search, not a proof that none exists.

## Quickstart

```bash
git clone https://github.com/<your-org>/ai-interview-coach.git
cd ai-interview-coach
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run the tests** (no model needed):
```bash
python -m unittest discover -s tests -t .
```

**Check the evaluation pipeline on synthetic demo data** (uses a mock LLM; numbers are NOT evidence):
```bash
python run_pipeline.py --demo        # writes results_demo.md and eval/outputs/demo/
```

**Run the app with a real model** (default: local Ollama):
```bash
ollama pull llama3
streamlit run app/main.py            # http://localhost:8501
```
Choose the backend in `config.yaml` (`ollama`, `openai_compatible`; `mock` is for tests only). For an API model set the key first: `export LLM_API_KEY="..."`.

## Running the study

The code is only half of the project. The results come from real people.

1. **Approval.** Ask your guide/HOD whether ethics or committee approval is needed. Use `docs/consent_form.md`.
2. **Pre-register** the non-inferiority margin (`eval.non_inferiority_margin` in `config.yaml`) *before* looking at data.
3. **Recruit participants**: about 20-30 students with varied first languages, regions and experience, giving about 150 answers. You need at least 10 answers per group (`eval.min_group_size_for_inference`) for any group-level conclusion.
4. **Recruit 2-3 raters** (placement-experienced seniors or staff; not close friends of participants and not the system builders). Run the calibration session in `docs/rater_guide.md`.
5. **Collect answers** in the app (Practice Session). Raters then use **Blind Rater Portal** with their own unique Rater IDs.
6. **Optional second task** for raters: **Feedback & SUS** to audit feedback faithfulness. Participants fill in the SUS questionnaire.
7. **ASR audit (optional):** manually correct the transcripts of ~30 spoken answers into `data/transcription_subset.csv` (see `data/transcription_subset_template.csv`).
8. **Evaluate:**
   ```bash
   python run_pipeline.py --rescore
   ```
   `--rescore` re-scores every answer with one fixed model and prompt, exports the database, runs all stages and writes `results.md`.

Stages can also be run one at a time: `python -m eval.score_all`, `eval.export`, `eval.agreement`, `eval.reliability`, `eval.fairness`, `eval.report`.

## How scoring works

| Method | What it is |
|---|---|
| B1 | TF-IDF similarity to a reference answer, mapped to 1-5 |
| B2 | Sentence-embedding similarity to a reference answer, mapped to 1-5 |
| B3 | Plain LLM prompt "score out of 10", rescaled to 1-5 |
| **S1 (proposed)** | LLM prompted with the rubric's written level descriptors; one answer per call; temperature 0; structured JSON with a verbatim evidence quote per dimension (the code checks the quote really appears in the answer) |
| S2 | S1 repeated and aggregated (only differs from S1 if temperature > 0 or two models are used) |

Failed or unparsable outputs are recorded as invalid and **excluded** from the evaluation. They are never replaced by a default score.

## What the evaluation reports

- **Human ceiling:** mean pairwise single-rater ICC(2,1) and quadratic weighted kappa between raters
- **System vs human:** ICC, QWK, MAE, exact and within-1-point agreement, Pearson, Spearman, Bland-Altman bias, bootstrap 95% CIs. `ICC_vs_Single_Rater` is the like-for-like comparison with the ceiling
- **Non-inferiority of S1** with a bootstrap CI: `non_inferior`, `inconclusive` or `inferior`
- **Reliability:** test-retest and prompt-sensitivity
- **Fairness:** counterfactual identity test; WER by language group and its link to score error; system-minus-human error by group (groups under 10 answers are labelled "too small for inference")

## Project structure

```
app/main.py                 Streamlit app: Practice, History, Blind Rater, Feedback & SUS, Dashboard
coach/                      rubric, S1/S2 scorers, baselines, feedback, ASR, DB, LLM backends
eval/                       common, export, score_all, agreement, reliability, fairness, report
data/                       questions.csv, CSV templates, demo/ (synthetic, labelled DEMO ONLY)
docs/                       consent_form.md, rater_guide.md, CHANGES.md
tests/                      metric tests + regression tests
rubric.yaml  config.yaml    rubric and all settings
run_pipeline.py             runs the whole evaluation
results.md                  generated study report (placeholder until you run the study)
```

## Limitations

- Small sample from one institution; results may not generalise
- Human raters are imperfect (reported as the ceiling)
- Results apply only to the recorded model, version and prompt
- **Delivery (pace, pauses, fillers) is not implemented or validated.** It is off by default because S1 sees only the transcript
- A pre/post pilot without a control group cannot show cause and effect
- See `docs/CHANGES.md` for what has been tested and what you should still verify on your own machine (real Ollama/Whisper/embedding runs)

## Ethics and privacy

Informed consent is required. Participants get random IDs and no names are stored. Participants can delete their data in the app. Raw answers, audio and ratings are git-ignored; do not publish them unless your consent form allows it.

## Our team

| Name | Role | Focus |
|---|---|---|
| [Member 1] | Backend and scoring | S1/S2, baselines, LLM backends |
| [Member 2] | Frontend | Streamlit pages |
| [Member 3] | Data and raters | Consent, collection, rater training |
| [Member 4] | Evaluation and fairness | Agreement metrics, bias audit |
| [Prof. Name] | Guide | Supervision |

## License
MIT. See `LICENSE`.
