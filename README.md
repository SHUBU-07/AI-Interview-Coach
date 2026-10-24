# AI-Interview-Coach
Interview practice that shows its work. Our scorer is checked against trained human raters and tested for bias, so students get feedback they can trust.
AI Interview Coach
> Interview practice that shows its work. Our scorer is checked against trained human raters and tested for bias, so students get feedback they can trust.
A group mini project built by students, for students, at [College Name], Department of [Department], 2026.
---
What is this?
AI Interview Coach lets you practise interview questions by typing or speaking your answer. It scores the answer on a simple 1-5 rubric (relevance, depth, structure, language) and gives feedback that quotes your own words, so you can see exactly why you got the score.
Many interview apps give you a number and expect you to trust it. We wanted to find out whether that trust is deserved. So this project does three things most tools don't:
Compares the AI with human raters. Trained people score the same answers independently, and we measure how close the AI gets relative to how closely the humans agree with each other.
Checks for bias. We test whether scores change when only a stated name, gender, region or religion changes, and whether speech-to-text makes more mistakes for some accents.
Reports results honestly, including the weak ones, with confidence intervals and sample sizes.
> **Note:** Scores are advisory practice feedback. They are not hiring decisions.
Why not just use an existing app?
Most interview tools show a score but never show that it matches what a human interviewer would say, and rarely test for bias. Our literature survey of 24 papers found that many prototypes report no results, report numbers with no definition (for example "91% accuracy" with no dataset), or never compare against human raters. Integrated systems (resume + voice + feedback dashboards) already exist, so the architecture is not our novelty. The novelty we aim for is a validated and audited scoring component:
Trust: scores validated against human raters, with numbers
Fairness: tested across background and accent, not just claimed
Explainable: every comment quotes the candidate's actual answer
Privacy: can run on a locally hosted model, so data stays on your machine
Open method: rubric, prompts and evaluation code are all in this repo
We did not find a peer-reviewed study doing exactly this in our search. That is a statement about our search, not proof that none exists.
Features
Practice page: answer by text or voice, get rubric scores and evidence-linked feedback
History page: see progress across sessions
Blind rater page: raters score answers independently, without seeing other raters' scores or the AI score
Evaluation dashboard: agreement tables, fairness results and plots
Pluggable LLM backend: local model through Ollama (default) or an API model
Baselines for comparison: TF-IDF, sentence embeddings, and a plain "score out of 10" LLM prompt
Privacy tools: consent checkbox, random participant IDs, delete-my-data button
How it works
```
Answer (text or voice)
      |
      v
 [Whisper] -> transcript
      |
      v
 Scorers  ->  B1 TF-IDF | B2 Embeddings | B3 Plain LLM | S1 Rubric LLM | S2 Ensemble
      |
      v
 Evidence-linked feedback  ----->  Student
      |
      v
 Evaluation vs. human raters + fairness audit  ----->  results.md
```
S1 (our proposed scorer) scores one answer per call at the lowest randomness setting, using written level descriptors for each rubric dimension. It returns structured output with an exact quote from the answer as evidence, and the code checks that the quote really appears in the answer.
The rubric
Each dimension is scored 1-5, with written descriptors for the levels (see `rubric.yaml`).
Dimension	What it looks at
Relevance and correctness	Does the answer address the question accurately?
Depth and completeness	Does it give enough detail and examples?
Structure and clarity	Is it organised (STAR for behavioural questions)?
Language use	Is the wording clear and appropriate?
(optional) Delivery	Pace, pauses, fillers. Only when audio is available
Project structure
```
ai_interview_coach/
  app/                  Streamlit UI (Practice, History, Rater, Dashboard)
  coach/
    rubric.py           Loads rubric.yaml
    scorer_llm.py       S1 proposed scorer
    baselines.py        B1, B2, B3 baselines
    ensemble.py         S2 repeated runs / two models
    feedback.py         Evidence-linked feedback
    asr.py              Whisper wrapper
  eval/
    agreement.py        Human-human and system-human metrics
    reliability.py      Test-retest and prompt sensitivity
    fairness.py         Counterfactual, WER by group, group errors
    report.py           Builds results tables, figures, results.md
  data/
    questions.csv
    answers_template.csv
    human_ratings_template.csv
    demo/               DEMO ONLY data to test the pipeline
  rubric.yaml
  config.yaml
  README.md
```
Getting started
Requirements
Python 3.10 or newer
Ollama (for the default local model), or an API key for a hosted model
Optional: a microphone for voice answers
Install
```bash
git clone https://github.com/<your-org>/ai-interview-coach.git
cd ai-interview-coach
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Set up the model
```bash
# Local model (default)
ollama pull llama3

# Or use an API model
export LLM_API_KEY="your-key-here"
```
Choose the backend in `config.yaml`. The model name, version and settings are saved with every score so results can be reproduced.
Run the app
```bash
streamlit run app/main.py
```
Running the study
The code is only half the project. The results come from real people.
Get approval. Check with your guide or HOD whether ethics or committee approval is needed.
Recruit participants. About 20-30 students, with variety in first language, region and experience. Aim for about 150 answers in total, and at least 10 answers per group you want to compare.
Collect consent. Use the consent checkbox in the app. No names are stored.
Recruit 2-3 raters (placement-experienced seniors, staff or faculty, not close friends of participants and not the people who built the system).
Train raters (about 1 hour). Everyone scores the same 10 practice answers, then discusses differences.
Rate independently. Raters use the Rater page, with randomised order and no access to the AI score.
Export the dataset:
```bash
   python -m eval.export
   ```
Run the evaluation:
```bash
   python -m eval.agreement      # human-human ceiling and system-human agreement
   python -m eval.reliability    # test-retest and prompt sensitivity
   python -m eval.fairness       # counterfactual, WER by group, group errors
   python -m eval.report         # builds results.md
   ```
If there is not enough data, the report says "insufficient data" and does not invent numbers.
Evaluation metrics
Metric	Question it answers
Human-human ICC and quadratic weighted kappa	How reliable are the human scores? (the ceiling)
System-human QWK, ICC, MAE, within-one-point agreement, Pearson, Spearman	How closely does the system track humans?
Bland-Altman bias and limits	Does the system score systematically higher or lower?
Non-inferiority vs. human-human ICC	Is the system as consistent with humans as humans are with each other?
Test-retest ICC, prompt sensitivity	Are scores stable?
Bootstrap 95% confidence intervals	Are differences from the baselines reliable?
Counterfactual score change	Does a stated background change the score?
WER by group, group-wise score error	Are errors uneven across accents or groups?
Feedback faithfulness and actionability, SUS	Is the feedback accurate and useful?
Results
> **Status: pending data collection.**
>
> We will fill this section only with numbers produced by `eval/report.py`.
Method	QWK	ICC	MAE	Within 1 point
Human-human (ceiling)	-	-	-	-
B1 TF-IDF	-	-	-	-
B2 Embeddings	-	-	-	-
B3 Plain LLM	-	-	-	-
S1 Rubric LLM	-	-	-	-
S2 Ensemble	-	-	-	-
Limitations
Small sample from one institution, so results may not generalise
Human raters are imperfect, which is why their agreement is reported as the ceiling
LLM versions change over time, so we record model versions and settings
Group-level fairness conclusions are limited when groups are small; those groups are labelled "too small for inference"
Any pre/post pilot without a control group cannot show cause and effect
The delivery measures (pace, pauses, fillers) are simple proxies and are only validated if human delivery ratings are collected
Ethics and privacy
Informed consent is required before any answer is stored
Participants get random IDs, with no names stored
Audio is not kept longer than needed, and participants can delete their data at any time
Scores are practice feedback only and must not be used for hiring or selection decisions
Our team
Name	Role	Focus
[Member 1]	Backend and scoring pipeline	LLM scorer, baselines
[Member 2]	Frontend and UI	Practice, History, Rater pages
[Member 3]	Data and raters	Consent, collection, rater training
[Member 4]	Evaluation and fairness	Agreement metrics, bias audit
[Prof. Name]	Project guide	Supervision
Key references
Stockdale, Hickman and Liu (2026). Scoring employment interviews with large language models. Journal of Applied Psychology. (Method template: anchored rubrics, one answer at a time, low temperature, human comparison, subgroup analysis.)
Nofal et al. (2025). AI-enhanced interview simulation in the metaverse. Computers and Education: Artificial Intelligence, 8. (Counterfactual fairness test.)
Hoque et al. (2013). MACH: My Automated Conversation coacH. ACM UbiComp. (Controlled evidence that automated feedback improves human-rated interview performance.)
Wen et al. (2026). PolyInterview. arXiv:2607.10310. (Integrated coaching platform; its authors note that score-level validity is not yet established.)
License
Add a license (for example MIT) before making the repo public.
Acknowledgements
Thanks to our guide, the raters and every classmate who volunteered their answers.
