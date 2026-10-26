"""
AI Interview Coach - Streamlit Application
Pages:
1. Practice (Answer, Rubric-anchored scoring & evidence-linked feedback)
2. History (Participant performance trajectory)
3. Rater (Blind independent human rater scoring)
4. Feedback & SUS (Faithfulness, actionability, and System Usability Scale)
5. Evaluation Dashboard (Empirical validation tables & fairness audit plots)
"""

from datetime import datetime
import html
import json
import os
import random
import subprocess
import uuid
from pathlib import Path
import tempfile
import yaml

import pandas as pd
import streamlit as st

# Setup paths and imports
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from coach.rubric import Rubric
from coach.db import CoachDB, HumanRating
from coach.llm_backend import get_llm_backend
from coach.scorer_llm import S1Scorer, S1ScoringResult, DimensionScore
from coach.baselines import B1TfidfScorer, B2EmbeddingScorer, B3UnconstrainedLLMScorer
from coach.ensemble import S2EnsembleScorer
from coach.feedback import FeedbackGenerator, SUSQuestionnaire
from coach.asr import ASRProcessor


# --- Application Configuration ---

st.set_page_config(
    page_title="AI Interview Coach | Validated & Fairness-Audited",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .advisory-badge {
        background-color: #fef3c7;
        color: #92400e;
        padding: 12px 18px;
        border-radius: 8px;
        border-left: 5px solid #f59e0b;
        margin-bottom: 20px;
        font-weight: 500;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 8px;
        text-align: center;
    }
    .evidence-quote {
        background-color: #f1f5f9;
        border-left: 3px solid #3b82f6;
        padding: 6px 12px;
        font-style: italic;
        margin: 6px 0;
        font-size: 0.95em;
    }
    .dimension-box {
        border: 1px solid #e2e8f0;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_app_context():
    cfg_path = BASE_DIR / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rubric_path = BASE_DIR / config["paths"]["rubric"]
    rubric = Rubric.load_from_yaml(rubric_path)

    db_path = BASE_DIR / config["paths"]["database"]
    db = CoachDB(db_path)

    llm = get_llm_backend(config)
    s1 = S1Scorer(rubric=rubric, llm_backend=llm, db=db)
    b1 = B1TfidfScorer(db=db)
    b2 = B2EmbeddingScorer(model_name=config["baselines"]["embedding_model"], db=db)
    b3 = B3UnconstrainedLLMScorer(llm_backend=llm, db=db)
    s2 = S2EnsembleScorer(s1_scorers=[s1], aggregation=config["ensemble"]["aggregation"], db=db)
    feedback_gen = FeedbackGenerator(rubric=rubric)

    questions_df = pd.read_csv(BASE_DIR / config["paths"]["questions_csv"])
    asr = ASRProcessor(
        model_size=config["asr"]["model_size"],
        device=config["asr"]["device"],
        compute_type=config["asr"]["compute_type"],
        vad_filter=config["asr"]["vad_filter"],
    )

    return config, rubric, db, s1, b1, b2, b3, s2, feedback_gen, questions_df, asr

config, rubric, db, s1_scorer, b1_scorer, b2_scorer, b3_scorer, s2_scorer, feedback_gen, questions_df, asr = load_app_context()

# NOT VALIDATED: S1 only sees the transcript, so a Delivery score would be guessed from text. Keep off.
INCLUDE_DELIVERY = bool(config.get('scoring', {}).get('include_delivery', False))
ALLOW_EDIT = bool(config.get('study', {}).get('allow_transcript_editing', False))
KEEP_AUDIO = bool(config.get('privacy', {}).get('keep_audio', True))


def fmt_score(x) -> str:
    return 'n/a' if x is None or x != x else f'{x:.2f} / 5.0'


# --- Sidebar Navigation ---

st.sidebar.title("🎯 AI Interview Coach")
st.sidebar.caption("Human-Validated & Fairness-Audited Rubric Evaluation")

page = st.sidebar.radio(
    "Navigation",
    ["Practice Session", "My History", "Blind Rater Portal", "Feedback & SUS", "Evaluation & Fairness Dashboard"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Backend Model:** `" + str(config["llm"]["model"]) + "`")
st.sidebar.markdown("**LLM Temperature:** `" + str(config["llm"]["temperature"]) + "`")
if config["llm"]["backend"] == "mock":
    st.sidebar.error("MOCK backend active: scores are NOT real. Use only for testing.")


# ==============================================================================
# PAGE 1: PRACTICE SESSION
# ==============================================================================

if page == "Practice Session":
    st.title("Interview Practice & Rubric-Anchored Feedback")

    st.markdown("""
    <div class="advisory-badge">
        ⚠️ <strong>Formative Advisory Notice:</strong> All scores and feedback are generated strictly for practice and personal development.
        This system is NOT used for screening, hiring, or gatekeeping decisions.
    </div>
    """, unsafe_allow_html=True)

    # Participant setup & informed consent
    with st.expander("👤 Participant Profile & Informed Consent (Required)", expanded="participant_id" not in st.session_state):
        consent = st.checkbox(
            "I consent to take part in this interview-practice research study. I understand that my answers (and voice recordings, if I use spoken mode) are stored under a random ID without my name, are used only for this study, and can be deleted at any time with the 'Delete My Data' button. Scores are practice feedback only, not a judgement of me and not a hiring decision.",
            value=st.session_state.get("consent_given", False)
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            first_lang = st.selectbox("First / Primary Language:", ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Kannada", "Bengali", "Gujarati", "Urdu", "Other"])
        with col2:
            region = st.selectbox("Geographic Region:", ["Maharashtra", "Other West India", "South India", "North India", "East India", "Outside India", "Other"])
        with col3:
            gender = st.selectbox("Gender (Optional):", ["Prefer not to say", "Female", "Male", "Non-Binary", "Other"])

        if st.button("Register & Save Profile"):
            if not consent:
                st.error("You must check the informed consent box to proceed.")
            else:
                pid = db.create_participant(consent, first_lang, region, gender)
                st.session_state["participant_id"] = pid
                st.session_state["consent_given"] = True
                st.success(f"Participant Registered: ID `{pid}`. Data is de-identified.")

    if not st.session_state.get("consent_given"):
        st.info("Please complete the consent form above to begin your practice session.")
        st.stop()

    pid = st.session_state["participant_id"]
    st.write(f"Active Candidate: **`{pid}`**")

    # Question Selection
    q_options = {f"{r['question_id']}: {r['question_text']}": r['question_id'] for _, r in questions_df.iterrows()}
    selected_label = st.selectbox("Select Interview Question:", list(q_options.keys()))
    qid = q_options[selected_label]
    selected_row = questions_df[questions_df["question_id"] == qid].iloc[0]

    st.info(f"**Question Category:** {selected_row['category']} | **Expected Progression:** {selected_row['expected_structure']}")

    # Input Mode
    input_mode = st.radio("Response Mode:", ["Typed Answer", "Spoken Audio (File Upload)"], horizontal=True)
    transcript_input = ""
    audio_path = None
    asr_raw = None

    if input_mode == "Typed Answer":
        transcript_input = st.text_area(
            "Type your interview response here (aim for a structured, thorough answer):",
            height=180,
            placeholder="Structure your answer clearly, state concrete actions and results..."
        )
    else:
        audio_file = st.file_uploader("Upload Audio Response (.wav, .mp3, .m4a, .webm):", type=["wav", "mp3", "m4a", "webm"])
        if audio_file:
            file_key = f"{audio_file.name}_{audio_file.size}"
            if st.session_state.get("asr_key") != file_key:  # transcribe once, not on every Streamlit rerun
                suffix = Path(audio_file.name).suffix
                if KEEP_AUDIO:
                    audio_dir = BASE_DIR / config["privacy"]["audio_dir"]
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    saved_path = audio_dir / f"{pid}_{uuid.uuid4().hex[:8]}{suffix}"
                else:
                    saved_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}{suffix}"
                saved_path.write_bytes(audio_file.getvalue())
                with st.spinner("Transcribing spoken audio via faster-whisper..."):
                    result = asr.transcribe(saved_path)
                if not KEEP_AUDIO:
                    saved_path.unlink(missing_ok=True)
                st.session_state.update(asr_key=file_key, asr_result=result,
                                        asr_audio_path=str(saved_path) if KEEP_AUDIO else None)
            trans_res = st.session_state["asr_result"]
            audio_path = st.session_state["asr_audio_path"]
            if trans_res.success:
                asr_raw = trans_res.transcript
                st.success(f"Audio transcribed ({trans_res.duration_seconds}s, detected language: {trans_res.detected_language})")
                transcript_input = st.text_area(
                    "Transcribed speech:" + ("" if ALLOW_EDIT else " (read-only: your answer is scored exactly as transcribed)"),
                    value=asr_raw, height=140, disabled=not ALLOW_EDIT, key=f"tx_{file_key}")
            else:
                st.error(f"Speech transcription failed: {trans_res.error_message}")

    # Action Buttons
    col_submit, col_delete = st.columns([4, 1])
    with col_submit:
        submit_btn = st.button("Submit & Evaluate Answer", type="primary")

    with col_delete:
        if st.button("Delete My Data"):
            db.delete_participant_data(pid)
            st.session_state.clear()
            st.warning("All your participant records, answers, and audio files have been permanently purged.")
            st.rerun()

    if submit_btn:
        if not transcript_input.strip():
            st.error("Please enter a response or upload audio before submitting.")
            st.stop()

        with st.spinner("Evaluating response against rubric anchors and computing baseline metrics..."):
            # 1. Save Answer to DB
            aid = db.save_answer(
                participant_id=pid,
                question_id=qid,
                transcript=transcript_input,
                audio_path=audio_path,
                input_mode="spoken" if input_mode.startswith("Spoken") else "typed",
                asr_transcript=asr_raw,
            )

            # 2. Run S1 (Proposed Rubric LLM Scorer)
            s1_res = s1_scorer.score_answer(
                answer_id=aid,
                question=selected_row["question_text"],
                transcript=transcript_input,
                has_audio=INCLUDE_DELIVERY and bool(audio_path)
            )

            # 3. Run Baselines (B1, B2, B3) & S2
            ref_ans = str(selected_row.get("reference_answer", ""))
            b1_res = b1_scorer.score_answer(aid, transcript_input, ref_ans)
            b2_res = b2_scorer.score_answer(aid, transcript_input, ref_ans)
            b3_res = b3_scorer.score_answer(aid, selected_row["question_text"], transcript_input)
            s2_res = s2_scorer.score_answer(aid, selected_row["question_text"], transcript_input,
                                           has_audio=INCLUDE_DELIVERY and bool(audio_path),
                                           num_runs=int(config["ensemble"]["num_runs"]))

            # 4. Generate Evidence-Linked Feedback
            feedback = feedback_gen.generate_feedback(s1_res, selected_row["question_text"], transcript_input, ref_ans)

        if not s1_res.validation_passed:
            st.error("The scoring model did not return a valid rubric evaluation (is Ollama running and the model pulled?). "
                     "No score was produced for this answer, so none is shown. Please try again.")
            st.stop()

        st.success(f"Evaluation Complete! Saved as Answer `{aid}`")

        # Display Top Summary Score
        st.markdown("### 📊 Performance Summary")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("S1 Rubric Score", fmt_score(s1_res.overall_score))
        m_col2.metric("S2 Ensemble Score", fmt_score(s2_res.overall_score if s2_res.validation_passed else None))
        m_col3.metric("B2 Embedding Match", fmt_score(b2_res.overall_score))
        m_col4.metric("B3 Naive LLM", fmt_score(b3_res.overall_score))

        # Rubric Dimension Breakdown
        st.markdown("### 📋 Rubric Dimension Breakdown (S1)")
        for dim_id, d_score in s1_res.dimension_scores.items():
            dim_def = rubric.dimensions.get(dim_id)
            dim_name = dim_def.name if dim_def else dim_id
            st.markdown(f"""
            <div class="dimension-box">
                <h4>{html.escape(dim_name)}: Level {d_score.score} / 5</h4>
                <div class="evidence-quote"><strong>Evidence Quote:</strong> "{html.escape(d_score.evidence_quote)}"</div>
                <p><strong>Rationale:</strong> {html.escape(d_score.reason)}</p>
                <small>Grounding Status: {'✅ Verified in Transcript' if d_score.quote_grounded else '⚠️ Quote not strictly found'}</small>
            </div>
            """, unsafe_allow_html=True)

        # Actionable Feedback
        st.markdown("### 💡 Evidence-Linked Feedback")
        st.warning(f"🎯 **{feedback.prioritized_action}**")

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("#### ✅ Demonstrated Strengths")
            if feedback.strengths:
                for s in feedback.strengths:
                    st.write(f"- **{s.dimension} (Level {s.rubric_level})**: {s.claim_text}")
                    st.caption(f"Evidence: {s.evidence_quote}")
            else:
                st.write("Keep practicing to elevate your response to the Level 4-5 anchors.")

        with f_col2:
            st.markdown("#### 🔍 Development Opportunities")
            if feedback.gaps:
                for g in feedback.gaps:
                    st.write(f"- **{g.dimension}**: {g.claim_text}")
                    st.caption(f"Evidence: {g.evidence_quote}")
            else:
                st.write("All dimensions performed at or above Level 4!")

        # High-Scoring Benchmark Outline
        with st.expander("📖 Recommended Response Benchmark Outline", expanded=False):
            st.markdown(feedback.model_answer_outline)


# ==============================================================================
# PAGE 2: MY HISTORY
# ==============================================================================

elif page == "My History":
    st.title("Candidate Practice History")
    pid = st.session_state.get("participant_id")
    if not pid:
        st.info("No active participant session. Complete a practice session to see history.")
        st.stop()

    answers = db.get_all_answers()
    my_answers = [a for a in answers if a["participant_id"] == pid]
    if not my_answers:
        st.info("No answers submitted under this session yet.")
    else:
        df_my = pd.DataFrame(my_answers)
        st.write(f"Total Answers Recorded: **{len(df_my)}**")
        st.dataframe(df_my[["answer_id", "question_id", "input_mode", "timestamp", "transcript"]])


# ==============================================================================
# PAGE 3: BLIND RATER PORTAL
# ==============================================================================

elif page == "Blind Rater Portal":
    st.title("Blind Human Rater Evaluation Portal")
    st.markdown("""
    **Rater Protocol**: You will score answers independently against the rubric level anchors.
    You have **zero visibility** of other human raters' scores and **zero visibility** of the automated system's scores.
    """)

    rater_id = st.text_input("Enter Rater ID (e.g. RATER_A, RATER_B):", value=st.session_state.get("rater_id", ""))
    st.session_state["rater_id"] = rater_id.strip()
    rater_id = rater_id.strip()
    if not rater_id:
        st.info("Enter your own unique Rater ID to begin (two raters must never share an ID).")
        st.stop()

    answers = db.get_all_answers()
    if not answers:
        st.warning("No answers available in the database to rate yet.")
        st.stop()

    rated_answer_ids = {r["answer_id"] for r in db.get_all_human_ratings() if r["rater_id"] == rater_id}
    ordered = sorted(answers, key=lambda x: x["answer_id"])
    random.Random(rater_id).shuffle(ordered)  # stable, rater-specific random order
    show_rated = st.checkbox("Also show answers I have already rated (to edit)", value=False)
    todo = ordered if show_rated else [x for x in ordered if x["answer_id"] not in rated_answer_ids]
    st.write(f"Total answers: **{len(answers)}** | Rated by {rater_id}: **{len(rated_answer_ids)}** | Remaining: **{len(answers) - len(rated_answer_ids)}**")
    if not todo:
        st.success("You have rated every available answer.")
        st.stop()
    a_select = st.selectbox("Select Answer to Rate:", [f"{x['answer_id']} (Question: {x['question_id']})" for x in todo])
    selected_aid = a_select.split(" ")[0]
    current_answer = db.get_answer(selected_aid)
    q_info = questions_df[questions_df["question_id"] == current_answer["question_id"]].iloc[0]

    st.markdown("---")
    st.markdown(f"### Question: {q_info['question_text']}")
    st.markdown(f"**Category:** {q_info['category']}")
    st.markdown("#### Candidate Transcript:")
    st.info(current_answer["transcript"])

    st.markdown("### Rubric Evaluation Form")
    score_inputs = {}
    dims = rubric.get_eval_dimensions(has_audio=INCLUDE_DELIVERY and bool(current_answer.get("audio_path")))

    for dim in dims:
        st.markdown(f"**{dim.name}**")
        with st.expander(f"View Anchors for {dim.name}", expanded=False):
            for lvl, desc in sorted(dim.anchors.items()):
                st.write(f"- **Level {lvl}:** {desc}")
        score_inputs[dim.id] = st.radio(f"Score for {dim.name}:", [1, 2, 3, 4, 5], index=None, horizontal=True,
                                        key=f"rate_{selected_aid}_{rater_id}_{dim.id}")

    notes = st.text_area("Rater Evidence Notes / Justification:", placeholder="Note specific quotes or observations that influenced the score...")

    if st.button("Save Blind Rating", type="primary"):
        if any(v is None for v in score_inputs.values()):
            st.error("Please choose a score for every dimension before saving.")
            st.stop()
        hr = HumanRating(
            rating_id=f"HR_{selected_aid}_{rater_id}",
            rater_id=rater_id,
            answer_id=selected_aid,
            relevance_correctness=score_inputs.get("relevance_correctness", 3),
            depth_completeness=score_inputs.get("depth_completeness", 3),
            structure_clarity=score_inputs.get("structure_clarity", 3),
            language_use=score_inputs.get("language_use", 3),
            delivery=score_inputs.get("delivery"),
            evidence_notes=notes,
        )
        db.save_human_rating(hr)
        st.success(f"Rating saved successfully for {selected_aid}! Overall: {hr.overall_score}/5.0")


# ==============================================================================
# PAGE 4: FEEDBACK & SUS QUESTIONNAIRE
# ==============================================================================

elif page == "Feedback & SUS":
    st.title("Feedback Quality Audit & System Usability (SUS)")
    tab_fb, tab_sus = st.tabs(["Rater: is the feedback faithful?", "Participant: SUS questionnaire"])

    with tab_fb:
        st.caption("For raters. You can only audit feedback for answers you have ALREADY rated blind, so this never reveals the AI score before you rate.")
        rid = st.text_input("Your Rater ID:", value=st.session_state.get("rater_id", ""), key="fb_rater").strip()
        my_rated = {r["answer_id"] for r in db.get_all_human_ratings() if r["rater_id"] == rid} if rid else set()
        candidates = [x for x in db.get_all_answers() if x["answer_id"] in my_rated and db.get_latest_system_score(x["answer_id"], "S1")]
        if not rid:
            st.info("Enter your Rater ID.")
        elif not candidates:
            st.info("No answers available: rate answers in the Blind Rater Portal first.")
        else:
            pick = st.selectbox("Answer:", [x["answer_id"] for x in candidates])
            ans = db.get_answer(pick)
            rec = db.get_latest_system_score(pick, "S1")
            payload = json.loads(rec["scores_json"])
            if not payload.get("validation_passed", True):
                st.warning("The AI scoring for this answer failed validation, so there is no feedback to audit.")
            else:
                dims = {k: DimensionScore(dimension=k, score=int(v["score"]), evidence_quote=v["evidence_quote"],
                                          reason=v["reason"], quote_grounded=bool(v.get("quote_grounded", True)))
                        for k, v in payload["dimensions"].items()}
                s1_res = S1ScoringResult(answer_id=pick, overall_score=payload["overall"], dimension_scores=dims, raw_output="",
                                         model_name=rec["model_name"], model_version=rec["model_version"], temperature=rec["temperature"],
                                         prompt_version=rec["prompt_version"], prompt_hash=rec["prompt_hash"],
                                         latency_ms=rec["latency_ms"], validation_passed=True)
                q_row = questions_df[questions_df["question_id"] == ans["question_id"]].iloc[0]
                fb = feedback_gen.generate_feedback(s1_res, q_row["question_text"], ans["transcript"], str(q_row.get("reference_answer", "")))
                st.markdown(f"**Question:** {q_row['question_text']}")
                st.info(ans["transcript"])
                st.markdown("**Tick each feedback claim that is actually supported by the candidate's answer:**")
                claims = list(fb.strengths) + list(fb.gaps)
                supported = 0
                for i, c in enumerate(claims):
                    ok = st.checkbox(f"[{c.claim_type}] {c.dimension}: {c.claim_text}  \n*Evidence shown: {c.evidence_quote}*",
                                     key=f"fb_{pick}_{rid}_{i}")
                    supported += int(ok)
                act = st.radio("How actionable is the prioritised guidance? (1 = vague, 5 = specific and usable)", [1, 2, 3, 4, 5],
                               index=None, horizontal=True, key=f"act_{pick}_{rid}")
                notes = st.text_area("Notes (optional):", key=f"fbnotes_{pick}_{rid}")
                if st.button("Save feedback evaluation", type="primary"):
                    if act is None or not claims:
                        st.error("Select an actionability rating (and make sure there are claims to evaluate).")
                    else:
                        db.save_feedback_evaluation(rid, pick, supported, len(claims), int(act), notes)
                        st.success(f"Saved: {supported}/{len(claims)} claims supported, actionability {act}/5.")

    with tab_sus:
        st.caption("Standard 10-item System Usability Scale (Brooke, 1996). 1 = Strongly disagree, 5 = Strongly agree. No answers are pre-selected.")
        items = [
            "I think that I would like to use this system frequently.",
            "I found the system unnecessarily complex.",
            "I thought the system was easy to use.",
            "I think that I would need the support of a technical person to be able to use this system.",
            "I found the various functions in this system were well integrated.",
            "I thought there was too much inconsistency in this system.",
            "I would imagine that most people would learn to use this system very quickly.",
            "I found the system very cumbersome to use.",
            "I felt very confident using the system.",
            "I needed to learn a lot of things before I could get going with this system.",
        ]
        sus_pid = st.session_state.get("participant_id")
        if not sus_pid:
            st.info("Register and complete a practice session first so your answers can be linked to your anonymous ID.")
        else:
            vals = [st.radio(f"{i + 1}. {t}", [1, 2, 3, 4, 5], index=None, horizontal=True, key=f"sus_{i}") for i, t in enumerate(items)]
            if st.button("Submit SUS", type="primary"):
                if any(v is None for v in vals):
                    st.error("Please answer all 10 items.")
                else:
                    q = SUSQuestionnaire(participant_id=sus_pid, q1_frequent_use=vals[0], q2_complex=vals[1], q3_easy_to_use=vals[2],
                                         q4_technical_support=vals[3], q5_integrated_functions=vals[4], q6_inconsistency=vals[5],
                                         q7_learn_quickly=vals[6], q8_cumbersome=vals[7], q9_confident=vals[8], q10_learning_overhead=vals[9])
                    score = q.calculate_sus_score()
                    db.save_sus_response(sus_pid, [int(v) for v in vals], score)
                    st.success(f"Thank you. Your SUS score: {score:.1f} / 100 (68 is the usual average benchmark).")


# ==============================================================================
# PAGE 5: EVALUATION & FAIRNESS DASHBOARD
# ==============================================================================

elif page == "Evaluation & Fairness Dashboard":
    st.title("Empirical Agreement & Fairness Audit Dashboard")
    st.markdown("Shows results computed by the evaluation scripts from the data files. It never invents or extrapolates values.")

    view = st.radio("Dataset", ["Study data", "Demo (pipeline test only)"], horizontal=True)
    demo = view.startswith("Demo")
    base_out = BASE_DIR / config["paths"]["eval_output_dir"]
    out_dir = base_out / "demo" if demo else base_out
    results_md_path = BASE_DIR / ("results_demo.md" if demo else "results.md")
    if demo:
        st.warning("DEMO: synthetic data and a mock LLM. These numbers only test that the pipeline runs. They are NOT evidence.")

    c1, c2 = st.columns(2)
    run_label = "Run demo pipeline" if demo else "Run study pipeline (export + evaluate)"
    if c1.button(run_label, type="primary"):
        cmd = [sys.executable, "run_pipeline.py"] + (["--demo"] if demo else [])
        with st.spinner("Running evaluation pipeline..."):
            proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
        (st.success if proc.returncode == 0 else st.error)("Pipeline finished." if proc.returncode == 0 else "Pipeline failed; see log below.")
        with st.expander("Pipeline log"):
            st.code((proc.stdout or "") + (proc.stderr or ""))

    if not demo:
        n_ans, n_rat = len(db.get_all_answers()), db.get_all_human_ratings()
        sus = db.get_all_sus_responses()
        fbe = db.get_all_feedback_evaluations()
        m = st.columns(5)
        m[0].metric("Answers", n_ans)
        m[1].metric("Human ratings", len(n_rat))
        m[2].metric("Raters", len({r["rater_id"] for r in n_rat}))
        m[3].metric("SUS responses", len(sus), f"mean {sum(x['sus_score'] for x in sus) / len(sus):.1f}" if sus else None)
        m[4].metric("Feedback audits", len(fbe),
                    f"{sum(x['claims_supported'] for x in fbe) / max(1, sum(x['claims_total'] for x in fbe)):.0%} claims supported" if fbe else None)

    def show_json(title, name):
        path = out_dir / name
        if path.exists():
            st.markdown(f"#### {title}")
            st.json(json.loads(path.read_text(encoding="utf-8")))

    show_json("Human-human reliability (ceiling)", "human_ceiling.json")
    st.markdown("### System vs human agreement")
    agr = out_dir / "system_human_agreement_table.csv"
    st.dataframe(pd.read_csv(agr), width="stretch") if agr.exists() else st.info("Not generated yet (or insufficient data).")
    show_json("Non-inferiority of S1", "non_inferiority.json")

    p1, p2 = st.columns(2)
    if (out_dir / "bland_altman_s1.png").exists():
        p1.image(str(out_dir / "bland_altman_s1.png"), caption="Bland-Altman: S1 vs human mean")
    if (out_dir / "agreement_comparison.png").exists():
        p2.image(str(out_dir / "agreement_comparison.png"), caption="Agreement by method vs human ceiling")

    st.markdown("### Fairness audits")
    for title, name in [("Counterfactual identity test", "fairness_counterfactual_audit.csv"),
                        ("Transcription error by language group", "fairness_transcription_audit.csv"),
                        ("System-minus-human error by first language", "fairness_group_analysis_first_language.csv")]:
        if (out_dir / name).exists():
            st.markdown(f"#### {title}")
            st.dataframe(pd.read_csv(out_dir / name), width="stretch")

    st.markdown(f"### Full report (`{results_md_path.name}`)")
    if results_md_path.exists():
        with st.expander("View full report", expanded=False):
            st.markdown(results_md_path.read_text(encoding="utf-8"))
