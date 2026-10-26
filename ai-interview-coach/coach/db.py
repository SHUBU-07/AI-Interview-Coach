"""
Database module for AI Interview Coach.
Handles participants, answers, blind human ratings, and audited system inference logs.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd


@dataclass
class Participant:
    participant_id: str
    consent_given: bool
    first_language: str = "Unspecified"
    region: str = "Unspecified"
    gender: str = "Unspecified"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


@dataclass
class Answer:
    answer_id: str
    participant_id: str
    question_id: str
    transcript: str
    audio_path: Optional[str] = None
    input_mode: str = "typed"  # 'typed' or 'spoken'
    timestamp: str = ""
    asr_transcript: Optional[str] = None  # raw speech-to-text output (spoken answers only)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


@dataclass
class HumanRating:
    rating_id: str
    rater_id: str
    answer_id: str
    relevance_correctness: int
    depth_completeness: int
    structure_clarity: int
    language_use: int
    delivery: Optional[int] = None
    overall_score: float = 0.0
    evidence_notes: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        if not self.overall_score:
            scores = [
                self.relevance_correctness,
                self.depth_completeness,
                self.structure_clarity,
                self.language_use,
            ]
            if self.delivery is not None:
                scores.append(self.delivery)
            self.overall_score = round(sum(scores) / len(scores), 2)


@dataclass
class SystemScoreRecord:
    score_id: str
    answer_id: str
    method: str  # B1, B2, B3, S1, S2
    model_name: str
    model_version: str
    temperature: float
    prompt_version: str
    prompt_hash: str
    scores_json: str  # JSON dict with dimension scores & overall
    raw_output: str
    latency_ms: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


class CoachDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                consent_given INTEGER NOT NULL,
                first_language TEXT,
                region TEXT,
                gender TEXT,
                created_at TEXT NOT NULL
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                answer_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                transcript TEXT NOT NULL,
                audio_path TEXT,
                asr_transcript TEXT,
                input_mode TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (participant_id) REFERENCES participants(participant_id)
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_ratings (
                rating_id TEXT PRIMARY KEY,
                rater_id TEXT NOT NULL,
                answer_id TEXT NOT NULL,
                relevance_correctness INTEGER NOT NULL,
                depth_completeness INTEGER NOT NULL,
                structure_clarity INTEGER NOT NULL,
                language_use INTEGER NOT NULL,
                delivery INTEGER,
                overall_score REAL NOT NULL,
                evidence_notes TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (answer_id) REFERENCES answers(answer_id),
                UNIQUE(rater_id, answer_id)
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_scores (
                score_id TEXT PRIMARY KEY,
                answer_id TEXT NOT NULL,
                method TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                temperature REAL NOT NULL,
                prompt_version TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                raw_output TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (answer_id) REFERENCES answers(answer_id)
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_evaluations (
                eval_id TEXT PRIMARY KEY,
                rater_id TEXT NOT NULL,
                answer_id TEXT NOT NULL,
                claims_supported INTEGER NOT NULL,
                claims_total INTEGER NOT NULL,
                actionability INTEGER NOT NULL,
                notes TEXT,
                timestamp TEXT NOT NULL,
                UNIQUE(rater_id, answer_id)
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sus_responses (
                sus_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                items_json TEXT NOT NULL,
                sus_score REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                details_json TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """)
            # Lightweight migration for databases created before asr_transcript existed
            cols = [r["name"] for r in cursor.execute("PRAGMA table_info(answers)").fetchall()]
            if "asr_transcript" not in cols:
                cursor.execute("ALTER TABLE answers ADD COLUMN asr_transcript TEXT")
            conn.commit()

    # Participant methods
    def create_participant(self, consent: bool, first_language: str, region: str, gender: str) -> str:
        if not consent:
            raise ValueError("Informed consent is strictly required.")
        pid = f"P_{uuid.uuid4().hex[:8]}"
        p = Participant(
            participant_id=pid,
            consent_given=consent,
            first_language=first_language,
            region=region,
            gender=gender,
        )
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO participants (participant_id, consent_given, first_language, region, gender, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (p.participant_id, 1 if p.consent_given else 0, p.first_language, p.region, p.gender, p.created_at),
            )
            conn.commit()
        return pid

    def delete_participant_data(self, participant_id: str) -> int:
        """Right to be forgotten: purge all traces of participant data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Fetch audio paths to delete on disk
            cursor.execute("SELECT audio_path FROM answers WHERE participant_id = ?", (participant_id,))
            audio_paths = [r["audio_path"] for r in cursor.fetchall() if r["audio_path"]]

            # Fetch answer IDs
            cursor.execute("SELECT answer_id FROM answers WHERE participant_id = ?", (participant_id,))
            a_ids = [r["answer_id"] for r in cursor.fetchall()]

            for aid in a_ids:
                cursor.execute("DELETE FROM human_ratings WHERE answer_id = ?", (aid,))
                cursor.execute("DELETE FROM system_scores WHERE answer_id = ?", (aid,))
                cursor.execute("DELETE FROM feedback_evaluations WHERE answer_id = ?", (aid,))

            cursor.execute("DELETE FROM answers WHERE participant_id = ?", (participant_id,))
            cursor.execute("DELETE FROM sus_responses WHERE participant_id = ?", (participant_id,))
            deleted = cursor.execute("DELETE FROM participants WHERE participant_id = ?", (participant_id,)).rowcount

            # Log the deletion action without retaining PII
            log_entry = {
                "action": "delete_participant_data",
                "participant_id": participant_id,
                "answers_purged": len(a_ids),
                "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            }
            cursor.execute(
                "INSERT INTO audit_logs (log_id, action, details_json, timestamp) VALUES (?, ?, ?, ?)",
                (f"LOG_{uuid.uuid4().hex[:8]}", "deletion", json.dumps(log_entry), datetime.now(timezone.utc).replace(tzinfo=None).isoformat()),
            )
            conn.commit()

        # Delete physical audio files if present
        for ap in audio_paths:
            try:
                p = Path(ap)
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        return deleted

    # Answer methods
    def save_answer(self, participant_id: str, question_id: str, transcript: str, audio_path: Optional[str] = None, input_mode: str = "typed", asr_transcript: Optional[str] = None) -> str:
        aid = f"ANS_{uuid.uuid4().hex[:8]}"
        ans = Answer(
            answer_id=aid,
            participant_id=participant_id,
            question_id=question_id,
            transcript=transcript,
            audio_path=audio_path,
            input_mode=input_mode,
            asr_transcript=asr_transcript,
        )
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO answers (answer_id, participant_id, question_id, transcript, audio_path, input_mode, timestamp, asr_transcript)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ans.answer_id, ans.participant_id, ans.question_id, ans.transcript, ans.audio_path, ans.input_mode, ans.timestamp, ans.asr_transcript),
            )
            conn.commit()
        return aid

    def get_answer(self, answer_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.*, p.first_language, p.region, p.gender
                FROM answers a
                JOIN participants p ON a.participant_id = p.participant_id
                WHERE a.answer_id = ?
                """,
                (answer_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_answers(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.*, p.first_language, p.region, p.gender
                FROM answers a
                JOIN participants p ON a.participant_id = p.participant_id
                ORDER BY a.timestamp DESC
                """
            )
            return [dict(r) for r in cursor.fetchall()]

    # Rater methods
    def save_human_rating(self, rating: HumanRating) -> str:
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO human_ratings (
                    rating_id, rater_id, answer_id, relevance_correctness,
                    depth_completeness, structure_clarity, language_use, delivery,
                    overall_score, evidence_notes, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rater_id, answer_id) DO UPDATE SET
                    relevance_correctness=excluded.relevance_correctness,
                    depth_completeness=excluded.depth_completeness,
                    structure_clarity=excluded.structure_clarity,
                    language_use=excluded.language_use,
                    delivery=excluded.delivery,
                    overall_score=excluded.overall_score,
                    evidence_notes=excluded.evidence_notes,
                    timestamp=excluded.timestamp
                """,
                (
                    rating.rating_id, rating.rater_id, rating.answer_id,
                    rating.relevance_correctness, rating.depth_completeness,
                    rating.structure_clarity, rating.language_use, rating.delivery,
                    rating.overall_score, rating.evidence_notes, rating.timestamp
                ),
            )
            conn.commit()
        return rating.rating_id

    def get_human_ratings_for_answer(self, answer_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM human_ratings WHERE answer_id = ?", (answer_id,))
            return [dict(r) for r in cursor.fetchall()]

    def get_all_human_ratings(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM human_ratings ORDER BY timestamp DESC")
            return [dict(r) for r in cursor.fetchall()]

    # System Score methods
    def save_system_score(self, record: SystemScoreRecord) -> str:
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO system_scores (
                    score_id, answer_id, method, model_name, model_version,
                    temperature, prompt_version, prompt_hash, scores_json,
                    raw_output, latency_ms, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.score_id, record.answer_id, record.method, record.model_name,
                    record.model_version, record.temperature, record.prompt_version,
                    record.prompt_hash, record.scores_json, record.raw_output,
                    record.latency_ms, record.timestamp
                ),
            )
            conn.commit()
        return record.score_id

    def get_system_scores_for_answer(self, answer_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM system_scores WHERE answer_id = ?", (answer_id,))
            return [dict(r) for r in cursor.fetchall()]

    # Feedback evaluation + SUS
    def save_feedback_evaluation(self, rater_id: str, answer_id: str, claims_supported: int,
                                 claims_total: int, actionability: int, notes: str = "") -> str:
        eid = f"FE_{answer_id}_{rater_id}"
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback_evaluations
                    (eval_id, rater_id, answer_id, claims_supported, claims_total, actionability, notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rater_id, answer_id) DO UPDATE SET
                    claims_supported=excluded.claims_supported, claims_total=excluded.claims_total,
                    actionability=excluded.actionability, notes=excluded.notes, timestamp=excluded.timestamp
                """,
                (eid, rater_id, answer_id, claims_supported, claims_total, actionability, notes,
                 datetime.now(timezone.utc).replace(tzinfo=None).isoformat()),
            )
            conn.commit()
        return eid

    def get_all_feedback_evaluations(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM feedback_evaluations").fetchall()]

    def save_sus_response(self, participant_id: str, items: List[int], sus_score: float) -> str:
        sid = f"SUS_{uuid.uuid4().hex[:8]}"
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO sus_responses (sus_id, participant_id, items_json, sus_score, timestamp) VALUES (?, ?, ?, ?, ?)",
                (sid, participant_id, json.dumps(items), float(sus_score), datetime.now(timezone.utc).replace(tzinfo=None).isoformat()),
            )
            conn.commit()
        return sid

    def get_all_sus_responses(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM sus_responses").fetchall()]

    def get_latest_system_score(self, answer_id: str, method: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM system_scores WHERE answer_id = ? AND method = ? ORDER BY timestamp DESC LIMIT 1",
                (answer_id, method),
            ).fetchone()
            return dict(row) if row else None

    # Export merged dataset
    def export_merged_dataset(self, output_path: str | Path) -> pd.DataFrame:
        """
        Merge answers, participant attributes, human ratings (mean across raters),
        and latest system scores per method into a unified evaluation dataset.
        """
        answers = self.get_all_answers()
        if not answers:
            df_empty = pd.DataFrame()
            df_empty.to_csv(output_path, index=False)
            return df_empty

        df_ans = pd.DataFrame(answers)
        ratings = self.get_all_human_ratings()

        if ratings:
            df_rat = pd.DataFrame(ratings)
            # Group by answer_id to compute rater statistics
            rat_agg = df_rat.groupby("answer_id").agg(
                human_count=("rater_id", "nunique"),
                human_relevance_mean=("relevance_correctness", "mean"),
                human_depth_mean=("depth_completeness", "mean"),
                human_structure_mean=("structure_clarity", "mean"),
                human_language_mean=("language_use", "mean"),
                human_delivery_mean=("delivery", "mean"),
                human_overall_mean=("overall_score", "mean"),
                human_overall_std=("overall_score", "std"),
            ).reset_index()
            merged = pd.merge(df_ans, rat_agg, on="answer_id", how="left")
        else:
            merged = df_ans.copy()
            merged["human_count"] = 0
            merged["human_overall_mean"] = None

        # Latest valid system score per (answer, method) -> score_B1 ... score_S2 columns.
        # Scores from failed/invalid LLM runs are excluded so fallback values never enter the evaluation.
        latest: Dict[tuple, float] = {}
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT answer_id, method, scores_json FROM system_scores ORDER BY timestamp ASC"
            ).fetchall()
        for r in rows:
            try:
                payload = json.loads(r["scores_json"])
            except Exception:
                continue
            overall = payload.get("overall")
            if payload.get("validation_passed", True) is False or overall is None or overall != overall:
                continue
            latest[(r["answer_id"], r["method"])] = float(overall)
        for method in ["B1", "B2", "B3", "S1", "S2"]:
            merged[f"score_{method}"] = merged["answer_id"].map(lambda a, m=method: latest.get((a, m)))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        return merged
