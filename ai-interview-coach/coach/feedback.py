"""
Evidence-Linked Feedback Generator.
Produces grounded, actionable feedback derived strictly from rubric anchors and verified quotes.
Drops any unsupported or hallucinated claims.
"""

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional

from coach.scorer_llm import S1ScoringResult, DimensionScore
from coach.rubric import Rubric

logger = logging.getLogger(__name__)


@dataclass
class FeedbackClaim:
    dimension: str
    claim_type: str  # 'strength' or 'gap'
    claim_text: str
    evidence_quote: str
    is_grounded: bool
    rubric_level: int


@dataclass
class ActionableFeedback:
    answer_id: str
    overall_score: float
    strengths: List[FeedbackClaim]
    gaps: List[FeedbackClaim]
    prioritized_action: str
    model_answer_outline: str
    dropped_unsupported_claims: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "overall_score": self.overall_score,
            "strengths": [
                {
                    "dimension": c.dimension,
                    "text": c.claim_text,
                    "quote": c.evidence_quote,
                    "level": c.rubric_level,
                }
                for c in self.strengths
            ],
            "gaps": [
                {
                    "dimension": c.dimension,
                    "text": c.claim_text,
                    "quote": c.evidence_quote,
                    "level": c.rubric_level,
                }
                for c in self.gaps
            ],
            "prioritized_action": self.prioritized_action,
            "model_answer_outline": self.model_answer_outline,
            "dropped_unsupported_claims": self.dropped_unsupported_claims,
        }


class FeedbackGenerator:
    def __init__(self, rubric: Rubric):
        self.rubric = rubric

    def generate_feedback(
        self,
        scoring_result: S1ScoringResult,
        question_text: str,
        transcript: str,
        reference_answer: Optional[str] = None,
    ) -> ActionableFeedback:
        strengths: List[FeedbackClaim] = []
        gaps: List[FeedbackClaim] = []
        dropped_count = 0

        # Sort dimensions by score (lowest first to identify critical gaps)
        sorted_dims = sorted(
            scoring_result.dimension_scores.items(),
            key=lambda x: x[1].score
        )

        for dim_id, dim_score in sorted_dims:
            dim_def = self.rubric.dimensions.get(dim_id)
            dim_name = dim_def.name if dim_def else dim_id

            # Strictly drop claims if quote was ungrounded/hallucinated
            if not dim_score.quote_grounded and dim_score.evidence_quote != "N/A":
                logger.warning(f"Dropping ungrounded claim for dimension {dim_id}: '{dim_score.evidence_quote}'")
                dropped_count += 1
                continue

            quote_display = f'"{dim_score.evidence_quote}"' if dim_score.evidence_quote != "N/A" else "No direct quote cited"

            if dim_score.score >= 4:
                strengths.append(
                    FeedbackClaim(
                        dimension=dim_name,
                        claim_type="strength",
                        claim_text=f"Demonstrated competence in {dim_name}: {dim_score.reason}",
                        evidence_quote=quote_display,
                        is_grounded=dim_score.quote_grounded,
                        rubric_level=dim_score.score,
                    )
                )
            else:
                # Score 1, 2, or 3 represents an improvement opportunity
                anchor_target = dim_def.anchors.get(5, "Level 5 execution") if dim_def else "Level 5 standard"
                gaps.append(
                    FeedbackClaim(
                        dimension=dim_name,
                        claim_type="gap",
                        claim_text=f"Opportunity in {dim_name} (Current Level {dim_score.score}): {dim_score.reason}. Target benchmark: {anchor_target}",
                        evidence_quote=quote_display,
                        is_grounded=dim_score.quote_grounded,
                        rubric_level=dim_score.score,
                    )
                )

        # Formulate ONE single prioritized high-leverage action
        if gaps:
            lowest_gap = gaps[0]
            focus_ref = (
                lowest_gap.evidence_quote
                if lowest_gap.evidence_quote != "No direct quote cited"
                else "this part of your answer"
            )
            prioritized_action = (
                f"PRIORITY FOCUS: Elevate '{lowest_gap.dimension}'. "
                f"When discussing {focus_ref}, replace generalities with concrete technical/operational specifics: "
                f"state exact metrics, root-cause mechanisms, and explicit trade-offs."
            )
        else:
            prioritized_action = (
                "PRIORITY FOCUS: Maintain strong baseline performance. "
                "Challenge yourself to proactively anticipate edge cases and scale limits in your opening framing."
            )

        # Formulate model answer outline
        if reference_answer:
            model_outline = (
                f"### High-Scoring Benchmark Outline:\n"
                f"- **Context & Mechanism**: {reference_answer[:220]}...\n"
                f"- **Key Structuring Rule**: Ensure immediate direct answer in sentence 1, followed by STAR progression or explicit algorithmic comparison."
            )
        else:
            model_outline = (
                f"### Recommended Response Architecture for '{question_text}':\n"
                f"1. **Direct Thesis (Sentence 1)**: Directly state your stance or define the fundamental mechanism without introductory preamble.\n"
                f"2. **Evidence / Mechanism (Sentences 2-4)**: Explain the technical or behavioral specifics, citing exact numbers and concrete actions.\n"
                f"3. **Impact & Trade-offs (Sentence 5)**: Conclude with quantifiable business or architectural results and what was learned."
            )

        return ActionableFeedback(
            answer_id=scoring_result.answer_id,
            overall_score=scoring_result.overall_score,
            strengths=strengths,
            gaps=gaps,
            prioritized_action=prioritized_action,
            model_answer_outline=model_outline,
            dropped_unsupported_claims=dropped_count,
        )


@dataclass
class FeedbackFaithfulnessEvaluation:
    """Rater evaluation of feedback quality."""
    eval_id: str
    rater_id: str
    answer_id: str
    claims_supported_count: int
    claims_total_count: int
    actionability_score: int  # 1 to 5 scale
    rater_notes: str
    timestamp: str = ""

    @property
    def faithfulness_ratio(self) -> float:
        return self.claims_supported_count / max(1, self.claims_total_count)


@dataclass
class SUSQuestionnaire:
    """System Usability Scale (SUS) 10-item standard questionnaire (1=Strongly Disagree, 5=Strongly Agree)."""
    participant_id: str
    q1_frequent_use: int
    q2_complex: int
    q3_easy_to_use: int
    q4_technical_support: int
    q5_integrated_functions: int
    q6_inconsistency: int
    q7_learn_quickly: int
    q8_cumbersome: int
    q9_confident: int
    q10_learning_overhead: int

    def calculate_sus_score(self) -> float:
        """Standard SUS calculation: score out of 100."""
        # Odd items (1, 3, 5, 7, 9): score - 1
        odd_sum = (
            (self.q1_frequent_use - 1)
            + (self.q3_easy_to_use - 1)
            + (self.q5_integrated_functions - 1)
            + (self.q7_learn_quickly - 1)
            + (self.q9_confident - 1)
        )
        # Even items (2, 4, 6, 8, 10): 5 - score
        even_sum = (
            (5 - self.q2_complex)
            + (5 - self.q4_technical_support)
            + (5 - self.q6_inconsistency)
            + (5 - self.q8_cumbersome)
            + (5 - self.q10_learning_overhead)
        )
        return (odd_sum + even_sum) * 2.5
