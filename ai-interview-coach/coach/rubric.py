"""
Rubric loader and schema definitions.
Provides type-safe access to dimensions, level anchors, and rater training examples.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class TrainingExample:
    question: str
    answer_excerpt: str
    score: int
    evidence_quote: str
    rationale: str


@dataclass
class RubricDimension:
    id: str
    name: str
    description: str
    anchors: Dict[int, str]
    training_examples: List[TrainingExample] = field(default_factory=list)
    optional: bool = False

    def format_for_prompt(self) -> str:
        """Render dimension and anchors into a concise LLM scoring guide block."""
        anchor_text = "\n".join([f"    Level {k}: {v}" for k, v in sorted(self.anchors.items())])
        return (
            f"### Dimension: {self.name} (ID: {self.id})\n"
            f"Description: {self.description}\n"
            f"Anchors (Scale 1 to 5):\n{anchor_text}\n"
        )


@dataclass
class Rubric:
    version: str
    description: str
    dimensions: Dict[str, RubricDimension]

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "Rubric":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Rubric configuration not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        dims = {}
        for dim_id, d_data in data.get("dimensions", {}).items():
            anchors = {int(k): str(v) for k, v in d_data.get("anchors", {}).items()}
            examples = [
                TrainingExample(
                    question=ex.get("question", ""),
                    answer_excerpt=ex.get("answer_excerpt", ""),
                    score=int(ex.get("score", 1)),
                    evidence_quote=ex.get("evidence_quote", ""),
                    rationale=ex.get("rationale", ""),
                )
                for ex in d_data.get("training_examples", [])
            ]
            dims[dim_id] = RubricDimension(
                id=dim_id,
                name=d_data.get("name", dim_id),
                description=d_data.get("description", ""),
                anchors=anchors,
                training_examples=examples,
                optional=d_data.get("optional", False),
            )

        return cls(
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            dimensions=dims,
        )

    def get_eval_dimensions(self, has_audio: bool = False) -> List[RubricDimension]:
        """Return dimensions to evaluate based on audio availability."""
        return [
            dim for dim in self.dimensions.values()
            if not dim.optional or (dim.optional and has_audio)
        ]

    def format_all_for_prompt(self, has_audio: bool = False) -> str:
        dims = self.get_eval_dimensions(has_audio=has_audio)
        return "\n".join([d.format_for_prompt() for d in dims])
