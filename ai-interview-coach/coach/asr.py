"""
Speech Recognition module.
Wraps faster-whisper with fallback handling and transcript extraction.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class TranscriptionResult:
    transcript: str
    duration_seconds: float
    detected_language: str
    success: bool
    error_message: Optional[str] = None


class ASRProcessor:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8", vad_filter: bool = False):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except ImportError:
                raise ImportError(
                    "faster-whisper is not installed. Spoken audio processing requires 'faster-whisper'. "
                    "You can still use typed answers."
                )
        return self._model

    def transcribe(self, audio_file_path: str | Path) -> TranscriptionResult:
        path_str = str(audio_file_path)
        if not os.path.exists(path_str):
            return TranscriptionResult(
                transcript="",
                duration_seconds=0.0,
                detected_language="unknown",
                success=False,
                error_message=f"Audio file not found: {path_str}",
            )

        try:
            model = self._get_model()
            # Disable aggressive vad_filter by default to preserve real candidate speech
            segments, info = model.transcribe(
                path_str,
                vad_filter=self.vad_filter,
                beam_size=5,
            )
            segments_list = list(segments)
            transcript = " ".join([s.text.strip() for s in segments_list]).strip()
            duration = getattr(info, "duration", 0.0) or 0.0
            if not duration and segments_list:
                duration = max((s.end for s in segments_list), default=0.0)

            return TranscriptionResult(
                transcript=transcript,
                duration_seconds=round(duration, 2),
                detected_language=getattr(info, "language", "en") or "en",
                success=True,
            )
        except Exception as e:
            return TranscriptionResult(
                transcript="",
                duration_seconds=0.0,
                detected_language="unknown",
                success=False,
                error_message=str(e),
            )
