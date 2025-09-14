"""Typhoon ASR wrapper module

Provides TyphoonASR class with methods:
 - prepare_audio(input_path, output_path=None, target_sr=16000)
 - load_model(device='cpu')  # lazy
 - transcribe_file(input_path)

Designed for use inside APIs (Flask/FastAPI). Heavy dependencies are imported
inside methods to avoid slowing imports.
"""

from __future__ import annotations

import os
import time
from typing import Optional


class TyphoonASR:
    """Object-oriented wrapper around the Typhoon ASR realtime model.

    Usage:
        asr = TyphoonASR()
        text = asr.transcribe_file('path/to/file.wav')
    """

    def __init__(self, model_name: str = "scb10x/typhoon-asr-realtime") -> None:
        self.model_name = model_name
        self._model = None
        self._device = None

    def load_model(self, device: str = "cpu") -> None:
        """Lazy-load the NeMo ASR model. Call before transcribing if you want to
        control device placement.
        """
        if self._model is not None and self._device == device:
            return

        # import heavy deps only when needed
        try:
            import nemo.collections.asr as nemo_asr
        except Exception as e:  # pragma: no cover - environment dependent
            raise RuntimeError("nemo.collections.asr is required. Install NeMo and dependencies") from e

        self._device = device
        # map_location uses torch's device mapping internally in NeMo
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name,
            map_location=device,
        )

    def prepare_audio(self, input_path: str, output_path: Optional[str] = None, target_sr: int = 16000) -> Optional[str]:
        """Load, resample, normalize and write a WAV file suitable for model input.

        Returns output_path on success or None on failure.
        """
        try:
            import librosa
            import soundfile as sf
        except Exception as e:  # pragma: no cover - environment dependent
            raise RuntimeError("librosa and soundfile are required for audio preprocessing") from e

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(os.path.dirname(input_path), f"{base}.processed.wav")

        y, sr = librosa.load(input_path, sr=None)
        duration = len(y) / sr if sr else 0.0

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # avoid division by zero
        peak = max(abs(y)) if len(y) else 1.0
        if peak > 0:
            y = y / peak

        sf.write(output_path, y, target_sr)
        return output_path

    def transcribe_file(self, input_path: str, device: str = "cpu") -> str:
        """High-level helper: prepare audio, lazy-load model and transcribe.

        Returns the transcription text.
        """
        # prepare audio
        processed = self.prepare_audio(input_path)

        # load model (lazy)
        self.load_model(device=device)

        if self._model is None:
            raise RuntimeError("ASR model not loaded")

        # import soundfile for metadata
        import soundfile as sf

        start_time = time.time()
        # NeMo model's transcribe accepts list of file paths
        transcriptions = self._model.transcribe(audio=[processed])
        processing_time = time.time() - start_time

        # gather duration
        audio_info = sf.info(processed)
        audio_duration = audio_info.duration if audio_info else 0.0
        rtf = processing_time / audio_duration if audio_duration else float('inf')

        # First transcription
        if not transcriptions:
            raise RuntimeError("Transcription returned no results")

        transcript = transcriptions[0]
        # transcript can be a simple string or object with .text
        text = transcript if isinstance(transcript, str) else getattr(transcript, 'text', str(transcript))

        # Attach performance metadata as attributes (optional)
        result = {
            'text': text,
            'processing_time': processing_time,
            'audio_duration': audio_duration,
            'rtf': rtf,
        }

        return result


if __name__ == '__main__':
    # small CLI for quick local testing
    import argparse

    parser = argparse.ArgumentParser(description='Typhoon ASR CLI')
    parser.add_argument('input', help='Path to input audio file (wav)')
    parser.add_argument('--device', default='cpu', help='Device to run model on (cpu or cuda)')
    args = parser.parse_args()

    asr = TyphoonASR()
    out = asr.transcribe_file(args.input, device=args.device)
    print('Transcription:')
    print(out['text'])
    print('\nPerformance:')
    print(f"processing_time={out['processing_time']:.2f}s audio_duration={out['audio_duration']:.2f}s rtf={out['rtf']:.2f}")