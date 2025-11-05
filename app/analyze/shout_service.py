import io
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000
WIN_MS = 200
HOP_MS = 100
THRESH_DBFS = -10.0
MIN_RUN_MS = 600
MAX_CREST_DB = 18.0


@dataclass
class ShoutResult:
    present: bool
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    peak_dbfs: Optional[float] = None
    confidence: Optional[float] = None


logger = logging.getLogger(__name__)


def load_audio_from_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return samples, sr


def to_mono_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)

    if sr != TARGET_SR:
        gcd = np.gcd(sr, TARGET_SR)
        up = TARGET_SR // gcd
        down = sr // gcd
        samples = resample_poly(samples, up, down)

    if samples.dtype not in (np.float32, np.float64):
        samples = samples.astype(np.float32)

    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32768.0).astype(np.float32)


def frame_signal(signal: np.ndarray, sr: int, win_ms: int, hop_ms: int):
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    frames: List[np.ndarray] = []
    idxs: List[int] = []
    offset = 0
    while offset + win <= len(signal):
        frames.append(signal[offset : offset + win])
        idxs.append(offset)
        offset += hop
    return idxs, frames, win


def rms(seg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-9)


def crest_factor_db(seg: np.ndarray, rms_val: float) -> float:
    peak = float(np.max(np.abs(seg))) + 1e-9
    crest_linear = peak / rms_val
    return 20.0 * np.log10(crest_linear + 1e-12)


def dbfs_from_rms(rms_val: float) -> float:
    return 20.0 * np.log10(max(rms_val / 32768.0, 1e-9))


def detect_shout(signal: np.ndarray, sr: int = TARGET_SR) -> ShoutResult:
    logger.info("Analyzing audio signal: samples=%d, sample_rate=%d", len(signal), sr)
    idxs, frames, win = frame_signal(signal, sr, WIN_MS, HOP_MS)
    if not frames:
        logger.info("No frames available for shout detection")
        return ShoutResult(present=False)

    run_start = None
    run_end = None
    peak_db = -120.0

    for index, segment in zip(idxs, frames):
        r = rms(segment)
        db = dbfs_from_rms(r)
        crest_db = crest_factor_db(segment, r)

        loud = db >= THRESH_DBFS
        not_impulsive = crest_db < MAX_CREST_DB
        if loud and not_impulsive:
            if run_start is None:
                run_start = index
            run_end = index + win
            peak_db = max(peak_db, db)
        else:
            if run_start is not None:
                duration_ms = int((run_end - run_start) * 1000 / sr)
                if duration_ms >= MIN_RUN_MS:
                    logger.info(
                        "Shout detected mid-stream (start=%dms, end=%dms, peak=%.2f dBFS)",
                        int(run_start * 1000 / sr),
                        int(run_end * 1000 / sr),
                        peak_db,
                    )
                    return ShoutResult(
                        present=True,
                        start_ms=int(run_start * 1000 / sr),
                        end_ms=int(run_end * 1000 / sr),
                        peak_dbfs=float(round(peak_db, 2)),
                        confidence=0.6,
                    )
                run_start = None
                run_end = None
                peak_db = -120.0

    if run_start is not None and run_end is not None:
        duration_ms = int((run_end - run_start) * 1000 / sr)
        if duration_ms >= MIN_RUN_MS:
            logger.info(
                "Shout detected at stream end (start=%dms, end=%dms, peak=%.2f dBFS)",
                int(run_start * 1000 / sr),
                int(run_end * 1000 / sr),
                peak_db,
            )
            return ShoutResult(
                present=True,
                start_ms=int(run_start * 1000 / sr),
                end_ms=int(run_end * 1000 / sr),
                peak_dbfs=float(round(peak_db, 2)),
                confidence=0.6,
            )

    logger.info("Shout not detected")
    return ShoutResult(present=False)
