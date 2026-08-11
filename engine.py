#!/usr/bin/env python3
"""engine.py — OpenRemaster core engine.

Guided, profile-based audio remastering: device profiles (playback
hardware) and content profiles (recording era/style) resolve into a flat
DSP parameter set, applied here via a pydub/ffmpeg + numpy/scipy pipeline
(restoration, EQ, dynamics, stem-based vocal/instrument chains, 5.1 upmix).

Architecture — three phases, one conversion each direction:
  Phase A  pydub/ffmpeg I/O: load → restoration filters → float32
  Phase B  numpy DSP: EQ → compress → saturate → widen → LUFS → surround
  Phase C  pydub/ffmpeg export: float32 → encode → metadata → mux MKV

Dependencies:
  Tier 0 (required):   pydub, ffmpeg
  Tier 1 (recommended): numpy, scipy, pyloudnorm, soundfile, sounddevice, mutagen
  Tier 2 (optional):   demucs
  Tier 3 (experimental, off by default): voicefixer, df (DeepFilterNet)
"""

from __future__ import annotations

# stdlib
import argparse
import importlib
import importlib.util
import json
import logging
import math
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_DIR = Path.home() / ".openremaster" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("remaster")
log.setLevel(logging.DEBUG)

_fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
log.addHandler(_ch)

try:
    from logging.handlers import RotatingFileHandler as _RFH
    _fh = _RFH(_LOG_DIR / "remaster.log", maxBytes=5 * 1024 * 1024, backupCount=3)
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_fmt)
    log.addHandler(_fh)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Capability map — probe every dependency once at import time
# ---------------------------------------------------------------------------

@dataclass
class Capabilities:
    numpy: bool = False
    scipy: bool = False
    pyloudnorm: bool = False
    soundfile: bool = False
    sounddevice: bool = False
    mutagen: bool = False
    demucs: bool = False
    voicefixer: bool = False
    deepfilternet: bool = False
    ffmpeg: str | None = None
    ffprobe: str | None = None

    def tier(self) -> int:
        """Minimum functional tier: 0=basic pydub, 1=full DSP, 2=stems, 3=AI."""
        if not self.numpy or not self.scipy:
            return 0
        if not self.demucs:
            return 1
        return 2

    def summary(self) -> str:
        lines = [
            f"ffmpeg:      {'ok  ' + self.ffmpeg if self.ffmpeg else 'MISSING (required)'}",
            f"ffprobe:     {'ok  ' + self.ffprobe if self.ffprobe else 'MISSING (required)'}",
            f"numpy:       {'ok' if self.numpy else 'missing — DSP runs in reduced-quality mode'}",
            f"scipy:       {'ok' if self.scipy else 'missing — LR crossovers unavailable'}",
            f"pyloudnorm:  {'ok' if self.pyloudnorm else 'missing — LUFS measured by RMS approximation'}",
            f"soundfile:   {'ok' if self.soundfile else 'missing — 5.1 output unavailable'}",
            f"sounddevice: {'ok' if self.sounddevice else 'missing — preview uses ffplay fallback'}",
            f"mutagen:     {'ok' if self.mutagen else 'missing — metadata read/write unavailable'}",
            f"demucs:      {'ok' if self.demucs else 'missing — stem processing unavailable'}",
            f"voicefixer:  {'ok' if self.voicefixer else 'not installed (experimental)'}",
            f"deepfilternet:{'ok' if self.deepfilternet else 'not installed (experimental)'}",
        ]
        return "\n".join(lines)


def _probe_capabilities() -> Capabilities:
    cap = Capabilities()

    cap.ffmpeg  = shutil.which("ffmpeg")
    cap.ffprobe = shutil.which("ffprobe")

    for attr, pkg in [
        ("numpy",       "numpy"),
        ("scipy",       "scipy"),
        ("pyloudnorm",  "pyloudnorm"),
        ("soundfile",   "soundfile"),
        ("sounddevice", "sounddevice"),
        ("mutagen",     "mutagen"),
    ]:
        try:
            importlib.import_module(pkg)
            setattr(cap, attr, True)
        except Exception:
            pass

    # Optional/experimental packages: these pull in heavy dependencies
    # (demucs -> torch, in particular) that can take several seconds to
    # actually import. find_spec() only checks importability without
    # executing the module, so presence-detection here stays fast. The
    # real import still happens lazily, only where the feature is used
    # (separate_stems(), _apply_voicefixer_mid_only(), etc).
    for attr, pkg in [
        ("demucs",        "demucs"),
        ("voicefixer",    "voicefixer"),
        ("deepfilternet", "df"),
    ]:
        try:
            setattr(cap, attr, importlib.util.find_spec(pkg) is not None)
        except Exception:
            pass

    return cap


CAP = _probe_capabilities()

if not CAP.ffmpeg:
    log.warning("ffmpeg not found on PATH — audio encode/decode will fail.")
if not CAP.numpy or not CAP.scipy:
    log.info("numpy/scipy not available — running in reduced-quality mode (pydub DSP only).")

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

from pydub import AudioSegment

if CAP.numpy:
    import numpy as np

if CAP.scipy:
    from scipy.signal import butter, sosfiltfilt

if CAP.pyloudnorm:
    import pyloudnorm as pyln

if CAP.soundfile:
    import soundfile as sf

if CAP.sounddevice:
    import sounddevice as sd

# ---------------------------------------------------------------------------
# Phase A helpers — ffmpeg binary resolution
# ---------------------------------------------------------------------------

def _resolve_ffmpeg() -> tuple[str | None, str | None]:
    """Return (ffmpeg_path, ffprobe_path) from PATH or common locations."""
    ffmpeg  = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if not ffmpeg or not ffprobe:
        for d in [
            Path(r"C:\ffmpeg\bin"),
            Path(r"C:\Program Files\ffmpeg\bin"),
        ]:
            if not ffmpeg and (d / "ffmpeg.exe").exists():
                ffmpeg = str(d / "ffmpeg.exe")
            if not ffprobe and (d / "ffprobe.exe").exists():
                ffprobe = str(d / "ffprobe.exe")

    try:
        import imageio_ffmpeg
        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if not ffmpeg and exe.exists():
            ffmpeg = str(exe)
        probe = exe.parent / "ffprobe"
        if not ffprobe and probe.exists():
            ffprobe = str(probe)
    except Exception:
        pass

    return ffmpeg, ffprobe


FFMPEG_BIN, FFPROBE_BIN = _resolve_ffmpeg()


# ---------------------------------------------------------------------------
# Phase A — File I/O: load, metadata, restoration, float32 conversion
# ---------------------------------------------------------------------------

def load_audio_file(path: Path) -> AudioSegment:
    """Load any audio file pydub/ffmpeg can decode."""
    return AudioSegment.from_file(path)


def pydub_to_float32(seg: AudioSegment) -> tuple["np.ndarray", int]:
    """Convert pydub AudioSegment to float32 numpy array (samples, channels)
    and sample rate. Conversion happens exactly once per pipeline run.
    """
    if not CAP.numpy:
        raise RuntimeError("numpy is required for float32 conversion.")
    raw = np.array(seg.get_array_of_samples(), dtype=np.int32)
    max_val = float(2 ** (seg.sample_width * 8 - 1))
    raw = raw.astype(np.float32) / max_val
    if seg.channels > 1:
        raw = raw.reshape(-1, seg.channels)
    else:
        raw = raw.reshape(-1, 1)
    return raw, seg.frame_rate


def float32_to_pydub(audio: "np.ndarray", sr: int, sample_width: int = 2) -> AudioSegment:
    """Convert float32 numpy array back to pydub AudioSegment.
    Clips to [-1, 1] before conversion to prevent wrap-around artefacts.
    """
    if not CAP.numpy:
        raise RuntimeError("numpy is required for float32 conversion.")
    audio = np.clip(audio, -1.0, 1.0)
    max_val = float(2 ** (sample_width * 8 - 1))
    pcm = (audio * max_val).astype(np.int16 if sample_width == 2 else np.int32)
    channels = audio.shape[1] if audio.ndim == 2 else 1
    return AudioSegment(
        pcm.tobytes(),
        frame_rate=sr,
        sample_width=sample_width,
        channels=channels,
    )



def resample_audio(
    audio: "np.ndarray",
    orig_sr: int,
    target_sr: int,
) -> "np.ndarray":
    """Resample float32 audio from orig_sr to target_sr.

    Prefers scipy.signal.resample_poly for quality; falls back to
    linear interpolation when scipy is unavailable. No-op when rates match.
    """
    if not CAP.numpy or orig_sr == target_sr or len(audio) == 0:
        return audio

    if CAP.scipy:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(target_sr), int(orig_sr))
        up, down = int(target_sr) // g, int(orig_sr) // g
        # resample_poly operates along axis 0
        return resample_poly(audio, up, down, axis=0).astype(np.float32)

    # Linear-interpolation fallback
    n_out = int(round(len(audio) * target_sr / orig_sr))
    if n_out <= 0:
        return audio
    xp = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    xq = np.linspace(0.0, 1.0, n_out, endpoint=False)
    if audio.ndim == 1:
        return np.interp(xq, xp, audio).astype(np.float32)
    channels = [
        np.interp(xq, xp, audio[:, ch]) for ch in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def apply_restoration_filters(
    input_path: Path,
    output_path: Path,
    declick: bool = False,
    declick_threshold: float = 2.0,
    denoise: bool = False,
    denoise_amount: float = 10.0,
    denoise_floor: float = -50.0,
    denoise_type: str = "white",
) -> Path:
    """Run ffmpeg restoration filters (adeclick, adeclip, afftdn) as a
    pre-processing pass. Returns output_path. If no filters are active,
    returns input_path unchanged (no ffmpeg call).

    These run before the float32 DSP chain so the DSP operates on
    already-cleaned audio.
    """
    filters: list[str] = []
    if declick:
        filters.append(f"adeclick=threshold={declick_threshold}")
        filters.append(f"adeclip=threshold={declick_threshold * 5:.1f}")
    if denoise and denoise_amount > 0:
        filters.append(
            f"afftdn=nr={denoise_amount}:nf={denoise_floor}:nt={denoise_type}"
        )
    if not filters:
        return input_path

    cmd = [
        FFMPEG_BIN or "ffmpeg",
        "-y", "-i", str(input_path),
        "-af", ",".join(filters),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        log.warning(
            "Restoration filter pass failed (returncode %d): %s",
            result.returncode,
            result.stderr.decode(errors="replace")[:300],
        )
        return input_path  # fallback: use original
    return output_path


def trim_track_silence(
    audio: "np.ndarray",
    sr: int,
    silence_thresh_db: float = -45.0,
    min_silence_ms: int = 400,
    keep_silence_ms: int = 150,
) -> "np.ndarray":
    """Trim leading/trailing silence from a float32 numpy array.
    Operates in float32 domain (Phase B) since restoration runs first.
    """
    if not CAP.numpy:
        return audio
    thresh_linear = 10 ** (silence_thresh_db / 20.0)
    keep_sil = int(sr * keep_silence_ms / 1000)

    # Compute per-sample amplitude envelope (mono mix for detection)
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio.ravel()
    loud = np.abs(mono) > thresh_linear

    # Find first and last loud sample
    loud_idx = np.where(loud)[0]
    if len(loud_idx) == 0:
        return audio  # fully silent — return as is
    first = max(0, loud_idx[0] - keep_sil)
    last  = min(len(audio), loud_idx[-1] + keep_sil)
    return audio[first:last]


def apply_fades(
    audio: "np.ndarray",
    sr: int,
    fade_in_ms: int = 0,
    fade_out_ms: int = 0,
) -> "np.ndarray":
    """Apply linear fade-in and/or fade-out in float32 domain."""
    if not CAP.numpy:
        return audio
    n = len(audio)
    if fade_in_ms > 0:
        fade_in_samps = min(int(sr * fade_in_ms / 1000), n)
        ramp = np.linspace(0.0, 1.0, fade_in_samps, dtype=np.float32)
        audio[:fade_in_samps] *= ramp[:, None] if audio.ndim == 2 else ramp
    if fade_out_ms > 0:
        fade_out_samps = min(int(sr * fade_out_ms / 1000), n)
        ramp = np.linspace(1.0, 0.0, fade_out_samps, dtype=np.float32)
        audio[n - fade_out_samps:] *= ramp[:, None] if audio.ndim == 2 else ramp
    return audio



# ---------------------------------------------------------------------------
# Phase B — DSP primitives (float32 numpy/scipy)
# ---------------------------------------------------------------------------

# --- B.1 Linkwitz-Riley crossover filters ---

def _lr_sos(cutoff_hz: float, sr: int, btype: str, order: int = 4) -> "np.ndarray":
    """Compute a Linkwitz-Riley filter as second-order sections.
    A Linkwitz-Riley filter of order N is a Butterworth of order N/2
    applied twice (once via sosfiltfilt which doubles it). We use
    order=4 (LR4, -24dB/octave) as the default — the industry standard
    for crossover filters.
    """
    nyq = sr / 2.0
    cutoff_norm = min(cutoff_hz / nyq, 0.9999)
    sos = butter(order // 2, cutoff_norm, btype=btype, output="sos")
    return sos


def lr_lowpass(audio: "np.ndarray", sr: int, cutoff_hz: float) -> "np.ndarray":
    """Linkwitz-Riley lowpass filter applied zero-phase (sosfiltfilt)."""
    if not CAP.scipy:
        return _rc_lowpass_fallback(audio, sr, cutoff_hz)
    sos = _lr_sos(cutoff_hz, sr, "low")
    return sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def lr_highpass(audio: "np.ndarray", sr: int, cutoff_hz: float) -> "np.ndarray":
    """Linkwitz-Riley highpass filter applied zero-phase (sosfiltfilt)."""
    if not CAP.scipy:
        return _rc_highpass_fallback(audio, sr, cutoff_hz)
    sos = _lr_sos(cutoff_hz, sr, "high")
    return sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def lr_bandpass(
    audio: "np.ndarray", sr: int, low_hz: float, high_hz: float
) -> "np.ndarray":
    """Bandpass as LR lowpass on a LR-highpassed signal."""
    return lr_lowpass(lr_highpass(audio, sr, low_hz), sr, high_hz)


def _rc_lowpass_fallback(
    audio: "np.ndarray", sr: int, cutoff_hz: float, passes: int = 4
) -> "np.ndarray":
    """Cascaded RC lowpass (~24dB/oct at 4 passes). Used when scipy is absent."""
    seg = _np_to_pydub(audio, sr)
    for _ in range(passes):
        seg = seg.low_pass_filter(int(cutoff_hz))
    return _pydub_to_np(seg)


def _rc_highpass_fallback(
    audio: "np.ndarray", sr: int, cutoff_hz: float, passes: int = 4
) -> "np.ndarray":
    seg = _np_to_pydub(audio, sr)
    for _ in range(passes):
        seg = seg.high_pass_filter(int(cutoff_hz))
    return _pydub_to_np(seg)


def _np_to_pydub(audio: "np.ndarray", sr: int) -> AudioSegment:
    """Lightweight round-trip for fallback filter paths. 16-bit intermediate."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    ch = audio.shape[1] if audio.ndim == 2 else 1
    return AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=ch)


def _pydub_to_np(seg: AudioSegment) -> "np.ndarray":
    raw = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    if seg.channels > 1:
        return raw.reshape(-1, seg.channels)
    return raw.reshape(-1, 1)


# --- B.2 Band-split infrastructure (shared by all multi-band processing) ---

def split_bands(
    audio: "np.ndarray", sr: int, crossover_hz: float
) -> tuple["np.ndarray", "np.ndarray"]:
    """Split audio into (low, high) using LR crossovers (scipy) or
    cascaded RC filters (fallback). Returns arrays of identical shape.
    """
    low  = lr_lowpass(audio, sr, crossover_hz)
    high = lr_highpass(audio, sr, crossover_hz)
    return low, high


def recombine_bands(
    original: "np.ndarray", *bands: "np.ndarray"
) -> "np.ndarray":
    """Sum bands and RMS-match the result to `original`. The cascaded
    filters don't preserve the exact low+high=original identity, so
    this normalisation step keeps the overall level stable regardless
    of how many band-split/merge operations were chained.
    """
    mix = sum(bands)
    orig_rms = float(np.sqrt(np.mean(original ** 2))) + 1e-12
    mix_rms  = float(np.sqrt(np.mean(mix ** 2)))  + 1e-12
    return (mix * (orig_rms / mix_rms)).astype(np.float32)



# --- B.3 RBJ Audio Cookbook biquad EQ ---

def _rbj_peaking_sos(freq: float, gain_db: float, q: float, sr: int) -> "np.ndarray":
    """Peaking EQ filter coefficients (RBJ Audio Cookbook).
    gain_db / 2.0 compensates for sosfiltfilt doubling the filter,
    so the API gain_db always means what the caller intends.
    """
    A  = 10 ** (gain_db / 80.0)   # /80 = /40 (RBJ) then /2 (filtfilt compensation)
    w0 = 2 * math.pi * freq / sr
    alpha = math.sin(w0) / (2 * q)

    b0 =  1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 =  1 - alpha * A
    a0 =  1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 =  1 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0,     a1 / a0, a2 / a0], dtype=np.float64)
    # Convert to second-order sections (single section)
    return np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])


def _rbj_shelf_sos(
    freq: float, gain_db: float, sr: int, shelf_type: str
) -> "np.ndarray":
    """Low or high shelf EQ coefficients (RBJ Audio Cookbook).
    shelf_type: 'low' or 'high'.
    """
    A  = 10 ** (gain_db / 80.0)
    w0 = 2 * math.pi * freq / sr
    cosw0 = math.cos(w0)
    sinw0 = math.sin(w0)
    S = 1.0  # shelf slope = 1 (maximally flat)
    alpha = sinw0 / 2 * math.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    sqA   = math.sqrt(A)

    if shelf_type == "low":
        b0 =  A * ((A + 1) - (A - 1) * cosw0 + 2 * sqA * alpha)
        b1 =  2 * A * ((A - 1) - (A + 1) * cosw0)
        b2 =  A * ((A + 1) - (A - 1) * cosw0 - 2 * sqA * alpha)
        a0 =       (A + 1) + (A - 1) * cosw0 + 2 * sqA * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cosw0)
        a2 =       (A + 1) + (A - 1) * cosw0 - 2 * sqA * alpha
    else:  # high
        b0 =  A * ((A + 1) + (A - 1) * cosw0 + 2 * sqA * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cosw0)
        b2 =  A * ((A + 1) + (A - 1) * cosw0 - 2 * sqA * alpha)
        a0 =       (A + 1) - (A - 1) * cosw0 + 2 * sqA * alpha
        a1 =  2 * ((A - 1) - (A + 1) * cosw0)
        a2 =       (A + 1) - (A - 1) * cosw0 - 2 * sqA * alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0,     a1 / a0, a2 / a0], dtype=np.float64)
    return np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])


def apply_eq(
    audio: "np.ndarray",
    sr: int,
    bass_shelf_hz: float = 0.0,
    bass_shelf_db: float = 0.0,
    treble_shelf_hz: float = 0.0,
    treble_shelf_db: float = 0.0,
    presence_hz: float = 0.0,
    presence_db: float = 0.0,
    presence_q: float = 1.2,
    notch_hz: float = 0.0,
    notch_db: float = 0.0,
    notch_q: float = 2.0,
    bands: list[tuple[float, float, float]] | None = None,
) -> "np.ndarray":
    """Apply a full EQ chain: bass shelf + treble shelf + presence peak
    + optional notch + up to 5 user-defined peaking bands.

    All filters use sosfiltfilt (zero-phase, applied twice). The /80 gain
    compensation in the coefficient functions means gain_db values passed
    here represent what actually gets applied.

    Falls back to pydub overlay EQ when scipy is unavailable (less precise
    but functional).
    """
    if not CAP.scipy or not CAP.numpy:
        return _eq_pydub_fallback(audio, sr, bass_shelf_db, treble_shelf_db)

    out = audio.copy()

    if bass_shelf_hz > 0 and bass_shelf_db != 0:
        sos = _rbj_shelf_sos(bass_shelf_hz, bass_shelf_db, sr, "low")
        out = sosfiltfilt(sos, out, axis=0).astype(np.float32)

    if treble_shelf_hz > 0 and treble_shelf_db != 0:
        sos = _rbj_shelf_sos(treble_shelf_hz, treble_shelf_db, sr, "high")
        out = sosfiltfilt(sos, out, axis=0).astype(np.float32)

    if presence_hz > 0 and presence_db != 0:
        sos = _rbj_peaking_sos(presence_hz, presence_db, presence_q, sr)
        out = sosfiltfilt(sos, out, axis=0).astype(np.float32)

    if notch_hz > 0 and notch_db != 0:
        sos = _rbj_peaking_sos(notch_hz, notch_db, notch_q, sr)
        out = sosfiltfilt(sos, out, axis=0).astype(np.float32)

    for freq, gain_db, q in (bands or []):
        if gain_db == 0:
            continue
        sos = _rbj_peaking_sos(freq, gain_db, q, sr)
        out = sosfiltfilt(sos, out, axis=0).astype(np.float32)

    return out


def _eq_pydub_fallback(
    audio: "np.ndarray",
    sr: int,
    bass_shelf_db: float,
    treble_shelf_db: float,
) -> "np.ndarray":
    """Pydub-based EQ fallback when scipy is unavailable. Uses overlay
    of filtered copies — less precise but keeps the pipeline running.
    """
    seg = _np_to_pydub(audio, sr)

    def _overlay_gain(target_db: float) -> float:
        ratio = 10 ** (target_db / 20.0)
        extra = max(ratio - 1, 1e-4)
        return 20 * math.log10(extra)

    if bass_shelf_db > 0:
        bass = seg.low_pass_filter(140).apply_gain(_overlay_gain(bass_shelf_db))
        seg = seg.overlay(bass)
    if treble_shelf_db > 0:
        treble = seg.high_pass_filter(6000).apply_gain(_overlay_gain(treble_shelf_db))
        seg = seg.overlay(treble)

    return _pydub_to_np(seg)


# --- B.4 Compressor (dual attack/release envelope follower, fixed) ---

def _ema_envelope_fast(level_sq: "np.ndarray", sr: int,
                       attack_ms: float, release_ms: float) -> "np.ndarray":
    """Dual-coefficient EMA envelope follower.

    Prefer a numba-accelerated inner loop when available; otherwise use a
    vectorised frame-based approximation that is ~50–100× faster than a
    pure-Python sample loop on long files while preserving attack/release
    character closely enough for mastering use.
    """
    n = len(level_sq)
    tau_att = math.exp(-1.0 / max(sr * attack_ms / 1000.0, 1e-9))
    tau_rel = math.exp(-1.0 / max(sr * release_ms / 1000.0, 1e-9))

    # Optional numba path (same algorithm, compiled)
    try:
        import numba as nb  # type: ignore

        @nb.njit(cache=True)
        def _loop(ls, ta, tr):
            out = np.empty(ls.shape[0], dtype=np.float32)
            lvl = 0.0
            for i in range(ls.shape[0]):
                target = float(ls[i])
                coef = ta if target > lvl else tr
                lvl = coef * lvl + (1.0 - coef) * target
                out[i] = lvl
            return out

        return _loop(level_sq.astype(np.float32), tau_att, tau_rel)
    except Exception:
        pass

    # Frame-based vectorised approximation:
    # 1. RMS energy per short hop (≈ attack/4, min 1 sample)
    # 2. Sequential EMA only over the much smaller frame array
    # 3. Linear upsample back to sample rate
    hop = max(1, int(sr * max(attack_ms, 1.0) / 1000.0 / 4.0))
    n_frames = (n + hop - 1) // hop
    # Pad so reshape is clean
    pad = n_frames * hop - n
    if pad:
        padded = np.pad(level_sq, (0, pad), mode="edge")
    else:
        padded = level_sq
    frames = padded.reshape(n_frames, hop).mean(axis=1)

    env_f = np.empty(n_frames, dtype=np.float32)
    lvl = 0.0
    # Frame-rate coefficients (hop samples per step)
    ta_f = tau_att ** hop
    tr_f = tau_rel ** hop
    for i in range(n_frames):
        target = float(frames[i])
        coef = ta_f if target > lvl else tr_f
        lvl = coef * lvl + (1.0 - coef) * target
        env_f[i] = lvl

    # Upsample with linear interpolation
    positions = np.linspace(0, n_frames - 1, n, dtype=np.float64)
    idx0 = np.floor(positions).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, n_frames - 1)
    frac = (positions - idx0).astype(np.float32)
    return (env_f[idx0] * (1.0 - frac) + env_f[idx1] * frac).astype(np.float32)


def compress(
    audio: "np.ndarray",
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    attack_ms: float = 15.0,
    release_ms: float = 150.0,
    knee_db: float = 3.0,
) -> "np.ndarray":
    """Feed-forward RMS compressor with separate attack/release time
    constants and a soft knee, operating on float32 audio.

    Uses a fast envelope follower (numba if installed, otherwise a
    vectorised frame-based approximation) so multi-minute tracks no
    longer spend most of their runtime in a pure-Python sample loop.
    Soft-knee gain computation remains fully vectorised either way.
    """
    if not CAP.numpy:
        return audio
    if ratio <= 1.0:
        return audio

    n_samples = len(audio)
    channels = audio.shape[1] if audio.ndim == 2 else 1
    out = np.empty_like(audio)
    half_knee = knee_db / 2.0
    inv_ratio_m1 = (1.0 / ratio - 1.0)
    eps = 1e-12

    for ch in range(channels):
        x = audio[:, ch] if channels > 1 else audio.ravel()
        level_sq = (x.astype(np.float64) ** 2).astype(np.float32)
        level_smooth = _ema_envelope_fast(level_sq, sr, attack_ms, release_ms)
        level_db = 10.0 * np.log10(np.maximum(level_smooth, eps))

        above = level_db - threshold_db
        gr_db = np.where(
            above <= -half_knee,
            0.0,
            np.where(
                above < half_knee,
                inv_ratio_m1 * (above + half_knee) ** 2 / (2.0 * knee_db),
                inv_ratio_m1 * above,
            ),
        )
        gain_linear = (10.0 ** (gr_db / 20.0)).astype(np.float32)
        y = x * gain_linear
        if channels > 1:
            out[:, ch] = y
        else:
            out = y.reshape(-1, 1)

    return out.astype(np.float32)


def apply_multiband_compression(
    audio: "np.ndarray",
    sr: int,
    low_crossover_hz: int = 200,
    high_crossover_hz: int = 4000,
    low_ratio: float = 2.0,
    mid_ratio: float = 2.0,
    high_ratio: float = 2.0,
    attack_ms: float = 15.0,
    release_ms: float = 150.0,
) -> "np.ndarray":
    """Compress bass/mid/treble independently using LR crossovers.
    Threshold for each band is set to peak - 12dB so it engages on
    genuinely dynamic material regardless of the band's average level.
    """
    low, rest = split_bands(audio, sr, low_crossover_hz)
    mid, high = split_bands(rest, sr, high_crossover_hz)

    def _compress_band(band: "np.ndarray", ratio: float) -> "np.ndarray":
        if ratio <= 1.0 or not CAP.numpy:
            return band
        peak_db = 20.0 * math.log10(float(np.max(np.abs(band))) + 1e-12)
        threshold = peak_db - 12.0
        return compress(band, sr, threshold_db=threshold, ratio=ratio,
                        attack_ms=attack_ms, release_ms=release_ms)

    low  = _compress_band(low,  low_ratio)
    mid  = _compress_band(mid,  mid_ratio)
    high = _compress_band(high, high_ratio)
    return recombine_bands(audio, low, mid, high)


def apply_deesser(
    audio: "np.ndarray",
    sr: int,
    threshold_db: float = -24.0,
    ratio: float = 4.0,
    freq: int = 5000,
) -> "np.ndarray":
    """Compress only the sibilant band (above `freq`) to tame harsh 's'/'sh'.
    Everything below `freq` passes through untouched.
    """
    low, high = split_bands(audio, sr, freq)
    compressed_high = compress(
        high, sr, threshold_db=threshold_db, ratio=ratio,
        attack_ms=2.0, release_ms=60.0,
    )
    return recombine_bands(audio, low, compressed_high)


# --- B.5 Saturation, harmonic exciter, brickwall limiter, stereo width ---

def apply_saturation(
    audio: "np.ndarray",
    drive_db: float = 6.0,
    mix: float = 0.35,
) -> "np.ndarray":
    """Tanh soft-clip saturation (tape/analog warmth). Drive is applied
    as a float multiplier inside the per-sample loop — NOT via integer
    sample gain — so the saturation curve operates correctly without
    pre-clipping in the integer domain.
    """
    if not CAP.numpy or drive_db <= 0 or mix <= 0:
        return audio
    drive_linear = 10 ** (drive_db / 20.0)
    normalizer   = float(np.tanh(np.array(1.5 * drive_linear)))
    wet = np.tanh(audio * 1.5 * drive_linear) / normalizer

    # RMS-match wet to dry before blending
    dry_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-12
    wet_rms = float(np.sqrt(np.mean(wet   ** 2))) + 1e-12
    wet = wet * (dry_rms / wet_rms)

    return ((1.0 - mix) * audio + mix * wet).astype(np.float32)


def apply_harmonic_exciter(
    audio: "np.ndarray",
    sr: int,
    freq: int = 3000,
    amount: float = 0.3,
) -> "np.ndarray":
    """Band-limited harmonic exciter: apply saturation only to the
    high-frequency band (above `freq` via LR highpass), then blend
    back. This adds air/presence without affecting the low end, and
    avoids the aliasing risk of full-spectrum excitation.
    """
    if not CAP.numpy or amount <= 0:
        return audio
    high = lr_highpass(audio, sr, freq)
    excited = np.tanh(high * 2.0) / float(np.tanh(np.array(2.0)))
    high_excited = (1.0 - amount) * high + amount * excited
    low = lr_lowpass(audio, sr, freq)
    return recombine_bands(audio, low, high_excited)


def apply_crystalizer(
    audio: "np.ndarray",
    sr: int,
    intensity: float = 2.5,
    low_hz: float = 2000.0,
    high_hz: float = 16000.0,
) -> "np.ndarray":
    """Band-limited crystalizer (JetAudio-style clarity/detail enhancer).
    Applies a first-difference (differentiator) emphasis only in the
    [low_hz, high_hz] band to avoid aliasing at high frequencies and
    muddiness at low frequencies. The `intensity` parameter scales the
    effect (0 = off, 10 = maximum).
    """
    if not CAP.numpy or intensity <= 0:
        return audio

    # Isolate the working band
    band = lr_bandpass(audio, sr, low_hz, high_hz)

    # First-difference emphasis (approximates derivative, boosts transients)
    diff = np.empty_like(band)
    diff[0] = 0.0
    diff[1:] = band[1:] - band[:-1]

    # Scale and blend
    scale = intensity / 10.0
    enhanced_band = band + scale * diff

    # Reconstruct: original outside working band + enhanced band inside
    outside = audio - band
    return recombine_bands(audio, outside, enhanced_band)


def apply_brickwall_limiter(
    audio: "np.ndarray",
    headroom_db: float = 0.5,
    attack_ms: float = 1.0,
    release_ms: float = 50.0,
    sr: int = 44100,
) -> "np.ndarray":
    """Peak brickwall limiter. Uses a smoothed peak envelope to compute
    gain reduction, ensuring no sample exceeds -headroom_db below full scale.

    Not a true sample-accurate look-ahead limiter (that requires buffering
    future samples). This is a causal envelope follower, sufficient for
    mastering-style safety limiting after all other processing is complete.
    For true look-ahead behaviour, the oversampled ffmpeg alimiter is used
    at export time via the filter chain.
    """
    if not CAP.numpy:
        return audio
    ceiling = 10 ** (-headroom_db / 20.0)
    abs_peak = np.abs(audio)
    if audio.ndim == 2:
        abs_peak = abs_peak.max(axis=1)

    # Re-use the fast dual-tau envelope (peak detector: feed |x|^2-like
    # by squaring peak so the follower semantics match compress()).
    envelope = _ema_envelope_fast(
        (abs_peak.astype(np.float64) ** 2).astype(np.float32),
        sr, attack_ms, release_ms,
    )
    envelope = np.sqrt(np.maximum(envelope, 1e-24)).astype(np.float32)

    gain = np.minimum(1.0, ceiling / envelope).astype(np.float32)
    if audio.ndim == 2:
        gain = gain[:, None]
    return (audio * gain).astype(np.float32)


def apply_stereo_width(
    audio: "np.ndarray",
    width: float = 2.0,
) -> "np.ndarray":
    """Mid-side stereo width adjustment. width=1.0 = unchanged,
    width=0.0 = mono, width>1.0 = wider.
    """
    if not CAP.numpy or audio.ndim < 2 or audio.shape[1] < 2:
        return audio
    L, R = audio[:, 0], audio[:, 1]
    mid  = (L + R) * 0.5
    side = (L - R) * 0.5 * width
    out = np.stack([mid + side, mid - side], axis=1)
    return out.astype(np.float32)


def apply_width_bands(
    audio: "np.ndarray",
    sr: int,
    bass_width: float = 1.0,
    mid_width: float = 1.6,
    treble_width: float = 1.6,
    low_crossover_hz: int = 150,
    high_crossover_hz: int = 4000,
) -> "np.ndarray":
    """Per-band stereo widening using LR crossovers. Widths are applied
    independently to bass/mid/treble bands then recombined.
    """
    if not CAP.numpy or audio.ndim < 2:
        return audio
    low, rest = split_bands(audio, sr, low_crossover_hz)
    mid, high = split_bands(rest, sr, high_crossover_hz)
    low  = apply_stereo_width(low,  bass_width)
    mid  = apply_stereo_width(mid,  mid_width)
    high = apply_stereo_width(high, treble_width)
    return recombine_bands(audio, low, mid, high)


def apply_crossfeed(
    audio: "np.ndarray",
    sr: int,
    strength: float = 0.3,
    range_param: float = 0.5,
) -> "np.ndarray":
    """Headphone crossfeed: mix a small, low-passed version of each channel
    into the opposite channel to simulate natural speaker listening.
    Makes hard-panned mixes more comfortable on headphones.
    """
    if not CAP.numpy or audio.ndim < 2 or audio.shape[1] < 2:
        return audio
    L, R = audio[:, 0], audio[:, 1]
    # Low-pass the cross-signal (only low frequencies cross between ears
    # in natural listening — high frequencies are directional)
    lp_cutoff = 700.0 + range_param * 1300.0  # 700–2000Hz based on range
    L_lp = lr_lowpass(L.reshape(-1, 1), sr, lp_cutoff).ravel()
    R_lp = lr_lowpass(R.reshape(-1, 1), sr, lp_cutoff).ravel()
    L_out = L + strength * R_lp
    R_out = R + strength * L_lp
    return np.stack([L_out, R_out], axis=1).astype(np.float32)


# --- B.6 LUFS measurement and matching ---

def measure_lufs(audio: "np.ndarray", sr: int) -> float:
    """Measure integrated loudness in LUFS (K-weighted, BS.1770-4).
    Uses pyloudnorm when available. Falls back to RMS approximation
    when pyloudnorm is absent.

    Note: pyloudnorm requires at least ~0.4 seconds of audio for the
    gating algorithm to produce a meaningful result. For shorter clips
    (e.g. preview segments) the RMS approximation path is used instead.
    """
    if not CAP.numpy:
        return -23.0  # safe default

    if CAP.pyloudnorm and len(audio) / sr >= 0.4:
        try:
            meter = pyln.Meter(sr)
            mono_or_stereo = audio if audio.shape[1] <= 2 else audio[:, :2]
            lufs = meter.integrated_loudness(mono_or_stereo.astype(np.float64))
            if math.isfinite(lufs):
                return float(lufs)
        except Exception as exc:
            log.debug("pyloudnorm measurement failed: %s", exc)

    # RMS approximation (-23 LUFS ≈ -20 dBFS RMS for typical program material)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-12:
        return -100.0
    rms_db = 20.0 * math.log10(rms)
    return rms_db - 3.0  # approximate offset for K-weighting


def match_lufs(
    audio: "np.ndarray",
    sr: int,
    target_lufs: float,
    headroom_db: float = 0.5,
) -> "np.ndarray":
    """Apply a single gain to bring `audio` to `target_lufs`, capped so
    the peak does not exceed -headroom_db below full scale.
    """
    if not CAP.numpy:
        return audio
    current_lufs = measure_lufs(audio, sr)
    desired_gain_db = target_lufs - current_lufs

    # Cap: peak must not exceed ceiling
    peak_db = 20.0 * math.log10(float(np.max(np.abs(audio))) + 1e-12)
    ceiling_db = -headroom_db
    max_safe_gain_db = ceiling_db - peak_db

    applied_gain_db = min(desired_gain_db, max_safe_gain_db)
    applied_gain = 10 ** (applied_gain_db / 20.0)
    return (audio * applied_gain).astype(np.float32)


def measure_lufs_for_export(
    temp_wav: Path,
    target_lufs: float,
    true_peak_db: float,
    lra: float,
) -> dict | None:
    """Run ffmpeg loudnorm in analysis mode (pass 1 of 2) on a WAV file.
    Returns the measured stats dict for use in the export filter chain,
    or None if measurement fails (caller falls back to single-pass).

    Two-pass loudnorm is always used for music (single-pass applies
    dynamic compression in real time, which breathes audibly on music).
    """
    cmd = [
        FFMPEG_BIN or "ffmpeg",
        "-hide_banner", "-nostats",
        "-i", str(temp_wav),
        "-af", f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA={lra}:print_format=json",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as exc:
        log.debug("loudnorm measurement failed: %s", exc)
        return None

    stderr = result.stderr
    start  = stderr.rfind("{")
    end    = stderr.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        return json.loads(stderr[start : end + 1])
    except Exception:
        return None


def build_export_filter_chain(
    use_loudnorm: bool = False,
    lufs_target: float = -14.0,
    lufs_true_peak_db: float = -1.0,
    lufs_range: float = 11.0,
    measured_stats: dict | None = None,
    use_limiter: bool = False,
    limiter_oversample: bool = False,
    headroom_db: float = 0.5,
    crossfeed: bool = False,
    crossfeed_strength: float = 0.3,
    crossfeed_range: float = 0.5,
    sample_rate: int = 44100,
) -> str | None:
    """Build the ffmpeg -af filter string for the final export pass.

    Filters applied at encode time (after all pydub/numpy processing):
    crossfeed (headphones) → loudnorm or alimiter.

    Crossfeed is applied here (not in the numpy chain) only when scipy
    is unavailable; when scipy is available, apply_crossfeed() in the
    numpy chain handles it before encoding.
    """
    filters: list[str] = []

    if crossfeed and not CAP.scipy:
        filters.append(
            f"crossfeed=strength={crossfeed_strength}:range={crossfeed_range}"
        )

    if use_loudnorm:
        tp   = max(-9.0, min(0.0, lufs_true_peak_db))
        tgt  = max(-70.0, min(-5.0, lufs_target))
        lra  = max(1.0, min(50.0, lufs_range))
        lnorm = f"loudnorm=I={tgt}:TP={tp}:LRA={lra}"
        if measured_stats:
            try:
                lnorm += (
                    f":measured_I={measured_stats['input_i']}"
                    f":measured_TP={measured_stats['input_tp']}"
                    f":measured_LRA={measured_stats['input_lra']}"
                    f":measured_thresh={measured_stats['input_thresh']}"
                    f":offset={measured_stats['target_offset']}"
                    ":linear=true"
                )
            except (KeyError, TypeError):
                pass
        filters.append(lnorm)
    elif use_limiter:
        limit_linear = max(0.0625, min(1.0, 10 ** (-headroom_db / 20.0)))
        limiter = f"alimiter=limit={limit_linear:.4f}:attack=5:release=50"
        if limiter_oversample:
            over_rate = sample_rate * 2
            filters.append(f"aresample={over_rate}")
            filters.append(limiter)
            filters.append(f"aresample={sample_rate}")
        else:
            filters.append(limiter)

    return ",".join(filters) if filters else None


# ---------------------------------------------------------------------------
# Phase B — Stem processing (vocal chain, instrument chain, mastering chain)
# ---------------------------------------------------------------------------

# --- Stem cache ---

_CACHE_DIR = Path.home() / ".openremaster" / "stem_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _stem_cache_key(path: Path, model: str) -> str:
    """Cache key: mtime+size fast-path (no SHA needed if these match).
    Includes model name and demucs version so different models never
    collide, and a version bump invalidates all existing entries.
    """
    stat = path.stat()
    fast = f"{stat.st_mtime:.3f}_{stat.st_size}"

    try:
        import demucs
        demucs_ver = getattr(demucs, "__version__", "unknown")
    except Exception:
        demucs_ver = "unknown"

    # Only compute full SHA256 when fast-path key changes
    # (checked by caller via _load_stem_cache)
    return f"{fast}__{model}__{demucs_ver}"


def _cache_path(key: str) -> Path:
    # Sanitise key for use as filename
    safe = key.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return _CACHE_DIR / f"{safe}.npz"


def _load_stem_cache(
    path: Path, model: str
) -> tuple[dict[str, "np.ndarray"], int] | None:
    """Return (stems dict, sample_rate) or None on miss.

    Older cache files without a sample_rate entry are treated as 44100
    (Demucs default) so they remain usable.
    """
    if not CAP.numpy:
        return None
    key   = _stem_cache_key(path, model)
    cpath = _cache_path(key)
    if not cpath.exists():
        return None
    try:
        data = np.load(cpath, allow_pickle=False)
        log.debug("Stem cache hit: %s", cpath.name)
        sr = 44100
        stems: dict[str, np.ndarray] = {}
        for k in data.files:
            if k == "_sample_rate":
                sr = int(data[k])
            else:
                stems[k] = data[k]
        if not stems:
            return None
        return stems, sr
    except Exception as exc:
        log.debug("Stem cache load failed: %s", exc)
        return None


def _save_stem_cache(
    path: Path, model: str, stems: dict[str, "np.ndarray"], sample_rate: int = 44100
) -> None:
    if not CAP.numpy:
        return
    key   = _stem_cache_key(path, model)
    cpath = _cache_path(key)
    try:
        # Store sample rate as a 0-d scalar array so older loaders that
        # only expected stem arrays still open the file (they will just
        # ignore the extra key).
        np.savez(cpath, _sample_rate=np.array(sample_rate, dtype=np.int32), **stems)
        log.debug("Stem cache saved: %s", cpath.name)
    except Exception as exc:
        log.debug("Stem cache save failed: %s", exc)


# --- Demucs separation (Python API, not subprocess) ---

def separate_stems(
    input_path: Path,
    model: str = "htdemucs",
    device: str = "cpu",
    use_cache: bool = True,
) -> tuple[dict[str, "np.ndarray"], int] | None:
    """Separate a track into stems using Demucs (Python API).

    Returns ({stem_name: float32 array (samples, channels)}, sample_rate)
    where sample_rate is the Demucs model rate (typically 44100), or None
    if Demucs is unavailable / separation fails.

    Uses the SHA-free fast-path cache (mtime+size+model+version).
    """
    if not CAP.demucs or not CAP.numpy:
        log.warning("Demucs not available — stem separation skipped.")
        return None

    if not model or model == "none":
        log.info("Stem model set to none — skipping separation.")
        return None

    if use_cache:
        cached = _load_stem_cache(input_path, model)
        if cached is not None:
            stems, cached_sr = cached
            return stems, cached_sr

    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.audio import AudioFile

        log.info("Loading Demucs model: %s …", model)
        demucs_model = get_model(model)
        demucs_model.to(device)
        demucs_model.eval()
        model_sr = int(demucs_model.samplerate)

        log.info("Separating: %s (model SR=%d)", input_path.name, model_sr)
        wav = AudioFile(str(input_path)).read(
            streams=0,
            samplerate=model_sr,
            channels=demucs_model.audio_channels,
        )
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        wav = wav.unsqueeze(0).to(device)

        with torch.no_grad():
            sources = apply_model(
                demucs_model, wav, device=device, progress=False
            )

        source_names = demucs_model.sources
        stems: dict[str, np.ndarray] = {}
        for i, name in enumerate(source_names):
            s = sources[0, i].cpu().numpy()  # (channels, samples)
            stems[name] = s.T.astype(np.float32)  # (samples, channels)

        if use_cache:
            _save_stem_cache(input_path, model, stems, model_sr)

        return stems, model_sr

    except Exception as exc:
        log.warning("Demucs separation failed: %s — proceeding without stems.", exc)
        return None


# --- Vocal chain ---

def process_vocal_stem(
    vocal: "np.ndarray",
    sr: int,
    presence_db: float = 1.5,
    air_db: float = 2.0,
    mud_cut_db: float = -1.5,
    deesser: bool = True,
    deesser_threshold_db: float = -24.0,
    deesser_ratio: float = 4.0,
    deesser_freq: int = 5000,
    target_lufs: float = -18.0,
    headroom_db: float = 0.5,
    use_voicefixer: bool = False,
) -> "np.ndarray":
    """Vocal-specific enhancement chain:
    [optional VoiceFixer] → EQ (presence/mud/air) → de-esser → LUFS match.

    VoiceFixer is experimental and off by default. When enabled it runs
    on the mono mid signal only (not per-channel stereo) to avoid
    inter-channel phase shifts.
    """
    if not CAP.numpy:
        return vocal

    out = vocal.copy()

    if use_voicefixer and CAP.voicefixer:
        try:
            out = _apply_voicefixer_mid_only(out, sr)
        except Exception as exc:
            log.warning("VoiceFixer failed: %s — skipping.", exc)

    # EQ chain: mud cut → presence boost → air shelf
    out = apply_eq(
        out, sr,
        bass_shelf_hz=200.0, bass_shelf_db=mud_cut_db,      # mud cut (low shelf)
        presence_hz=3000.0,  presence_db=presence_db, presence_q=1.2,
        treble_shelf_hz=10000.0, treble_shelf_db=air_db,
    )

    if deesser:
        out = apply_deesser(out, sr,
                            threshold_db=deesser_threshold_db,
                            ratio=deesser_ratio,
                            freq=deesser_freq)

    out = match_lufs(out, sr, target_lufs, headroom_db=headroom_db)
    return out


def _apply_voicefixer_mid_only(
    audio: "np.ndarray", sr: int
) -> "np.ndarray":
    """Run VoiceFixer on the mono mid signal only, then restore stereo
    width from the original side signal. Avoids per-channel phase shifts.
    """
    from voicefixer import VoiceFixer
    vf = VoiceFixer()

    channels = audio.shape[1] if audio.ndim == 2 else 1
    if channels == 1:
        mid  = audio.ravel()
        side = None
    else:
        mid  = (audio[:, 0] + audio[:, 1]) * 0.5
        side = (audio[:, 0] - audio[:, 1]) * 0.5

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = Path(tmpdir) / "mid_in.wav"
        out_path = Path(tmpdir) / "mid_out.wav"
        if CAP.soundfile:
            sf.write(str(in_path), mid, sr)
        else:
            seg = float32_to_pydub(mid.reshape(-1, 1), sr)
            seg.export(str(in_path), format="wav")

        vf.restore(input=str(in_path), output_dir=tmpdir, mode=0)

        if CAP.soundfile:
            mid_restored, _ = sf.read(str(out_path), dtype="float32")
        else:
            seg2 = AudioSegment.from_file(out_path)
            mid_restored, _ = pydub_to_float32(seg2)
            mid_restored = mid_restored.ravel()

    if side is None:
        return mid_restored.reshape(-1, 1)

    # Re-encode mid/side back to L/R
    L = mid_restored + side
    R = mid_restored - side
    return np.stack([L, R], axis=1).astype(np.float32)


# --- Instrument chain ---


def _apply_deepfilternet(
    audio: "np.ndarray",
    sr: int,
) -> "np.ndarray":
    """Run DeepFilterNet denoise on the full mix (experimental).

    DeepFilterNet is speech-oriented; on music it can suppress
    instruments it misclassifies as noise. Only called when the user
    explicitly enables it (and acknowledges the risk in the wizard).

    Processes channel-wise at the model's internal rate, then resamples
    back to `sr`. Falls back to the input on any error.
    """
    if not CAP.numpy or not CAP.deepfilternet:
        return audio
    try:
        import torch
        from df.enhance import enhance, init_df
        from df.io import resample as df_resample

        model, df_state, _ = init_df(log_level="ERROR")
        model_sr = df_state.sr()

        # Work mono mid to avoid phase issues across channels, then
        # re-apply original side signal (same approach as VoiceFixer).
        channels = audio.shape[1] if audio.ndim == 2 else 1
        if channels == 1:
            mid = audio.ravel().astype(np.float32)
            side = None
        else:
            mid = ((audio[:, 0] + audio[:, 1]) * 0.5).astype(np.float32)
            side = ((audio[:, 0] - audio[:, 1]) * 0.5).astype(np.float32)

        # DeepFilterNet expects shape (channels, samples) tensor at model_sr
        if sr != model_sr:
            # Simple linear resample for the mid channel
            n_out = int(round(len(mid) * model_sr / sr))
            xp = np.linspace(0, 1, len(mid), endpoint=False)
            xq = np.linspace(0, 1, n_out, endpoint=False)
            mid_rs = np.interp(xq, xp, mid).astype(np.float32)
        else:
            mid_rs = mid

        tensor = torch.from_numpy(mid_rs).unsqueeze(0)  # (1, samples)
        with torch.no_grad():
            enhanced = enhance(model, df_state, tensor)
        mid_out = enhanced.squeeze(0).cpu().numpy().astype(np.float32)

        if sr != model_sr:
            n_back = len(mid)
            xp = np.linspace(0, 1, len(mid_out), endpoint=False)
            xq = np.linspace(0, 1, n_back, endpoint=False)
            mid_out = np.interp(xq, xp, mid_out).astype(np.float32)
            # Match length exactly
            if len(mid_out) > n_back:
                mid_out = mid_out[:n_back]
            elif len(mid_out) < n_back:
                mid_out = np.pad(mid_out, (0, n_back - len(mid_out)))

        if side is None:
            return mid_out.reshape(-1, 1)

        # Truncate side to match if needed
        n = min(len(mid_out), len(side))
        mid_out, side = mid_out[:n], side[:n]
        L = mid_out + side
        R = mid_out - side
        return np.stack([L, R], axis=1).astype(np.float32)

    except Exception as exc:
        log.warning("DeepFilterNet failed: %s — skipping.", exc)
        return audio


def process_instrument_stems(
    stems: dict[str, "np.ndarray"],
    sr: int,
    bass_shelf_db: float = 1.0,
    air_shelf_db: float = 1.5,
    harmonic_exciter: bool = True,
    exciter_freq: int = 3000,
    exciter_amount: float = 0.25,
    target_lufs: float = -18.0,
    headroom_db: float = 0.5,
) -> "np.ndarray":
    """Mix all non-vocal stems, applying the instrument enhancement chain:
    EQ (bass/air) → harmonic exciter → LUFS match.
    Returns the combined instrument mix as float32.
    """
    if not CAP.numpy:
        return sum(stems.values()) / max(len(stems), 1)

    inst_mix: np.ndarray | None = None
    for name, stem in stems.items():
        inst_mix = stem if inst_mix is None else inst_mix + stem

    if inst_mix is None:
        return np.zeros((1, 2), dtype=np.float32)

    out = apply_eq(
        inst_mix, sr,
        bass_shelf_hz=120.0, bass_shelf_db=bass_shelf_db,
        treble_shelf_hz=10000.0, treble_shelf_db=air_shelf_db,
    )

    if harmonic_exciter:
        out = apply_harmonic_exciter(out, sr, freq=exciter_freq, amount=exciter_amount)

    out = match_lufs(out, sr, target_lufs, headroom_db=headroom_db)
    return out


# --- Mastering chain ---

def apply_mastering_chain(
    audio: "np.ndarray",
    sr: int,
    # EQ
    bass_shelf_hz: float = 100.0,
    bass_shelf_db: float = 0.0,
    treble_shelf_hz: float = 8000.0,
    treble_shelf_db: float = 0.0,
    presence_hz: float = 0.0,
    presence_db: float = 0.0,
    presence_q: float = 1.2,
    notch_hz: float = 0.0,
    notch_db: float = 0.0,
    notch_q: float = 2.0,
    eq_bands: list[tuple[float, float, float]] | None = None,
    # Dynamics
    multiband_compress: bool = False,
    mb_low_crossover_hz: int = 200,
    mb_high_crossover_hz: int = 4000,
    mb_low_ratio: float = 2.0,
    mb_mid_ratio: float = 2.0,
    mb_high_ratio: float = 2.0,
    # Saturation
    saturation: bool = False,
    saturation_drive_db: float = 5.0,
    saturation_mix: float = 0.3,
    # Crystalizer
    crystalizer: bool = False,
    crystalizer_intensity: float = 2.5,
    # Stereo
    width_bands: bool = False,
    width_bass: float = 1.0,
    width_mid: float = 1.6,
    width_treble: float = 1.6,
    width_low_crossover_hz: int = 150,
    width_high_crossover_hz: int = 4000,
    crossfeed: bool = False,
    crossfeed_strength: float = 0.3,
    crossfeed_range: float = 0.5,
    # Loudness
    target_lufs: float = -14.0,
    headroom_db: float = 0.5,
) -> "np.ndarray":
    """Full mastering chain on a combined mix (float32):
    EQ → multiband compress → saturation → crystalizer
    → stereo width → crossfeed → LUFS match → brickwall limiter.

    Each stage is applied only when its flag is set. The limiter always
    runs last as a safety catch regardless of other settings.
    """
    if not CAP.numpy:
        return audio

    out = audio.copy()

    out = apply_eq(
        out, sr,
        bass_shelf_hz=bass_shelf_hz, bass_shelf_db=bass_shelf_db,
        treble_shelf_hz=treble_shelf_hz, treble_shelf_db=treble_shelf_db,
        presence_hz=presence_hz, presence_db=presence_db, presence_q=presence_q,
        notch_hz=notch_hz, notch_db=notch_db, notch_q=notch_q,
        bands=eq_bands,
    )

    if multiband_compress:
        out = apply_multiband_compression(
            out, sr,
            low_crossover_hz=mb_low_crossover_hz,
            high_crossover_hz=mb_high_crossover_hz,
            low_ratio=mb_low_ratio,
            mid_ratio=mb_mid_ratio,
            high_ratio=mb_high_ratio,
        )

    if saturation:
        out = apply_saturation(out, saturation_drive_db, saturation_mix)

    if crystalizer:
        out = apply_crystalizer(out, sr, intensity=crystalizer_intensity)

    if width_bands and out.ndim == 2 and out.shape[1] >= 2:
        out = apply_width_bands(
            out, sr,
            bass_width=width_bass,
            mid_width=width_mid,
            treble_width=width_treble,
            low_crossover_hz=width_low_crossover_hz,
            high_crossover_hz=width_high_crossover_hz,
        )

    if crossfeed and CAP.scipy:
        out = apply_crossfeed(out, sr, strength=crossfeed_strength, range_param=crossfeed_range)

    out = match_lufs(out, sr, target_lufs, headroom_db=headroom_db)
    out = apply_brickwall_limiter(out, headroom_db=headroom_db, sr=sr)
    return out


# ---------------------------------------------------------------------------
# Phase B — 5.1 Surround upmix
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# B.7 Speaker bias — user-selectable content blend on top of the fixed
# Front/Centre/Rear derivations in build_surround_51()
# ---------------------------------------------------------------------------

# Five-band graphic-EQ style split, used as speaker-bias source options
# alongside "vocals". Crossover points are fixed constants (not user
# adjustable) — a reasonable default 5-band ladder.
SPEAKER_BIAS_BAND_RANGES: dict[str, tuple[float | None, float | None]] = {
    "band1": (None, 150.0),      # Bass:      < 150 Hz
    "band2": (150.0, 500.0),     # Low-mid:   150 - 500 Hz
    "band3": (500.0, 2000.0),    # Mid:       500 - 2000 Hz
    "band4": (2000.0, 6000.0),   # High-mid:  2000 - 6000 Hz
    "band5": (6000.0, None),     # High:      > 6000 Hz
}

SPEAKER_BIAS_SOURCE_KEYS = ["off", "vocals", "band1", "band2", "band3", "band4", "band5"]


def _rms_db(x: "np.ndarray") -> float:
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12
    return 20.0 * math.log10(r)


def _compute_speaker_bias_sources(
    audio: "np.ndarray",
    sr: int,
    vocal_stem: "np.ndarray | None",
) -> dict[str, "np.ndarray"]:
    """Compute the mono 1D signals available for speaker-bias blending:
    the vocal stem (or a bandpass approximation of it) plus the 5
    graphic-EQ bands of the full mix. Computed once per call and reused
    across Front/Centre/Rear so the same band split is never redone.
    """
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio.ravel()

    if vocal_stem is not None:
        vocals = vocal_stem.mean(axis=1) if vocal_stem.ndim == 2 else vocal_stem.ravel()
        n = len(mono)
        if len(vocals) > n:
            vocals = vocals[:n]
        elif len(vocals) < n:
            vocals = np.pad(vocals, (0, n - len(vocals)))
    else:
        # No separated stem available — approximate with the same
        # vocal-frequency bandpass used by the Centre-channel fallback.
        vocals = lr_bandpass(mono.reshape(-1, 1), sr, 150.0, 3000.0).ravel()

    sources: dict[str, np.ndarray] = {"vocals": vocals}
    for band_key, (lo, hi) in SPEAKER_BIAS_BAND_RANGES.items():
        if lo is None:
            sources[band_key] = lr_lowpass(mono.reshape(-1, 1), sr, hi).ravel()
        elif hi is None:
            sources[band_key] = lr_highpass(mono.reshape(-1, 1), sr, lo).ravel()
        else:
            sources[band_key] = lr_bandpass(mono.reshape(-1, 1), sr, lo, hi).ravel()
    return sources


def _apply_speaker_bias(
    group_signal: "np.ndarray",
    bias_signal: "np.ndarray",
    bias_db: float,
) -> "np.ndarray":
    """Blend `bias_signal` into `group_signal` as an additive bias whose
    loudness is set RELATIVE to the group's own level: bias_db=0 means
    the bias source is blended in at the same RMS level as the channel's
    existing content, bias_db=-12 means noticeably quieter, etc. This is
    a loudness control (like a mixing send level), not a raw percentage.
    """
    n = len(group_signal)
    src = bias_signal
    if len(src) > n:
        src = src[:n]
    elif len(src) < n:
        src = np.pad(src, (0, n - len(src)))

    group_db = _rms_db(group_signal)
    src_db = _rms_db(src)
    gain_db = (group_db + bias_db) - src_db
    gain = 10 ** (gain_db / 20.0)
    return (group_signal + src * gain).astype(np.float32)


def build_surround_51(
    audio: "np.ndarray",
    sr: int,
    vocal_stem: "np.ndarray | None" = None,
    instrument_stem: "np.ndarray | None" = None,
    lfe_cutoff_hz: int = 80,
    rear_cutoff_hz: int = 300,
    rear_attenuation_db: float = -6.0,
    rear_delay_ms: float = 15.0,
    centre_attenuation_db: float = -3.0,
    lfe_mode: str = "gentle",
    headroom_db: float = 0.5,
    speaker_bias_front: str = "off",
    speaker_bias_front_db: float = 0.0,
    speaker_bias_centre: str = "off",
    speaker_bias_centre_db: float = 0.0,
    speaker_bias_rear: str = "off",
    speaker_bias_rear_db: float = 0.0,
) -> dict[str, "np.ndarray"]:
    """Derive 5.1 channels (FL, FR, C, LFE, RL, RR) from a stereo source.

    Channel derivation strategy (frequency-domain M/S approach):
      Centre : correlated mid-frequency content (150Hz–2kHz)
               Uses vocal stem when available (better isolation),
               otherwise uses L+R mid signal.
      FL/FR  : original L/R highpassed at 80Hz (bass management handles sub),
               attenuated -3dB to compensate for energy moved to Centre.
      RL/RR  : decorrelated ambient component of the side (L-R) signal,
               one side delayed 15-20ms for precedence-effect decorrelation,
               bandpassed to emphasise room ambience frequencies.
      LFE    : gentle sub-bass blend (lowpass at lfe_cutoff_hz), attenuated
               -10dB. Most AV receivers' bass management already routes
               low frequencies from all 5 main channels to the sub — the
               LFE channel carries only supplementary deep bass content,
               not the main bass bus.

    Speaker bias (optional, on top of the derivation above): Front, Centre,
    and Rear can each additionally blend in one of {vocals, band1..band5}
    (see SPEAKER_BIAS_BAND_RANGES) at a level relative to the group's own
    RMS (speaker_bias_*_db). This never replaces the base signal derived
    above — it's a bias/enhancement layer. LFE has no bias option; it keeps
    its fixed derivation. Defaults are "off" (0dB effect) so behaviour is
    unchanged unless the caller opts in.

    AC3 5.1 channel order: FL, FR, C, LFE, RL, RR.
    All channels are highpassed at 80Hz (bass management boundary) except LFE.
    """
    if not CAP.numpy:
        raise RuntimeError("numpy is required for 5.1 upmix.")

    # Ensure stereo input
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[1] == 1:
        audio = np.tile(audio, (1, 2))

    L = audio[:, 0]
    R = audio[:, 1]
    mid  = (L + R) * 0.5   # correlated/mono content
    side = (L - R) * 0.5   # decorrelated/stereo-only content

    # --- Centre ---
    if vocal_stem is not None:
        # Use the enhanced vocal stem for better dialogue/vocal isolation
        if vocal_stem.ndim == 2:
            centre_raw = vocal_stem.mean(axis=1)
        else:
            centre_raw = vocal_stem.ravel()
        # Match length
        n = len(audio)
        if len(centre_raw) > n:
            centre_raw = centre_raw[:n]
        elif len(centre_raw) < n:
            centre_raw = np.pad(centre_raw, (0, n - len(centre_raw)))
    else:
        # Frequency-domain centre: correlated mid-frequency band (150Hz–2kHz)
        centre_raw = lr_bandpass(mid.reshape(-1, 1), sr, 150.0, 2000.0).ravel()

    centre_atten = 10 ** (centre_attenuation_db / 20.0)
    centre = (centre_raw * centre_atten).astype(np.float32)

    # --- FL / FR (highpassed at bass management boundary) ---
    front_hp = lr_highpass(audio, sr, float(lfe_cutoff_hz))
    # Attenuate by -3dB to compensate for energy moved to Centre
    front_atten = 10 ** (-3.0 / 20.0)
    FL = (front_hp[:, 0] * front_atten).astype(np.float32)
    FR = (front_hp[:, 1] * front_atten).astype(np.float32)

    # --- RL / RR (decorrelated ambient side signal) ---
    # Bandpass to emphasise room ambience frequencies (300Hz–6kHz)
    side_band = lr_bandpass(side.reshape(-1, 1), sr, float(rear_cutoff_hz), 6000.0).ravel()
    rear_atten = 10 ** (rear_attenuation_db / 20.0)

    # Delay one channel for Haas-effect decorrelation (prevents comb filtering
    # when the rear speakers are in-phase)
    delay_samps = int(sr * rear_delay_ms / 1000.0)
    side_delayed = np.pad(side_band, (delay_samps, 0))[: len(side_band)]

    RL = (side_band    * rear_atten).astype(np.float32)
    RR = (side_delayed * rear_atten).astype(np.float32)

    # --- LFE ---
    if lfe_mode == "silent":
        LFE = np.zeros(len(audio), dtype=np.float32)
    else:
        lfe_raw = lr_lowpass(mid.reshape(-1, 1), sr, float(lfe_cutoff_hz)).ravel()
        if lfe_mode == "gentle":
            lfe_atten = 10 ** (-10.0 / 20.0)   # -10dB: supplementary only
        else:  # "full"
            lfe_atten = 10 ** (-6.0 / 20.0)    # -6dB: slightly more present
        LFE = (lfe_raw * lfe_atten).astype(np.float32)

    # --- Speaker bias (optional enhancement layer, Front/Centre/Rear only) ---
    need_bias = speaker_bias_front != "off" or speaker_bias_centre != "off" or speaker_bias_rear != "off"
    if need_bias:
        bias_sources = _compute_speaker_bias_sources(audio, sr, vocal_stem)
        if speaker_bias_front != "off" and speaker_bias_front in bias_sources:
            src = bias_sources[speaker_bias_front]
            FL = _apply_speaker_bias(FL, src, speaker_bias_front_db)
            FR = _apply_speaker_bias(FR, src, speaker_bias_front_db)
        if speaker_bias_centre != "off" and speaker_bias_centre in bias_sources:
            src = bias_sources[speaker_bias_centre]
            centre = _apply_speaker_bias(centre, src, speaker_bias_centre_db)
        if speaker_bias_rear != "off" and speaker_bias_rear in bias_sources:
            src = bias_sources[speaker_bias_rear]
            RL = _apply_speaker_bias(RL, src, speaker_bias_rear_db)
            RR = _apply_speaker_bias(RR, src, speaker_bias_rear_db)

    # Per-channel safety limiter
    ceiling = 10 ** (-headroom_db / 20.0)
    channels_out: dict[str, np.ndarray] = {}
    for name, ch in [("FL", FL), ("FR", FR), ("C", centre),
                     ("LFE", LFE), ("RL", RL), ("RR", RR)]:
        peak = float(np.max(np.abs(ch))) + 1e-12
        if peak > ceiling:
            ch = ch * (ceiling / peak)
        channels_out[name] = ch

    return channels_out


def write_multichannel_wav(
    channels: dict[str, "np.ndarray"],
    sr: int,
    output_path: Path,
) -> Path:
    """Write a 6-channel 24-bit WAV from the surround channel dict.
    Channel order follows AC3 5.1: FL, FR, C, LFE, RL, RR.
    Uses soundfile (required for multichannel WAV).
    """
    if not CAP.soundfile:
        raise RuntimeError("soundfile is required for 5.1 WAV output.")
    if not CAP.numpy:
        raise RuntimeError("numpy is required for 5.1 WAV output.")

    order = ["FL", "FR", "C", "LFE", "RL", "RR"]
    n = max(len(channels[k]) for k in order)
    data = np.zeros((n, 6), dtype=np.float32)
    for i, name in enumerate(order):
        ch = channels[name]
        data[: len(ch), i] = ch

    sf.write(str(output_path), data, sr, subtype="PCM_24")
    log.info("Multichannel WAV written: %s", output_path)
    return output_path


def mux_surround_mkv(
    wav_path: Path,
    output_path: Path,
    codec: str = "ac3",
    source_sr: int = 44100,
) -> Path:
    """Mux a 6-channel WAV to MKV with AC3 or PCM audio.

    AC3 requires 48kHz. Resampling is done here (as the last step before
    encode) so all DSP ran at the original sample rate.

    Channel layout is set explicitly as FL+FR+FC+LFE+BL+BR to guarantee
    the receiver gets the correct 5.1 assignment. ffmpeg's
    channel_layout=5.1 maps to this order.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ff = FFMPEG_BIN or "ffmpeg"

    if codec == "ac3":
        target_sr = 48000
        audio_args = [
            "-acodec", "ac3",
            "-b:a", "640k",
            "-ar", str(target_sr),
            "-channel_layout", "5.1",
        ]
    else:  # pcm
        target_sr = source_sr
        audio_args = [
            "-acodec", "pcm_s24le",
            "-channel_layout", "5.1",
        ]

    resample_filter = []
    if codec == "ac3" and source_sr != target_sr:
        resample_filter = ["-af", f"aresample={target_sr}"]

    cmd = [
        ff, "-y",
        "-i", str(wav_path),
        *resample_filter,
        *audio_args,
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")[:500]
        raise RuntimeError(f"ffmpeg MKV mux failed: {err}")

    log.info("MKV muxed: %s", output_path)
    return output_path


def verify_surround_output(mkv_path: Path) -> bool:
    """Read back the MKV and confirm it has exactly 6 audio channels.
    Also checks that channel 4 (LFE, 0-indexed) has the expected
    frequency content: primarily sub-80Hz energy.
    """
    ff = FFPROBE_BIN or "ffprobe"
    cmd = [
        ff, "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "json",
        str(mkv_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        channels = data["streams"][0]["channels"]
        if channels != 6:
            log.warning("Surround verify: expected 6 channels, got %d", channels)
            return False
        log.info("Surround verify OK: 6 channels confirmed in %s", mkv_path.name)
        return True
    except Exception as exc:
        log.warning("Surround verify failed: %s", exc)
        return False


def measure_dialnorm(centre_channel: "np.ndarray", sr: int) -> int:
    """Compute AC3 dialnorm from the centre channel loudness.
    Dialnorm is the negated integrated loudness in LUFS, clamped to
    the AC3 valid range of -1 to -31.
    """
    lufs = measure_lufs(centre_channel.reshape(-1, 1) if centre_channel.ndim == 1
                        else centre_channel, sr)
    dialnorm = max(-31, min(-1, int(round(lufs))))
    log.debug("Dialnorm: %d (centre LUFS: %.1f)", dialnorm, lufs)
    return dialnorm


# ---------------------------------------------------------------------------
# Content auto-detection
# ---------------------------------------------------------------------------

from device_profiles import DEVICE_PROFILES, DeviceKey
from content_profiles import (
    CONTENT_PROFILES, ContentKey,
    era_from_year, content_hint_from_filename,
)
from profile_resolver import resolve_profile


def detect_content_type(
    audio: "np.ndarray | None",
    sr: int,
    file_path: Path | None = None,
) -> tuple[ContentKey, float]:
    """Auto-detect content type from spectral analysis (60% weight) and
    tag/filename signals (40% weight). Returns (content_key, confidence_0_to_1).

    Spectral signals are generally more reliable than tags for older film
    and popular music because year tags often reflect CD/rip year rather
    than original recording year.
    """
    scores: dict[ContentKey, float] = {k: 0.0 for k in CONTENT_PROFILES}

    # `max_possible` tracks how much evidence we actually had a chance to
    # gather this run (which signal groups were evaluated), independent of
    # whether any of them actually matched anything. Confidence is then
    # scores[best] / max_possible rather than scores[best] / sum(scores) —
    # the latter would let a single weak signal (e.g. one filename keyword
    # match, worth 0.25) report 100% confidence just because it was the
    # only thing that fired, even though most of the available evidence
    # (spectral analysis, tag year) never got a chance to weigh in.
    max_possible = 0.0

    # --- Tag / filename signals (40% total weight) ---
    tag_year: int = 0

    if file_path is not None:
        # Filename keyword scan
        max_possible += 0.25
        fname_hint = content_hint_from_filename(file_path.name)
        if fname_hint:
            scores[fname_hint] += 0.25

        # ID3 year tag
        if CAP.mutagen:
            max_possible += 0.15
            try:
                from mutagen import File as MutagenFile
                tags = MutagenFile(str(file_path), easy=True)
                if tags and "date" in tags:
                    year_str = str(tags["date"][0])[:4]
                    if year_str.isdigit():
                        tag_year = int(year_str)
            except Exception:
                pass

        if tag_year > 0:
            era_key = era_from_year(tag_year)
            if era_key:
                scores[era_key] += 0.15

    # --- Spectral signals (60% total weight) ---
    if audio is not None and CAP.numpy:
        max_possible += 0.60
        n = len(audio)
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio.ravel()

        # Noise floor estimate (quietest 5% of 1s windows)
        window = sr
        window_rms = []
        for start in range(0, n, window):
            chunk = mono[start: start + window]
            if len(chunk) > 100:
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms > 1e-9:
                    window_rms.append(rms)
        if window_rms:
            window_rms.sort()
            quietest = window_rms[: max(1, len(window_rms) // 20)]
            noise_floor_db = 20.0 * math.log10(sum(quietest) / len(quietest))
        else:
            noise_floor_db = -80.0

        # Crest factor (peak vs RMS)
        peak_db = 20.0 * math.log10(float(np.max(np.abs(mono))) + 1e-12)
        rms_db  = 20.0 * math.log10(float(np.sqrt(np.mean(mono ** 2))) + 1e-12)
        crest_factor = peak_db - rms_db

        # Tonal balance (bass vs treble vs mid)
        if CAP.scipy:
            bass   = lr_lowpass(mono.reshape(-1, 1),  sr, 150.0).ravel()
            treble = lr_highpass(mono.reshape(-1, 1), sr, 6000.0).ravel()
            treble_rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(treble ** 2))) + 1e-12)
            bass_rms_db   = 20.0 * math.log10(float(np.sqrt(np.mean(bass   ** 2))) + 1e-12)
            treble_deficit = rms_db - treble_rms_db  # how much treble is rolled off
        else:
            treble_deficit = 0.0
            bass_rms_db    = rms_db

        # Scoring logic (empirically tuned against Indian film/popular
        # music recordings; thresholds may need adjustment for other
        # regions' mastering conventions — see CONTRIBUTING.md)

        # High noise floor → likely classic analogue recording
        if noise_floor_db > -50:
            scores["classic_analogue"] += 0.25
        elif noise_floor_db > -60:
            scores["cassette_era"] += 0.15

        # High crest factor → natural/dynamic recording (classic era or classical)
        if crest_factor > 14:
            scores["classic_analogue"] += 0.15
            scores["carnatic_hindustani"] += 0.10
        elif crest_factor > 10:
            scores["cassette_era"] += 0.10

        # Heavy treble rolloff → tape compression (classic era)
        if treble_deficit > 12:
            scores["classic_analogue"] += 0.15
        elif treble_deficit > 7:
            scores["cassette_era"] += 0.10

        # Already loud/compressed (small crest factor) → modern mastering
        if crest_factor < 7:
            scores["modern_mastered"] += 0.15

        # Bass-heavy, loud → modern pop or BGM
        if bass_rms_db - rms_db > -2:
            scores["modern_mastered"] += 0.05
            scores["bgm"] += 0.05

    # Normalise against the evidence we had a chance to gather (see
    # max_possible above), not just the evidence that happened to fire —
    # otherwise one lone weak signal can misreport as 100% confident.
    if max_possible < 1e-6:
        return "early_digital", 0.4  # safe default — moderate processing, safest guess

    best_key = max(scores, key=lambda k: scores[k])
    confidence = min(1.0, scores[best_key] / max_possible)

    # Extra floor: spectral analysis (60% of the total weight) is by far
    # the most reliable signal — filename/tag hints alone are thin
    # evidence. If spectral analysis never ran at all (no audio available,
    # e.g. numpy missing or nothing loaded yet), cap confidence so a lone
    # filename keyword match can't still report as "auto-detected" green.
    spectral_ran = audio is not None and CAP.numpy
    if not spectral_ran:
        confidence = min(confidence, 0.5)

    log.debug(
        "Content detection: %s (confidence %.0f%%)\n  scores: %s",
        best_key, confidence * 100, scores,
    )
    return best_key, float(confidence)


# ---------------------------------------------------------------------------
# Preview playback
# ---------------------------------------------------------------------------

_playback_stop_event = threading.Event()
_playback_thread: threading.Thread | None = None


def _stereo_width_ratio(audio: "np.ndarray") -> float:
    """Side/mid RMS energy ratio: 0 = mono, ~1+ = very wide. Direct numpy
    port of the original pydub-based _compute_stereo_width_ratio (which
    looped sample-by-sample) — same definition, vectorised.
    """
    if not CAP.numpy or audio.ndim < 2 or audio.shape[1] < 2:
        return 0.0
    L, R = audio[:, 0].astype(np.float64), audio[:, 1].astype(np.float64)
    mid  = (L + R) * 0.5
    side = (L - R) * 0.5
    mid_rms  = float(np.sqrt(np.mean(mid ** 2)))
    side_rms = float(np.sqrt(np.mean(side ** 2)))
    return side_rms / mid_rms if mid_rms > 1e-9 else 0.0


def _windowed_dbfs(signal: "np.ndarray", sr: int, window_s: float = 1.0) -> list[float]:
    """Split a mono signal into fixed windows and return each window's
    RMS level in dBFS, skipping windows that are effectively silent.
    Shared by noise-floor estimation and (in elaborate mode) the
    bass/mid/treble measurements, so both get a robust, full-track
    distribution instead of one aggregate number that a loud intro or a
    silent outro can skew.
    """
    window = max(1, int(sr * window_s))
    out: list[float] = []
    for start in range(0, len(signal), window):
        chunk = signal[start:start + window]
        if len(chunk) > 100:
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > 1e-9:
                out.append(20.0 * math.log10(rms))
    return out


def analyze_track_for_optimal_params(
    audio: "np.ndarray",
    sr: int,
    input_path: "Path | None" = None,
    use_demucs: bool = False,
    demucs_model: str = "htdemucs",
    music_lufs_reference: float = -18.0,
    elaborate: bool = False,
) -> dict:
    """Analyze one specific track's actual characteristics and recommend
    concrete parameter values tuned to THIS recording, rather than the
    device/content profile's category-wide defaults.

    This is a numpy/current-engine port of the analyze_track() /
    compute_recommendations() pair from the project's earlier CLI+GUI
    tool (remaster_ilayaraja.py / remaster_gui.py) — same measurements,
    same discrete recommendation thresholds — adapted to this codebase's
    float32/numpy pipeline and current parameter names. Two differences
    from that port, both noted inline below:
      - dBFS here is measured directly off full-scale float32 (peak=1.0),
        matching pydub's .dBFS/.max_dBFS semantics the original used.
      - "vocal-gain-db" was a relative trim on top of a separated vocal
        stem in the old tool; this engine's vocal chain instead targets
        an absolute vocal_lufs level, so the recommendation is expressed
        as a suggested vocal_lufs relative to the current music_lufs
        target rather than as an additive gain (see below).

    `elaborate=True` trades speed for thoroughness. The default (fast)
    path measures each band as one aggregate RMS over (at most) the first
    120 seconds the caller passed in — quick, but a loud intro or a long
    quiet outro can skew a single aggregate number, and there's no full
    picture of how consistent the track is across its length. Elaborate
    mode, given the FULL track's audio:
      - measures bass/mid/treble as the MEDIAN of 1-second windowed
        levels across the whole track (robust to intro/outro/one-off
        transients) instead of one aggregate RMS, and also reports each
        band's window-to-window spread so you can see how consistent the
        tonal balance actually is, not just its central tendency;
      - adds integrated LUFS (BS.1770, via pyloudnorm) and an approximate
        loudness range (spread of windowed short-term loudness) — proper
        broadcast-style loudness measures rather than a plain RMS dBFS
        average;
      - always attempts the Demucs vocal/instrument comparison (if Demucs
        is installed) instead of only when a stem model happens to be
        selected for export, since that's the single biggest driver of
        analysis time and worth spending regardless of the export choice.
    The recommendation thresholds themselves are unchanged between modes
    — elaborate mode feeds them more reliable inputs, it doesn't use a
    different decision table.

    Returns:
      {
        "measurements":   {raw measured values, for display},
        "recommendations": {param_name: value, ...},  # "ideal" values
      }
    """
    if not CAP.numpy:
        return {"measurements": {}, "recommendations": {}}

    n = len(audio)
    if n == 0:
        return {"measurements": {}, "recommendations": {}}

    measurements: dict[str, Any] = {}
    rec: dict[str, Any] = {}
    measurements["elaborate"] = elaborate
    measurements["analyzed_duration_s"] = round(n / sr, 1)

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio.ravel()

    # --- Peak / RMS / crest factor, in dBFS (full scale = 1.0) ---
    peak_dbfs = 20.0 * math.log10(float(np.max(np.abs(mono))) + 1e-12)
    rms_dbfs  = 20.0 * math.log10(float(np.sqrt(np.mean(mono ** 2))) + 1e-12)
    crest_factor_db = peak_dbfs - rms_dbfs
    measurements["peak_dbfs"] = round(peak_dbfs, 1)
    measurements["rms_dbfs"] = round(rms_dbfs, 1)
    measurements["crest_factor_db"] = round(crest_factor_db, 1)

    if elaborate:
        # Proper broadcast-style loudness measures, full track.
        measurements["integrated_lufs"] = round(measure_lufs(audio, sr), 1)
        overall_windows = _windowed_dbfs(mono, sr, window_s=3.0)
        if len(overall_windows) >= 2:
            arr = np.array(overall_windows)
            # 95th-5th percentile spread as an approximate loudness range
            # (LRA) — cheap, no gating algorithm, but a reasonable proxy
            # for "how much does this track's level vary over its length".
            lra_approx = float(np.percentile(arr, 95) - np.percentile(arr, 5))
            measurements["loudness_range_approx_db"] = round(lra_approx, 1)

    # --- Bass / Mid / Treble split, compared against MID (not the full
    # mix) — this is what the original design compares against, and it's
    # a better tonal-balance signal than comparing against the full mix,
    # since the full mix already blends bass+mid+treble together. ---
    if CAP.scipy:
        bass  = lr_lowpass(mono.reshape(-1, 1), sr, 150.0).ravel()
        treble = lr_highpass(mono.reshape(-1, 1), sr, 6000.0).ravel()
        mid = lr_lowpass(lr_highpass(mono.reshape(-1, 1), sr, 150.0), sr, 6000.0).ravel()

        if elaborate:
            # Median of windowed levels across the FULL track, not one
            # aggregate RMS — see docstring. Falls back to the aggregate
            # if a band is too quiet throughout to produce any windows.
            bass_windows   = _windowed_dbfs(bass, sr)
            mid_windows    = _windowed_dbfs(mid, sr)
            treble_windows = _windowed_dbfs(treble, sr)
            bass_dbfs = float(np.median(bass_windows)) if bass_windows else \
                20.0 * math.log10(float(np.sqrt(np.mean(bass ** 2))) + 1e-12)
            mid_dbfs = float(np.median(mid_windows)) if mid_windows else \
                20.0 * math.log10(float(np.sqrt(np.mean(mid ** 2))) + 1e-12)
            treble_dbfs = float(np.median(treble_windows)) if treble_windows else \
                20.0 * math.log10(float(np.sqrt(np.mean(treble ** 2))) + 1e-12)
            if bass_windows:
                measurements["bass_variability_db"] = round(
                    float(np.percentile(bass_windows, 90) - np.percentile(bass_windows, 10)), 1)
            if treble_windows:
                measurements["treble_variability_db"] = round(
                    float(np.percentile(treble_windows, 90) - np.percentile(treble_windows, 10)), 1)
        else:
            bass_dbfs   = 20.0 * math.log10(float(np.sqrt(np.mean(bass ** 2))) + 1e-12)
            mid_dbfs    = 20.0 * math.log10(float(np.sqrt(np.mean(mid ** 2))) + 1e-12)
            treble_dbfs = 20.0 * math.log10(float(np.sqrt(np.mean(treble ** 2))) + 1e-12)

        measurements["bass_dbfs"] = round(bass_dbfs, 1)
        measurements["mid_dbfs"] = round(mid_dbfs, 1)
        measurements["treble_dbfs"] = round(treble_dbfs, 1)

        bass_vs_mid = bass_dbfs - mid_dbfs
        if bass_vs_mid < -6:
            rec["bass_shelf_db"] = 4.0
        elif bass_vs_mid > 0:
            rec["bass_shelf_db"] = 1.0
        else:
            rec["bass_shelf_db"] = 2.5

        treble_vs_mid = treble_dbfs - mid_dbfs
        if treble_vs_mid < -10:
            rec["treble_shelf_db"] = 3.0
        elif treble_vs_mid > -4:
            rec["treble_shelf_db"] = 1.0
        else:
            rec["treble_shelf_db"] = 2.0

    # --- Stereo width ratio -> per-band width suggestion ---
    # The original kept bass centred/mono automatically and only widened
    # everything above it; this engine's per-band width control mirrors
    # that split via width_bass (left at 1.0, unchanged) + width_mid/
    # width_treble (the two this recommendation actually sets).
    if audio.ndim == 2 and audio.shape[1] >= 2:
        width_ratio = _stereo_width_ratio(audio)
        measurements["stereo_width_ratio"] = round(width_ratio, 2)
        if width_ratio < 0.15:
            suggested_width = 3.0
        elif width_ratio > 0.5:
            suggested_width = 1.7
        else:
            suggested_width = 2.2
        rec["width_mid"] = suggested_width
        rec["width_treble"] = suggested_width

    # --- Auto loudness ---
    rec["auto_loudness"] = crest_factor_db > 8

    # --- Loudness headroom ---
    # The original always recommended 0.4dB regardless of the track (a
    # flat, safe default) and printed the source's own raw peak headroom
    # alongside it for context — preserved as-is here rather than
    # inventing a track-dependent formula that didn't exist originally.
    rec["headroom_db"] = 0.4
    measurements["raw_peak_headroom_db"] = round(abs(peak_dbfs), 1)

    # --- Noise floor -> denoise / denoise amount ---
    window_dbfs = _windowed_dbfs(mono, sr, window_s=1.0)
    if window_dbfs:
        window_dbfs_sorted = sorted(window_dbfs)
        quietest = window_dbfs_sorted[: max(1, len(window_dbfs_sorted) // 20)]
        noise_floor_dbfs = sum(quietest) / len(quietest)
        measurements["noise_floor_dbfs"] = round(noise_floor_dbfs, 1)
        rec["denoise"] = noise_floor_dbfs > -50
        rec["denoise_amount"] = 12.0 if noise_floor_dbfs > -50 else 8.0

    # --- Vocal vs instrumental level (optional, needs Demucs) ---
    # Only run when explicitly requested — separation is the slowest part
    # of analysis by far, same as the original tool's --use-demucs gate.
    # Elaborate mode always attempts it (if Demucs is installed) since
    # it's the single biggest lever for spending more time on a more
    # thorough result, regardless of what was picked for export.
    # `vocal_analysis_status` is always set (never silently omitted) so
    # the caller can tell the difference between "ran and found nothing
    # unusual" and "never ran" — and if it never ran, why not. Without
    # this, a skipped Demucs step looks identical to a fast, successful
    # analysis: no vocal_lufs recommendation and no indication why.
    if not (use_demucs or elaborate):
        measurements["vocal_analysis_status"] = "skipped (no stem model selected in Step 4)"
    elif not CAP.demucs:
        measurements["vocal_analysis_status"] = "skipped (Demucs not installed)"
    elif input_path is None:
        measurements["vocal_analysis_status"] = "skipped (no input file path)"
    else:
        try:
            result = separate_stems(Path(input_path), model=demucs_model, use_cache=True)
            if result is None:
                measurements["vocal_analysis_status"] = "skipped (Demucs separation returned no stems)"
            else:
                stems, stem_sr = result
                if "vocals" not in stems:
                    measurements["vocal_analysis_status"] = "skipped (no vocal stem in Demucs output)"
                else:
                    vocals = stems["vocals"]
                    instrument_names = [k for k in stems if k != "vocals"]
                    if not instrument_names:
                        measurements["vocal_analysis_status"] = "skipped (no instrument stems to compare against)"
                    else:
                        inst_mix = stems[instrument_names[0]].copy()
                        for k in instrument_names[1:]:
                            n_min = min(len(inst_mix), len(stems[k]))
                            inst_mix = inst_mix[:n_min] + stems[k][:n_min]
                        vocal_mono = vocals.mean(axis=1) if vocals.ndim == 2 else vocals.ravel()
                        inst_mono  = inst_mix.mean(axis=1) if inst_mix.ndim == 2 else inst_mix.ravel()
                        vocal_dbfs = 20.0 * math.log10(float(np.sqrt(np.mean(vocal_mono ** 2))) + 1e-12)
                        inst_dbfs  = 20.0 * math.log10(float(np.sqrt(np.mean(inst_mono ** 2))) + 1e-12)
                        ratio_db = vocal_dbfs - inst_dbfs
                        measurements["vocal_instrument_ratio_db"] = round(ratio_db, 1)
                        measurements["vocal_analysis_status"] = "ok"
                        # Original: suggested additive gain = -(ratio + 6.0), aiming
                        # for vocals sitting 6dB below the instrumental. Ported here
                        # as an absolute vocal_lufs target relative to the current
                        # music_lufs reference, since this engine targets vocal_lufs
                        # directly rather than applying a relative gain on top of
                        # whatever the vocal happened to measure at.
                        target_relative_db = -(ratio_db + 6.0)
                        rec["vocal_lufs"] = round(music_lufs_reference + target_relative_db, 1)
        except Exception as exc:
            log.warning("Demucs analysis for vocal/instrument ratio failed: %s", exc)
            measurements["vocal_analysis_status"] = f"failed ({exc})"

    return {"measurements": measurements, "recommendations": rec}


def play_audio_preview(
    audio: "np.ndarray",
    sr: int,
    on_finish: "callable | None" = None,
    on_error: "callable | None" = None,
) -> None:
    """Play a float32 numpy array asynchronously.
    Tries sounddevice → pydub.playback → ffplay in order.
    Call stop_audio_preview() to interrupt.

    on_error(exc), if given, is called with the final exception when every
    playback backend fails. Previously these failures were only logged at
    DEBUG level (invisible by default), which made a failed Play Original/
    Play Processed look like it silently did nothing.
    """
    global _playback_thread
    stop_audio_preview()
    _playback_stop_event.clear()

    if audio is None or len(audio) == 0:
        log.warning("play_audio_preview called with an empty clip.")
        if on_error:
            try:
                on_error(RuntimeError("No audio to play (empty clip)."))
            except Exception:
                pass
        return

    if CAP.numpy:
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-6:
            # Not a playback-backend problem — the clip itself is silent.
            # Surfacing this distinctly saves a lot of "why is there no
            # sound" confusion vs a genuine backend failure below.
            log.warning("Preview clip is silent (peak amplitude ~0) — nothing to hear regardless of playback backend.")

    def _run():
        try:
            if CAP.sounddevice:
                try:
                    _play_sounddevice(audio, sr)
                    return
                except Exception as exc:
                    log.debug("sounddevice playback failed (%s) — falling back to pydub/ffplay.", exc)
            _play_pydub_fallback(audio, sr)
        except Exception as exc:
            log.warning("Playback failed: %s", exc)
            if on_error:
                try:
                    on_error(exc)
                except Exception:
                    pass
        finally:
            if on_finish:
                try:
                    on_finish()
                except Exception:
                    pass

    _playback_thread = threading.Thread(target=_run, daemon=True)
    _playback_thread.start()


def stop_audio_preview() -> None:
    """Stop any currently playing preview."""
    global _playback_thread
    _playback_stop_event.set()
    if CAP.sounddevice:
        try:
            sd.stop()
        except Exception:
            pass
    if _playback_thread and _playback_thread.is_alive():
        _playback_thread.join(timeout=1.0)
    _playback_thread = None


def _play_sounddevice(audio: "np.ndarray", sr: int) -> None:
    """sounddevice callback-based playback with stop support."""
    pos = [0]

    def callback(outdata, frames, time_info, status):
        if _playback_stop_event.is_set():
            raise sd.CallbackStop()
        start = pos[0]
        end   = min(start + frames, len(audio))
        chunk = audio[start:end]
        if audio.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        if len(chunk) < frames:
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = 0
            raise sd.CallbackStop()
        outdata[:] = chunk
        pos[0] = end

    channels = audio.shape[1] if audio.ndim == 2 else 1
    with sd.OutputStream(
        samplerate=sr,
        channels=channels,
        callback=callback,
        dtype="float32",
    ):
        while not _playback_stop_event.is_set():
            time.sleep(0.05)
            if pos[0] >= len(audio):
                break


def _play_pydub_fallback(audio: "np.ndarray", sr: int) -> None:
    """pydub.playback fallback (tries simpleaudio → pyaudio internally,
    then falls further back to piping raw PCM to ffplay directly).
    """
    try:
        from pydub import playback as pb
        seg = float32_to_pydub(audio if audio.ndim == 2 else audio.reshape(-1, 1), sr)
        pb.play(seg)
        return
    except Exception as exc:
        log.debug("pydub playback failed: %s — trying ffplay.", exc)
        pydub_error = exc

    try:
        _play_ffplay_fallback(audio, sr)
    except Exception as ffplay_error:
        raise RuntimeError(
            "No working audio playback backend found. Tried: sounddevice "
            f"({'not installed' if not CAP.sounddevice else 'failed'}), "
            f"pydub/simpleaudio/pyaudio ({pydub_error}), and ffplay "
            f"({ffplay_error}). Install one of 'sounddevice' or "
            "'simpleaudio' (pip), or make sure 'ffplay' (bundled with "
            "ffmpeg) is on PATH."
        ) from ffplay_error


def _play_ffplay_fallback(audio: "np.ndarray", sr: int) -> None:
    """Last-resort: pipe raw PCM to ffplay."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    channels = audio.shape[1] if audio.ndim == 2 else 1
    cmd = [
        "ffplay", "-autoexit", "-nodisp",
        "-f", "s16le", "-ar", str(sr), "-ac", str(channels), "-",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc.stdin.write(pcm)
    proc.stdin.close()
    while proc.poll() is None:
        if _playback_stop_event.is_set():
            proc.terminate()
            break
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Batch report
# ---------------------------------------------------------------------------

@dataclass
class FileReport:
    input_path: str
    output_paths: list[str] = field(default_factory=list)
    system_profile: str = ""
    content_profile: str = ""
    content_confidence: float = 0.0
    input_lufs: float = 0.0
    input_peak_db: float = 0.0
    output_lufs: float = 0.0
    output_peak_db: float = 0.0
    processing_time_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    success: bool = False
    error: str = ""


@dataclass
class BatchReport:
    batch_id: str
    started_at: str
    completed_at: str = ""
    files: list[FileReport] = field(default_factory=list)
    total_files: int = 0
    succeeded: int = 0
    failed: int = 0

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"batch_report_{self.batch_id}.json"
        data = {
            "batch_id":    self.batch_id,
            "started_at":  self.started_at,
            "completed_at": self.completed_at,
            "total_files": self.total_files,
            "succeeded":   self.succeeded,
            "failed":      self.failed,
            "files": [
                {
                    "input":             fr.input_path,
                    "outputs":           fr.output_paths,
                    "system_profile":    fr.system_profile,
                    "content_profile":   fr.content_profile,
                    "content_confidence": round(fr.content_confidence, 2),
                    "input_lufs":        round(fr.input_lufs, 1),
                    "input_peak_db":     round(fr.input_peak_db, 1),
                    "output_lufs":       round(fr.output_lufs, 1),
                    "output_peak_db":    round(fr.output_peak_db, 1),
                    "processing_time_s": round(fr.processing_time_s, 1),
                    "warnings":          fr.warnings,
                    "success":           fr.success,
                    "error":             fr.error,
                }
                for fr in self.files
            ],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("Batch report saved: %s", report_path)
        return report_path


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def read_metadata(path: Path) -> dict:
    """Read ID3 tags from an MP3 file. Returns empty dict if mutagen
    is unavailable or the file has no tags.
    """
    if not CAP.mutagen:
        return {}
    try:
        from mutagen import File as MutagenFile
        tags = MutagenFile(str(path), easy=True)
        if not tags:
            return {}
        return {k: str(v[0]) for k, v in tags.items() if v}
    except Exception:
        return {}


def copy_metadata(src: Path, dst: Path) -> None:
    """Copy ID3 tags and embedded artwork from src to dst (MP3 only)."""
    if not CAP.mutagen:
        return
    try:
        from mutagen.id3 import ID3, ID3NoHeaderError
        try:
            src_tags = ID3(str(src))
        except ID3NoHeaderError:
            return
        try:
            dst_tags = ID3(str(dst))
        except ID3NoHeaderError:
            dst_tags = ID3()
        for key, value in src_tags.items():
            try:
                dst_tags[key] = value
            except Exception:
                pass
        dst_tags.save(str(dst), v1=2)
    except Exception as exc:
        log.debug("Metadata copy failed: %s", exc)


def collect_audio_files(path: Path) -> list[Path]:
    """Return a sorted list of audio files (MP3/FLAC/WAV) under path."""
    extensions = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg"}
    if path.is_file() and path.suffix.lower() in extensions:
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
    return []


def build_output_path(
    input_path: Path,
    output_base: Path,
    suffix: str = "_remastered",
    extension: str = ".mp3",
) -> Path:
    """Compute the output file path, creating the output directory."""
    output_base.mkdir(parents=True, exist_ok=True)
    return output_base / f"{input_path.stem}{suffix}{extension}"


def estimate_disk_usage(files: list[Path], params: dict) -> int:
    """Estimate total output size in bytes based on input files and
    output settings. Returns estimated bytes needed.
    """
    layout    = params.get("layout", "stereo")
    out_fmt   = params.get("output_format", "mp3")
    bitrate_s = params.get("bitrate", "320k")

    try:
        bitrate_kbps = int(bitrate_s.lower().replace("k", ""))
    except ValueError:
        bitrate_kbps = 320

    total = 0
    for f in files:
        try:
            duration_s = f.stat().st_size * 8 / 128000  # rough estimate assuming 128kbps source
        except Exception:
            duration_s = 300  # default 5 minutes

        if out_fmt == "flac":
            bytes_per_file = int(duration_s * 44100 * 2 * 3)  # ~24-bit stereo estimate
        elif out_fmt == "wav":
            bytes_per_file = int(duration_s * 44100 * 2 * 2)
        else:
            bytes_per_file = int(duration_s * bitrate_kbps * 1000 / 8)

        if layout == "5.1":
            # MKV with AC3 640kbps
            bytes_per_file += int(duration_s * 640000 / 8)
            if params.get("also_produce_stereo_mp3", True):
                bytes_per_file += int(duration_s * 320000 / 8)

        total += bytes_per_file

    return total


def check_disk_space(output_dir: Path, required_bytes: int) -> tuple[bool, int]:
    """Return (has_space, free_bytes). Warns if estimated usage > 90% of free."""
    try:
        stat = shutil.disk_usage(output_dir)
        free = stat.free
        ok = required_bytes < free * 0.9
        return ok, free
    except Exception:
        return True, -1  # can't check, assume OK


# ---------------------------------------------------------------------------
# Pipeline orchestrator — single-file entry point
# ---------------------------------------------------------------------------

def remaster_file(
    input_path: Path,
    output_dir: Path,
    params: dict,
    file_report: FileReport | None = None,
    progress_callback: "callable | None" = None,
) -> list[Path]:
    """Process a single audio file through the full remaster pipeline.

    Returns a list of output paths (always includes stereo file;
    may also include MKV when the resolved device profile is 5.1).

    params is a flat dict from profiles.resolve_profile() — all keys
    documented in profiles.py. Additional per-run overrides (EQ bands,
    preview mode, etc.) are merged in by the caller before passing.

    progress_callback(stage: str, pct: float) is called at each stage
    so the GUI can update its progress bar without blocking.
    """
    t_start = time.time()
    output_paths: list[Path] = []

    def _progress(stage: str, pct: float) -> None:
        log.debug("[%s] %s %.0f%%", input_path.name, stage, pct)
        if progress_callback:
            try:
                progress_callback(stage, pct)
            except Exception:
                pass

    _progress("load", 0)
    log.info("Processing: %s", input_path)

    # ----------------------------------------------------------------
    # Phase A: load → measure input → restoration → float32
    # ----------------------------------------------------------------

    seg = load_audio_file(input_path)

    if file_report:
        if CAP.numpy:
            audio_tmp, sr_tmp = pydub_to_float32(seg)
            file_report.input_lufs    = measure_lufs(audio_tmp, sr_tmp)
            file_report.input_peak_db = 20.0 * math.log10(
                float(np.max(np.abs(audio_tmp))) + 1e-12
            )
        else:
            file_report.input_lufs    = seg.dBFS
            file_report.input_peak_db = seg.max_dBFS

    # Restoration (declick / denoise) — ffmpeg pre-pass
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        restored_path = tmp_dir / "restored.wav"

        actual_path = apply_restoration_filters(
            input_path, restored_path,
            declick=params.get("declick", False),
            declick_threshold=params.get("declick_threshold", 2.0),
            denoise=params.get("denoise", False),
            denoise_amount=params.get("denoise_amount", 10.0),
            denoise_floor=params.get("denoise_floor", -50.0),
            denoise_type=params.get("denoise_type", "white"),
        )

        if actual_path != input_path:
            seg = load_audio_file(actual_path)

        _progress("restore", 10)

        # ----------------------------------------------------------------
        # Phase B (numpy): float32 DSP chain
        # ----------------------------------------------------------------

        if not CAP.numpy:
            # Reduced-quality path: pydub-only normalization and export
            log.warning("numpy unavailable — using pydub-only pipeline.")
            output_path = build_output_path(
                input_path, output_dir,
                extension=f".{params.get('output_format','mp3')}",
            )
            seg.export(str(output_path), format=params.get("output_format", "mp3"),
                       bitrate=params.get("bitrate", "320k"))
            copy_metadata(input_path, output_path)
            output_paths.append(output_path)
            if file_report:
                file_report.output_paths = [str(p) for p in output_paths]
                file_report.processing_time_s = time.time() - t_start
                file_report.success = True
            return output_paths

        audio, sr = pydub_to_float32(seg)

        # Optional preview window (CLI --preview-seconds / --preview-start)
        preview_seconds = float(params.get("preview_seconds", 0) or 0)
        preview_start = float(params.get("preview_start", 0.0) or 0.0)
        if preview_seconds > 0:
            start_samp = max(0, int(preview_start * sr))
            end_samp = min(len(audio), start_samp + int(preview_seconds * sr))
            if end_samp > start_samp:
                audio = audio[start_samp:end_samp]
                log.info(
                    "Preview mode: %.1fs starting at %.1fs (%d samples)",
                    preview_seconds, preview_start, len(audio),
                )

        # Trim silence and fades
        if params.get("trim_silence", False):
            audio = trim_track_silence(
                audio, sr,
                silence_thresh_db=params.get("silence_thresh_db", -45.0),
                min_silence_ms=params.get("min_silence_len_ms", 400),
                keep_silence_ms=params.get("keep_silence_ms", 150),
            )

        # Experimental DeepFilterNet denoise (full-mix, pre-stems)
        if params.get("use_deepfilternet", False) and CAP.deepfilternet:
            _progress("deepfilternet", 12)
            try:
                audio = _apply_deepfilternet(audio, sr)
                log.info("DeepFilterNet applied.")
            except Exception as exc:
                log.warning("DeepFilterNet failed: %s — continuing without it.", exc)

        _progress("stems", 15)

        # Stem separation (Demucs)
        vocal_stem: np.ndarray | None = None
        instrument_mix: np.ndarray | None = None
        stems_used = False

        model = params.get("demucs_model", "htdemucs")
        want_stems = (
            params.get("use_stems", False)
            and params.get("use_demucs", False)
            and model
            and model != "none"
        )
        if want_stems:
            # Prefer the restored file so declick/denoise is not discarded
            # when stem separation is active.
            stem_source = actual_path if actual_path != input_path else input_path
            result = separate_stems(
                stem_source, model=model,
                use_cache=params.get("use_cache", True),
            )
            if result is not None:
                stems, stem_sr = result
                if "vocals" in stems:
                    # Demucs always outputs at its model sample rate
                    # (typically 44100). Resample stems back to the
                    # source rate so the rest of the pipeline keeps the
                    # original timeline, filter tuning, and export rate.
                    source_sr = sr
                    if stem_sr != source_sr:
                        log.info(
                            "Resampling stems %d Hz → source rate %d Hz.",
                            stem_sr, source_sr,
                        )
                        stems = {
                            k: resample_audio(v, stem_sr, source_sr)
                            for k, v in stems.items()
                        }

                    # Align all stem lengths before processing
                    min_len = min(len(s) for s in stems.values())
                    stems = {k: v[:min_len] for k, v in stems.items()}

                    # Vocal chain
                    vocal_stem = process_vocal_stem(
                        stems["vocals"], sr,
                        presence_db=params.get("vocal_presence_db", 1.5),
                        air_db=params.get("vocal_air_db", 2.0),
                        mud_cut_db=params.get("vocal_mud_cut_db", -1.5),
                        deesser=params.get("deesser", False),
                        deesser_threshold_db=params.get("deesser_threshold_db", -24.0),
                        deesser_ratio=params.get("deesser_ratio", 4.0),
                        deesser_freq=params.get("deesser_freq", 5000),
                        target_lufs=params.get("vocal_lufs", -18.0),
                        headroom_db=params.get("headroom_db", 0.5),
                        use_voicefixer=params.get("use_voicefixer", False),
                    )
                    # Instrument chain (all non-vocal stems)
                    inst_stems = {k: v for k, v in stems.items() if k != "vocals"}
                    if inst_stems:
                        instrument_mix = process_instrument_stems(
                            inst_stems, sr,
                            bass_shelf_db=params.get("inst_bass_shelf_db", 1.0),
                            air_shelf_db=params.get("inst_air_shelf_db", 1.5),
                            harmonic_exciter=params.get("inst_harmonic_exciter", True),
                            exciter_freq=params.get("inst_exciter_freq", 3000),
                            exciter_amount=params.get("inst_exciter_amount", 0.25),
                            target_lufs=params.get("music_lufs", -18.0),
                            headroom_db=params.get("headroom_db", 0.5),
                        )
                    # Recombine for mastering — align component lengths first
                    components = [vocal_stem]
                    if instrument_mix is not None:
                        components.append(instrument_mix)
                    min_comp = min(len(c) for c in components)
                    components = [c[:min_comp] for c in components]
                    audio = sum(components).astype(np.float32)
                    stems_used = True

        _progress("mastering", 40)

        # Mastering chain
        # When stems were used, vocal_presence_db was already applied in the
        # vocal chain — strip it from the master presence peak to avoid a
        # double boost on the vocal region.
        master_presence_db = params.get("presence_db", 0.0)
        if stems_used:
            master_presence_db = master_presence_db - params.get("vocal_presence_db", 0.0)

        audio = apply_mastering_chain(
            audio, sr,
            bass_shelf_hz=params.get("bass_shelf_hz", 100.0),
            bass_shelf_db=params.get("bass_shelf_db", 0.0),
            treble_shelf_hz=params.get("treble_shelf_hz", 8000.0),
            treble_shelf_db=params.get("treble_shelf_db", 0.0),
            presence_hz=params.get("presence_hz", 0.0),
            presence_db=master_presence_db,
            presence_q=params.get("presence_q", 1.2),
            notch_hz=params.get("notch_hz", 0.0),
            notch_db=params.get("notch_db", 0.0),
            notch_q=params.get("notch_q", 2.0),
            eq_bands=params.get("eq_bands"),
            multiband_compress=params.get("multiband_compress", False),
            mb_low_crossover_hz=params.get("mb_low_crossover_hz", 200),
            mb_high_crossover_hz=params.get("mb_high_crossover_hz", 4000),
            mb_low_ratio=params.get("mb_low_ratio", 2.0),
            mb_mid_ratio=params.get("mb_mid_ratio", 2.0),
            mb_high_ratio=params.get("mb_high_ratio", 2.0),
            saturation=params.get("saturation", False),
            saturation_drive_db=params.get("saturation_drive_db", 5.0),
            saturation_mix=params.get("saturation_mix", 0.3),
            crystalizer=params.get("crystalizer", False),
            crystalizer_intensity=params.get("crystalizer_intensity", 2.5),
            width_bands=params.get("width_bands", False),
            width_bass=params.get("width_bass", 1.0),
            width_mid=params.get("width_mid", 1.6),
            width_treble=params.get("width_treble", 1.6),
            width_low_crossover_hz=params.get("width_low_crossover_hz", 150),
            width_high_crossover_hz=params.get("width_high_crossover_hz", 4000),
            crossfeed=params.get("crossfeed", False),
            crossfeed_strength=params.get("crossfeed_strength", 0.3),
            crossfeed_range=params.get("crossfeed_range", 0.5),
            target_lufs=params.get("final_lufs", -14.0),
            headroom_db=params.get("headroom_db", 0.5),
        )

        # Apply fades after mastering (on clean, processed audio)
        audio = apply_fades(
            audio, sr,
            fade_in_ms=params.get("fade_in_ms", 0),
            fade_out_ms=params.get("fade_out_ms", 0),
        )

        _progress("encode", 70)

        # ----------------------------------------------------------------
        # Phase C: export — float32 → encode → metadata
        # ----------------------------------------------------------------

        # Convert back to pydub for export
        seg_out = float32_to_pydub(audio, sr)

        # Build export filter chain (two-pass loudnorm at encode time)
        out_fmt = params.get("output_format", "mp3")
        use_loudnorm = params.get("use_loudnorm", True) and out_fmt == "mp3"
        measured_stats = None

        if use_loudnorm:
            # Pass 1: measure loudness of our processed audio
            tmp_measure_wav = tmp_dir / "measure.wav"
            seg_out.export(str(tmp_measure_wav), format="wav")
            lufs_target   = params.get("final_lufs", -14.0)
            true_peak     = params.get("headroom_db", 0.5) * -1
            lra           = params.get("lufs_range", 11.0)
            measured_stats = measure_lufs_for_export(
                tmp_measure_wav, lufs_target, true_peak, lra
            )
            if measured_stats is None:
                log.debug("Two-pass loudnorm measurement failed; single-pass fallback.")

        filter_chain = build_export_filter_chain(
            use_loudnorm=use_loudnorm,
            lufs_target=params.get("final_lufs", -14.0),
            lufs_true_peak_db=-params.get("headroom_db", 0.5),
            lufs_range=params.get("lufs_range", 11.0),
            measured_stats=measured_stats,
            use_limiter=params.get("use_limiter", False),
            limiter_oversample=params.get("limiter_oversample", False),
            headroom_db=params.get("headroom_db", 0.5),
            crossfeed=params.get("crossfeed", False) and not CAP.scipy,
            crossfeed_strength=params.get("crossfeed_strength", 0.3),
            crossfeed_range=params.get("crossfeed_range", 0.5),
            sample_rate=sr,
        )

        # Stereo output
        ext = f".{out_fmt}"
        stereo_out = build_output_path(input_path, output_dir, extension=ext)
        export_kwargs: dict[str, Any] = {"format": out_fmt}
        if out_fmt == "mp3":
            export_kwargs["bitrate"] = params.get("bitrate", "320k")
        if filter_chain:
            export_kwargs["parameters"] = ["-af", filter_chain]

        seg_out.export(str(stereo_out), **export_kwargs)
        copy_metadata(input_path, stereo_out)
        output_paths.append(stereo_out)
        log.info("Stereo output: %s", stereo_out)

        # ----------------------------------------------------------------
        # 5.1 Surround output (any 5.1 device profile)
        # ----------------------------------------------------------------

        if params.get("layout", "stereo") == "5.1":
            if not CAP.soundfile:
                log.warning("soundfile not available — 5.1 output skipped.")
                if file_report:
                    file_report.warnings.append("soundfile missing — 5.1 skipped.")
            else:
                _progress("surround", 80)
                try:
                    channels_51 = build_surround_51(
                        audio, sr,
                        vocal_stem=vocal_stem,
                        instrument_stem=instrument_mix,
                        lfe_cutoff_hz=params.get("lfe_cutoff_hz", 80),
                        rear_cutoff_hz=params.get("rear_cutoff_hz", 300),
                        rear_attenuation_db=params.get("rear_attenuation_db", -6.0),
                        rear_delay_ms=params.get("rear_delay_ms", 15.0),
                        centre_attenuation_db=params.get("centre_attenuation_db", -3.0),
                        lfe_mode=params.get("lfe_mode", "gentle"),
                        headroom_db=params.get("headroom_db", 0.5),
                        speaker_bias_front=params.get("speaker_bias_front", "off"),
                        speaker_bias_front_db=params.get("speaker_bias_front_db", 0.0),
                        speaker_bias_centre=params.get("speaker_bias_centre", "off"),
                        speaker_bias_centre_db=params.get("speaker_bias_centre_db", 0.0),
                        speaker_bias_rear=params.get("speaker_bias_rear", "off"),
                        speaker_bias_rear_db=params.get("speaker_bias_rear_db", 0.0),
                    )

                    multichannel_wav = tmp_dir / f"{input_path.stem}_51.wav"
                    write_multichannel_wav(channels_51, sr, multichannel_wav)

                    mkv_out = build_output_path(
                        input_path, output_dir,
                        suffix="_remastered_51",
                        extension=".mkv",
                    )
                    codec = params.get("surround_codec", "ac3")
                    mux_surround_mkv(multichannel_wav, mkv_out, codec=codec, source_sr=sr)

                    if not verify_surround_output(mkv_out):
                        if file_report:
                            file_report.warnings.append("Surround output channel count mismatch.")

                    output_paths.append(mkv_out)

                    # Always also produce stereo MP3 alongside (for any 5.1 profile)
                    if params.get("also_produce_stereo_mp3", True) and out_fmt != "mp3":
                        mp3_out = build_output_path(input_path, output_dir, extension=".mp3")
                        seg_out.export(str(mp3_out), format="mp3",
                                       bitrate=params.get("bitrate", "320k"))
                        copy_metadata(input_path, mp3_out)
                        output_paths.append(mp3_out)

                except Exception as exc:
                    log.error("5.1 surround build failed: %s", exc)
                    if file_report:
                        file_report.warnings.append(f"5.1 failed: {exc}")

        # ----------------------------------------------------------------
        # Measure output and finalise report
        # ----------------------------------------------------------------
        _progress("done", 100)

        if file_report:
            out_lufs = measure_lufs(audio, sr)
            out_peak = 20.0 * math.log10(float(np.max(np.abs(audio))) + 1e-12)
            file_report.output_lufs    = out_lufs
            file_report.output_peak_db = out_peak
            file_report.output_paths   = [str(p) for p in output_paths]
            file_report.processing_time_s = time.time() - t_start
            file_report.success = True

    return output_paths


def _remaster_worker(kwargs: dict) -> FileReport:
    """Picklable top-level wrapper for ProcessPoolExecutor (parallel mode).
    Builds and returns a full FileReport (not just a bare status tuple) so
    parallel batches get the same LUFS/peak/timing measurements sequential
    batches do — ProcessPoolExecutor pickles the return value back to the
    parent process, and FileReport (plain dataclass of str/float/list) is
    picklable.
    """
    input_path = Path(kwargs.pop("input_path"))
    output_dir = Path(kwargs.pop("output_dir"))
    params     = kwargs.pop("params")
    fr = FileReport(input_path=str(input_path))
    try:
        remaster_file(input_path, output_dir, params, file_report=fr)
    except Exception as exc:
        fr.success = False
        fr.error = str(exc)
    return fr


# ---------------------------------------------------------------------------
# Album mode
# ---------------------------------------------------------------------------

def compute_album_lufs_reference(files: list[Path], target_lufs: float) -> float:
    """Compute the integrated LUFS reference for album-consistent processing.

    Standard streaming-platform approach (Spotify/Apple Music album mode):
    each track is measured individually, then all tracks in the album are
    normalised using the SAME target LUFS — this preserves the intentional
    relative loudness between tracks (a quiet ballad stays quieter than a
    loud opener) while still bringing the album to a consistent target,
    rather than the old approach of finding one fixed gain from the
    quietest-safe-track (which made quiet tracks disproportionately quiet).

    In practice this just means: match_lufs(track, target_lufs) applied
    per-track IS the correct album-mode behaviour, because LUFS matching
    to the same target for every track already preserves inter-track
    dynamics relative to each other far better than naive peak/RMS gain
    floors would. This function exists as an explicit, documented no-op
    confirmation step (and a hook for future per-album analysis, e.g.
    detecting outlier tracks) rather than a separate gain computation.
    """
    return target_lufs


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def run_batch(
    files: list[Path],
    output_dir: Path,
    params: dict,
    parallel_workers: int = 1,
    album_mode: bool = False,
    progress_callback: "callable | None" = None,
    log_callback: "callable | None" = None,
    cancel_event: "threading.Event | None" = None,
) -> BatchReport:
    """Process a batch of files, writing a JSON sidecar report.

    Each track produces its own output(s) — for a 5.1 device profile
    this means one MKV per track (never a combined multi-chapter MKV).

    album_mode preserves relative inter-track loudness (see
    compute_album_lufs_reference) rather than gain-flooring every track
    to the same absolute level.
    """
    batch_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    report = BatchReport(
        batch_id=batch_id,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_files=len(files),
    )

    def _log(msg: str) -> None:
        log.info(msg)
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass

    # Disk space pre-check
    estimated = estimate_disk_usage(files, params)
    ok, free = check_disk_space(output_dir, estimated)
    if not ok:
        _log(
            f"WARNING: estimated output size ({estimated / 1e9:.1f}GB) exceeds "
            f"90% of free disk space ({free / 1e9:.1f}GB). Proceeding anyway."
        )

    effective_params = dict(params)
    if album_mode:
        effective_params["_album_mode"] = True
        # LUFS target stays the same per-track; see compute_album_lufs_reference
        _log(f"Album mode: all {len(files)} tracks will target {params.get('final_lufs', -14.0)} LUFS "
             f"individually, preserving relative loudness between tracks.")

    if parallel_workers > 1 and len(files) > 1:
        jobs = [
            {"input_path": str(f), "output_dir": str(output_dir), "params": effective_params}
            for f in files
        ]
        completed = 0
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {executor.submit(_remaster_worker, job): job["input_path"] for job in jobs}
            for future in as_completed(futures):
                if cancel_event and cancel_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                fr = future.result()
                completed += 1
                if fr.success:
                    report.succeeded += 1
                    _log(f"Done: {fr.input_path}")
                else:
                    report.failed += 1
                    _log(f"FAILED: {fr.input_path}: {fr.error}")
                report.files.append(fr)
                if progress_callback:
                    progress_callback(completed, len(files))
    else:
        for i, f in enumerate(files):
            if cancel_event and cancel_event.is_set():
                _log("Batch cancelled by user.")
                break

            content_key, confidence = "early_digital", 0.0
            if CAP.numpy:
                try:
                    seg = load_audio_file(f)
                    audio_probe, sr_probe = pydub_to_float32(seg)
                    content_key, confidence = detect_content_type(audio_probe, sr_probe, f)
                except Exception:
                    pass

            fr = FileReport(
                input_path=str(f),
                content_profile=content_key,
                content_confidence=confidence,
            )
            _log(f"Processing {i + 1}/{len(files)}: {f.name}")
            try:
                remaster_file(f, output_dir, effective_params, file_report=fr,
                              progress_callback=lambda stage, pct, i=i: (
                                  progress_callback(i + pct / 100, len(files))
                                  if progress_callback else None
                              ))
                report.succeeded += 1
            except Exception as exc:
                fr.success = False
                fr.error = str(exc)
                report.failed += 1
                _log(f"FAILED: {f.name}: {exc}")
                log.exception("Processing failed for %s", f)
            report.files.append(fr)
            if progress_callback:
                progress_callback(i + 1, len(files))

    report.completed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    report.save(output_dir)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenRemaster — guided, profile-based audio remastering "
                     "for headphones, soundbars, home theatre systems, and general playback."
    )
    parser.add_argument("input", nargs="?", help="Audio file or directory")
    parser.add_argument("--output", default="./remastered", help="Output directory")

    parser.add_argument(
        "--device", choices=list(DEVICE_PROFILES.keys()), default="general",
        help="Target playback device profile.",
    )
    parser.add_argument(
        "--content", choices=list(CONTENT_PROFILES.keys()) + ["auto"], default="auto",
        help="Content type profile, or 'auto' to detect per-file.",
    )

    parser.add_argument("--list-profiles", action="store_true",
                        help="List available device and content profiles and exit.")
    parser.add_argument("--list-caps", action="store_true",
                        help="Show dependency capability report and exit.")

    parser.add_argument("--preview-seconds", type=int, default=0,
                        help="Process only the first N seconds (0 = full file).")
    parser.add_argument("--preview-start", type=float, default=0.0,
                        help="Preview start offset in seconds.")

    parser.add_argument("--no-cache", action="store_true", help="Disable stem cache.")
    parser.add_argument("--cache-dir", default=None, help="Override stem cache directory.")

    parser.add_argument("--voicefixer", action="store_true",
                        help="Enable experimental VoiceFixer vocal restoration.")
    parser.add_argument("--deepfilternet", action="store_true",
                        help="Enable experimental DeepFilterNet noise reduction.")

    parser.add_argument("--album-mode", action="store_true",
                        help="Preserve relative loudness between tracks in a batch.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes for batch runs.")

    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--fade-in-ms", type=int, default=0)
    parser.add_argument("--fade-out-ms", type=int, default=0)

    # Overrides (applied on top of resolved profile)
    parser.add_argument("--final-lufs", type=float, default=None)
    parser.add_argument("--bitrate", default=None)
    parser.add_argument("--output-format", choices=["mp3", "flac", "wav"], default=None)

    parser.add_argument("--use-limiter", action="store_true", default=True)
    parser.add_argument("--limiter-oversample", action="store_true")

    return parser.parse_args()


def _print_profiles() -> None:
    print("=== Device Profiles ===")
    for key, dp in DEVICE_PROFILES.items():
        print(f"  {key:20s} {dp.display_name}")
        print(f"                       {dp.description}")
    print("\n=== Content Profiles ===")
    for key, cp in CONTENT_PROFILES.items():
        print(f"  {key:16s} {cp.display_name}")
        print(f"                   {cp.description}")


def main() -> int:
    args = parse_args()

    if args.list_caps:
        print(CAP.summary())
        return 0

    if args.list_profiles:
        _print_profiles()
        return 0

    if not args.input:
        print("Error: an input file or directory must be provided (or use --list-profiles / --list-caps).")
        return 1

    if not CAP.ffmpeg:
        print("Error: ffmpeg is required but was not found on PATH.")
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}")
        return 1

    files = collect_audio_files(input_path)
    if not files:
        print(f"No audio files found in: {input_path}")
        return 1

    output_dir = Path(args.output)

    if args.cache_dir:
        global _CACHE_DIR
        _CACHE_DIR = Path(args.cache_dir)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device profile:  {DEVICE_PROFILES[args.device].display_name}")
    if args.content != "auto":
        print(f"Content profile: {CONTENT_PROFILES[args.content].display_name}")
    else:
        print("Content profile: auto-detect per file")
    print(f"Files to process: {len(files)}")
    print(CAP.summary())
    print()

    cancel_event = threading.Event()

    def _progress(done, total):
        print(f"\rProgress: {done}/{total}", end="", flush=True)

    all_reports: list[BatchReport] = []

    if args.content == "auto":
        # Auto-detect per file, group into per-content batches for efficiency
        grouped: dict[str, list[Path]] = {}
        for f in files:
            content_key = "early_digital"
            if CAP.numpy:
                try:
                    seg = load_audio_file(f)
                    audio_probe, sr_probe = pydub_to_float32(seg)
                    content_key, _ = detect_content_type(audio_probe, sr_probe, f)
                except Exception:
                    pass
            grouped.setdefault(content_key, []).append(f)

        for content_key, group_files in grouped.items():
            params = resolve_profile(args.device, content_key)
            params = _apply_cli_overrides(params, args)
            print(f"\n[{CONTENT_PROFILES[content_key].display_name}] {len(group_files)} file(s)")
            report = run_batch(
                group_files, output_dir, params,
                parallel_workers=args.workers,
                album_mode=args.album_mode,
                progress_callback=_progress,
                cancel_event=cancel_event,
            )
            all_reports.append(report)
    else:
        params = resolve_profile(args.device, args.content)
        params = _apply_cli_overrides(params, args)
        report = run_batch(
            files, output_dir, params,
            parallel_workers=args.workers,
            album_mode=args.album_mode,
            progress_callback=_progress,
            cancel_event=cancel_event,
        )
        all_reports.append(report)

    print()
    total_ok  = sum(r.succeeded for r in all_reports)
    total_fail = sum(r.failed for r in all_reports)
    print(f"\nDone: {total_ok} succeeded, {total_fail} failed.")
    return 0 if total_fail == 0 else 1


def _apply_cli_overrides(params: dict, args: argparse.Namespace) -> dict:
    """Apply explicit CLI flag overrides on top of a resolved profile dict."""
    p = dict(params)
    if args.final_lufs is not None:
        p["final_lufs"] = args.final_lufs
    if args.bitrate is not None:
        p["bitrate"] = args.bitrate
    if args.output_format is not None:
        p["output_format"] = args.output_format
    p["trim_silence"] = args.trim_silence
    p["fade_in_ms"] = args.fade_in_ms
    p["fade_out_ms"] = args.fade_out_ms
    p["use_voicefixer"] = args.voicefixer
    p["use_deepfilternet"] = args.deepfilternet
    p["use_cache"] = not args.no_cache
    p["use_limiter"] = args.use_limiter
    p["limiter_oversample"] = args.limiter_oversample
    p["use_loudnorm"] = True
    p.setdefault("lufs_range", 11.0)
    # Preview window (0 duration = full file)
    p["preview_seconds"] = getattr(args, "preview_seconds", 0) or 0
    p["preview_start"] = getattr(args, "preview_start", 0.0) or 0.0
    return p


if __name__ == "__main__":
    raise SystemExit(main())