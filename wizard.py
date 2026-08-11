#!/usr/bin/env python3
"""wizard.py — OpenRemaster wizard UI.

6-step wizard: Source → Target Device → Content Type → Enhancement →
DSP Review → Output & Run.

Requires engine.py, device_profiles.py, content_profiles.py, and
profile_resolver.py in the same folder.

Run with:
    python wizard.py
"""

from __future__ import annotations

import base64
import json
import logging
import math
import queue
import threading
import traceback
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

# engine's import is deliberately NOT done here at module load time —
# importing it triggers dependency probing (numpy/scipy/pyloudnorm/etc,
# and historically demucs/torch) which can take a noticeable moment.
# Doing that before the window has even appeared makes the app feel frozen
# on launch. Instead `core` starts as None and main() loads it in a
# background thread right after the first window paints, showing a loading
# screen in the meantime — see _load_core_and_launch() / main().
core = None  # type: ignore[assignment]
from device_profiles import DEVICE_PROFILES
from content_profiles import CONTENT_PROFILES
from profile_resolver import resolve_profile

CONFIG_PATH = Path.home() / ".openremaster" / "config.json"

# Base64-encoded PNG data for a small flat cassette-tape icon, used as
# the window/title-bar/taskbar icon (see _set_app_icon() in main()).
# Generated once from a vector-style PIL drawing; no external asset file
# is needed at runtime, so a copy of this script is always self-contained.
_APP_ICON_PNG_B64: dict[int, str] = {
    16: (
        "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAACyUlEQVR4nG2STYiVZRTHf+d5n/fzvt4ZxjtRJkwwUEGS0WJw"
        "4Ubb9LFyFgO2SZCRgkpiIFAos0VloW40ykXQB9FC26hBSAXlzGAYQTBJq0KHcUZvyPXe6/143+c5Le40TeV/eZ5z/ufP7zzC"
        "qk6o5gpxAtoF4S76+02g96JIC0BA2PbCmefrC+cP3FpZjFSMEfyagQ6aMMYAouqdH65tLka3PHlk/oOpk3JEdcOH25/7o9b5"
        "ZWRyaooAYS2AQCCCorQ6Dq9KGArnvjzNjeiRxr65T8dsAHGjvlxO79mtz778ql64VJcsjVAFRbndLknjgL3bhgmA3KIikRz/"
        "5OuiA7GNQVVEksjKUh0u/7Yi1TyhdJ40Ctj56ChZbHEFlB5yC9YGghiJQG0TSOOE2dlZ7nlwglHXIL4jOBXCQtD6Tfo24PLV"
        "AuchToRL87MkUUyrCfbKFcjznG+/Ocv3P8xB0aZTWgwFWRrT7jpad7pgQmJTkmRVxHeobXmG5WUwAKpKllUoipKxiUneP/0d"
        "04c+okuOTavMvPMxn52/yGM7dyM4sqyCqgJgB7QFV/QIq5s4/PYxFpb6PLVrKyuLv+NKx9OTU/y4cJ3X3zrKnl0/UTavImLW"
        "Gay7uIjgvIKCiEGMogreKyL//18DA1UkiOk3ljh8cIb9M69w4ew85z4/garn/rFxHt/6MG8ePETv1iJZEqPq/51A1ZOkGb9e"
        "PMPeua8Q36eSxgCcfGMfaiKs75DlQ6jv/ZOgNAh49d7r0MaNOjxSo9n4ExPkeD/YMpJW8K5gw9AmvKKNG9dAVQuD2M0P0fs5"
        "ykMtu+L7bfGqFN02QRDAKmkvgneOMo5ARNR1CaJKmI3Ts+8Z05yY/uI1Y8MDvds3Q4wxQ/cGMoC6XoL3Drx3tQeeKCrjO949"
        "ZYLGGtaXVKtcJ261gBaQ/2d+tZbn0LuP/imRBsBfVCw24lLxQNwAAAAASUVORK5CYII="
    ),
    32: (
        "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"
        "AAAASUVORK5CYII="
    ),
    48: (
        "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"
    ),
    128: (
        "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"
    ),
}


# Speaker-bias source options for the 5.1 speaker-mapping controls
# (Step 5, Space section). Ordered list of (internal_key, display_label);
# must match engine.SPEAKER_BIAS_SOURCE_KEYS. "off" means the
# group keeps its normal fixed derivation with no extra content blended in.
SPEAKER_BIAS_SOURCE_LABELS: list[tuple[str, str]] = [
    ("off", "Off"),
    ("vocals", "Vocals"),
    ("band1", "Band 1 – Bass (<150Hz)"),
    ("band2", "Band 2 – Low-mid (150–500Hz)"),
    ("band3", "Band 3 – Mid (500–2000Hz)"),
    ("band4", "Band 4 – High-mid (2–6kHz)"),
    ("band5", "Band 5 – High (>6kHz)"),
]

# Minimum Python version this app is tested/supported on.
MIN_PYTHON: tuple[int, int] = (3, 9)

# CAP attribute -> pip package name, for building install commands.
# ffmpeg/ffprobe are excluded (system binaries, not pip-installable).
PIP_PACKAGE_NAMES: dict[str, str] = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pyloudnorm": "pyloudnorm",
    "soundfile": "soundfile",
    "sounddevice": "sounddevice",
    "mutagen": "mutagen",
    "demucs": "demucs",
    "voicefixer": "voicefixer",
    "deepfilternet": "deepfilternet",
}

# Which of the above are required for full-quality (non-experimental) operation.
REQUIRED_TIER_KEYS = ["numpy", "scipy", "pyloudnorm", "soundfile", "sounddevice", "mutagen"]
OPTIONAL_TIER_KEYS = ["demucs", "voicefixer", "deepfilternet"]


def missing_packages(keys: list[str]) -> list[str]:
    """Return the subset of `keys` whose CAP flag is False."""
    cap = core.CAP
    return [k for k in keys if not getattr(cap, k, False)]


def pip_install_command(keys: list[str]) -> str:
    """Build a `pip install ...` command for the given CAP keys."""
    names = [PIP_PACKAGE_NAMES[k] for k in keys if k in PIP_PACKAGE_NAMES]
    return f"{sys.executable} -m pip install " + " ".join(names)


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_config(data: dict) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as exc:
        core.log.debug("Config save failed: %s", exc)


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    """A ttk.Frame with a vertical scrollbar for content taller than the view."""

    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)

        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _resize(event):
            canvas.itemconfig(window_id, width=event.width)
        canvas.bind("<Configure>", _resize)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _bind(_e=None):
            canvas.bind_all("<MouseWheel>", _wheel)
        def _unbind(_e=None):
            canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _bind)
        canvas.bind("<Leave>", _unbind)


class SelectableCard(ttk.Frame):
    """A large clickable card used for system/content profile selection.
    Click anywhere on the card to select it; selected state shown via a
    coloured border and background tint.
    """

    def __init__(self, parent, title: str, description: str, on_select, key: str):
        super().__init__(parent, relief="solid", borderwidth=2)
        self.key = key
        self.on_select = on_select
        self.selected = False

        self.title_label = tk.Label(
            self, text=title, font=("TkDefaultFont", 12, "bold"),
            anchor="w", justify="left", wraplength=260,
        )
        self.title_label.pack(fill="x", padx=10, pady=(8, 2))

        self.desc_label = tk.Label(
            self, text=description, font=("TkDefaultFont", 9),
            anchor="w", justify="left", wraplength=260, fg="#555555",
        )
        self.desc_label.pack(fill="x", padx=10, pady=(0, 8))

        for widget in (self, self.title_label, self.desc_label):
            widget.bind("<Button-1>", lambda e: self.on_select(self.key))

        # Re-wrap text to the card's actual rendered width so cards look
        # right whether the window is small or maximised, instead of
        # wrapping at a fixed 260px regardless of available space.
        self.bind("<Configure>", self._on_resize)

        self._set_style(False)

    def _on_resize(self, event):
        wrap = max(event.width - 20, 80)  # account for internal padx
        self.title_label.configure(wraplength=wrap)
        self.desc_label.configure(wraplength=wrap)

    def _set_style(self, selected: bool):
        self.selected = selected
        if selected:
            self.configure(relief="solid", borderwidth=3)
            try:
                self.configure(style="Selected.TFrame")
            except Exception:
                pass
            self.title_label.configure(fg="#1a5fb4")
        else:
            self.configure(relief="solid", borderwidth=1)
            try:
                self.configure(style="TFrame")
            except Exception:
                pass
            self.title_label.configure(fg="#000000")


class ConfidenceBadge(tk.Label):
    """Small colour-coded confidence indicator (green/yellow/red)."""

    def __init__(self, parent):
        super().__init__(parent, text="", font=("TkDefaultFont", 9, "bold"))

    def set_confidence(self, key: str, confidence: float):
        pct = int(confidence * 100)
        if confidence >= 0.6:
            colour = "#1b7a43"
            text = f"Auto-detected: {key} ({pct}% confident)"
        elif confidence >= 0.35:
            colour = "#e5a50a"
            text = f"Best guess: {key} ({pct}% confident) — please verify"
        else:
            colour = "#c01c28"
            text = f"Low confidence ({pct}%) — please select manually"
        self.configure(text=text, fg=colour)


class DependencyPanel(ttk.Frame):
    """Green/yellow/red dependency status panel.

    NOTE: no longer used on Step 6 -- Step 1's EnvironmentCheckPanel
    already surfaces missing dependencies at the start of the wizard,
    so repeating it on the final step was redundant. Left in place
    (unused) in case a future page wants a compact status readout.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.rows: dict[str, tk.Label] = {}
        self._build()

    def _build(self):
        items = [
            ("ffmpeg", "Required — audio encode/decode"),
            ("ffprobe", "Required — audio inspection"),
            ("numpy", "DSP quality — required for all enhancement"),
            ("scipy", "DSP quality — LR crossovers, EQ"),
            ("pyloudnorm", "Accurate LUFS loudness measurement"),
            ("soundfile", "Required for 5.1 surround output"),
            ("sounddevice", "In-app audio preview playback"),
            ("mutagen", "Metadata read/write, content auto-detection"),
            ("demucs", "Stem separation (vocal/instrument chains)"),
            ("voicefixer", "Experimental AI vocal restoration"),
            ("deepfilternet", "Experimental AI noise reduction"),
        ]
        for i, (key, desc) in enumerate(items):
            dot = tk.Label(self, text="●", font=("TkDefaultFont", 12))
            dot.grid(row=i, column=0, sticky="w", padx=(4, 6), pady=1)
            name = tk.Label(self, text=key, font=("TkDefaultFont", 9, "bold"), width=14, anchor="w")
            name.grid(row=i, column=1, sticky="w")
            desc_lbl = tk.Label(self, text=desc, font=("TkDefaultFont", 9), fg="#555555", anchor="w")
            desc_lbl.grid(row=i, column=2, sticky="w", padx=(6, 0))
            self.rows[key] = dot

    def refresh(self):
        cap = core.CAP
        status = {
            "ffmpeg": bool(cap.ffmpeg), "ffprobe": bool(cap.ffprobe),
            "numpy": cap.numpy, "scipy": cap.scipy,
            "pyloudnorm": cap.pyloudnorm, "soundfile": cap.soundfile,
            "sounddevice": cap.sounddevice, "mutagen": cap.mutagen,
            "demucs": cap.demucs, "voicefixer": cap.voicefixer,
            "deepfilternet": cap.deepfilternet,
        }
        for key, dot in self.rows.items():
            ok = status.get(key, False)
            dot.configure(fg="#1b7a43" if ok else "#c01c28")


class EnvironmentCheckPanel(ttk.LabelFrame):
    """Step 1 environment check: Python version support + required/optional
    plugin (pip package) availability, with a ready-to-copy pip install
    command for anything missing.
    """

    def __init__(self, parent):
        super().__init__(parent, text="Environment Check")
        self.columnconfigure(0, weight=1)

        self.python_label = tk.Label(self, font=("TkDefaultFont", 9, "bold"), anchor="w", justify="left")
        self.python_label.grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))

        self.ffmpeg_label = tk.Label(self, font=("TkDefaultFont", 9), anchor="w", justify="left")
        self.ffmpeg_label.grid(row=1, column=0, sticky="w", padx=6, pady=(0, 4))

        self.required_label = tk.Label(self, font=("TkDefaultFont", 9), anchor="w", justify="left", wraplength=560)
        self.required_label.grid(row=2, column=0, sticky="w", padx=6, pady=(0, 2))

        self.pip_entry = tk.Entry(self, font=("TkFixedFont", 9))
        self.pip_entry.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))
        self.pip_entry.configure(state="readonly")

        self.optional_label = tk.Label(self, font=("TkDefaultFont", 8), fg="#888888", anchor="w",
                                       justify="left", wraplength=560)
        self.optional_label.grid(row=4, column=0, sticky="w", padx=6, pady=(0, 6))

        self.refresh()

    def refresh(self):
        # --- Python version ---
        py_version = ".".join(str(v) for v in sys.version_info[:3])
        min_str = ".".join(str(v) for v in MIN_PYTHON)
        if sys.version_info[:2] >= MIN_PYTHON:
            self.python_label.configure(
                text=f"✔ Python {py_version} (requires {min_str}+)", fg="#1b7a43",
            )
        else:
            self.python_label.configure(
                text=f"✘ Python {py_version} is not supported — requires {min_str} or newer. "
                     f"Please upgrade your Python interpreter.",
                fg="#c01c28",
            )

        # --- ffmpeg / ffprobe (required system binaries) ---
        cap = core.CAP
        if cap.ffmpeg and cap.ffprobe:
            self.ffmpeg_label.configure(text="✔ ffmpeg / ffprobe found on PATH", fg="#1b7a43")
        else:
            missing_bins = [n for n, v in (("ffmpeg", cap.ffmpeg), ("ffprobe", cap.ffprobe)) if not v]
            self.ffmpeg_label.configure(
                text=f"✘ Missing required binary: {', '.join(missing_bins)}. "
                     f"Install via your system package manager (e.g. 'apt install ffmpeg', "
                     f"'brew install ffmpeg', or download from ffmpeg.org) and ensure it's on PATH.",
                fg="#c01c28",
            )

        # --- Required-tier Python packages ---
        missing_req = missing_packages(REQUIRED_TIER_KEYS)
        if not missing_req:
            self.required_label.configure(text="✔ All required plugins installed (numpy, scipy, pyloudnorm, "
                                                "soundfile, sounddevice, mutagen).", fg="#1b7a43")
            self.pip_entry.grid_remove()
        else:
            self.required_label.configure(
                text=f"✘ Missing plugins: {', '.join(missing_req)}. Run this to install:",
                fg="#c01c28",
            )
            self.pip_entry.grid()
            self.pip_entry.configure(state="normal")
            self.pip_entry.delete(0, "end")
            self.pip_entry.insert(0, pip_install_command(missing_req))
            self.pip_entry.configure(state="readonly")

        # --- Optional-tier packages (informational only) ---
        missing_opt = missing_packages(OPTIONAL_TIER_KEYS)
        if missing_opt:
            self.optional_label.configure(
                text=f"Optional (experimental/stem features) not installed: {', '.join(missing_opt)}. "
                     f"Install with: {pip_install_command(missing_opt)}"
            )
        else:
            self.optional_label.configure(text="All optional plugins installed.")


# ---------------------------------------------------------------------------
# EQ curve canvas
# ---------------------------------------------------------------------------

class EQCurveCanvas(tk.Canvas):
    """Draws a simple frequency-response curve for the active EQ settings.
    Computes combined biquad response at 100 log-spaced points without
    requiring matplotlib.
    """

    def __init__(self, parent, width=420, height=140):
        super().__init__(parent, width=width, height=height, bg="#1e1e1e", highlightthickness=0)
        self.width = width
        self.height = height

    def redraw(self, params: dict):
        self.delete("all")
        freqs = [20 * (10 ** (i / 99 * 3)) for i in range(100)]  # 20Hz–20kHz log

        def response_db(f):
            total = 0.0
            if params.get("bass_shelf_db", 0) and params.get("bass_shelf_hz", 0) > 0:
                total += _shelf_response_db(f, params.get("bass_shelf_hz", 100), params["bass_shelf_db"], "low")
            if params.get("treble_shelf_db", 0) and params.get("treble_shelf_hz", 0) > 0:
                total += _shelf_response_db(f, params.get("treble_shelf_hz", 8000), params["treble_shelf_db"], "high")
            if params.get("presence_db", 0) and params.get("presence_hz", 0) > 0:
                total += _peak_response_db(f, params.get("presence_hz", 2800), params["presence_db"], params.get("presence_q", 1.2))
            if params.get("notch_db", 0) and params.get("notch_hz", 0) > 0:
                total += _peak_response_db(f, params.get("notch_hz", 450), params["notch_db"], params.get("notch_q", 2.0))
            return total

        points = [(f, response_db(f)) for f in freqs]
        max_db = max(6.0, max(abs(db) for _, db in points) + 1)

        # Axis: 0dB line
        mid_y = self.height / 2
        self.create_line(0, mid_y, self.width, mid_y, fill="#444444", dash=(2, 2))

        # Draw curve
        coords = []
        for i, (f, db) in enumerate(points):
            x = i / (len(points) - 1) * self.width
            y = mid_y - (db / max_db) * (self.height / 2 - 10)
            coords.extend([x, y])
        if len(coords) >= 4:
            self.create_line(*coords, fill="#4fc3f7", width=2, smooth=True)

        self.create_text(6, 6, text=f"+{max_db:.0f}dB", fill="#888888", anchor="nw", font=("TkDefaultFont", 7))
        self.create_text(6, self.height - 6, text=f"-{max_db:.0f}dB", fill="#888888", anchor="sw", font=("TkDefaultFont", 7))


def _peak_response_db(f, f0, gain_db, q):
    if f0 <= 0:
        return 0.0
    ratio = f / f0
    bw = ratio - 1 / max(ratio, 1e-6)
    x = q * bw
    return gain_db / (1 + x * x)


def _shelf_response_db(f, f0, gain_db, shelf_type):
    if f0 <= 0:
        return 0.0
    ratio = f / f0
    if shelf_type == "low":
        return gain_db / (1 + ratio ** 2) if ratio > 1 else gain_db * (1 - 1 / (1 + (1/max(ratio,1e-6)) ** 2))
    else:
        return gain_db / (1 + (1 / max(ratio, 1e-6)) ** 2) if ratio < 1 else gain_db * (1 - 1/(1+ratio**2))


# ---------------------------------------------------------------------------
# Wizard state (shared across all steps)
# ---------------------------------------------------------------------------

class WizardState:
    """Central mutable state shared by all wizard pages. Persisted to
    config.json between sessions."""

    def __init__(self):
        cfg = load_config()
        self.input_path: str = cfg.get("input_path", "")
        self.input_mode: str = cfg.get("input_mode", "file")  # file | folder
        self.output_dir: str = cfg.get("output_dir", str(Path.home() / "RemasterStudio" / "output"))
        self.output_format: str = cfg.get("output_format", "mp3")
        self.bitrate: str = cfg.get("bitrate", "320k")
        self.album_mode: bool = cfg.get("album_mode", False)
        self.parallel_workers: int = cfg.get("parallel_workers", 1)
        self.preview_start_s: float = cfg.get("preview_start_s", 0.0)
        self.preview_duration_s: int = cfg.get("preview_duration_s", 15)

        self.device_key: str = cfg.get("device_key", "")
        self.content_key: str = cfg.get("content_key", "")  # "" = auto-detect
        self.content_auto_detected: str = ""
        self.content_confidence: float = 0.0

        self.demucs_model: str = cfg.get("demucs_model", "htdemucs")
        self.use_cache: bool = cfg.get("use_cache", True)
        self.use_voicefixer: bool = cfg.get("use_voicefixer", False)
        self.use_deepfilternet: bool = cfg.get("use_deepfilternet", False)
        self.experimental_ack: bool = cfg.get("experimental_ack", False)

        # DSP overrides (populated from resolved profile, user-editable in Step 5)
        self.params: dict = {}

        self.trim_silence: bool = cfg.get("trim_silence", False)
        self.fade_in_ms: int = cfg.get("fade_in_ms", 0)
        self.fade_out_ms: int = cfg.get("fade_out_ms", 0)

        self.surround_codec: str = cfg.get("surround_codec", "ac3")
        self.lfe_mode: str = cfg.get("lfe_mode", "gentle")

    def resolve(self) -> dict:
        """Merge device + content profiles, then apply any Step 5 overrides."""
        content = self.content_key or self.content_auto_detected or "early_digital"
        p = resolve_profile(self.device_key, content)
        p.update(self.params)  # user overrides win
        p["output_format"] = self.output_format
        p["bitrate"] = self.bitrate
        p["trim_silence"] = self.trim_silence
        p["fade_in_ms"] = self.fade_in_ms
        p["fade_out_ms"] = self.fade_out_ms
        p["demucs_model"] = self.demucs_model
        # Selecting "none" in Step 4 must fully disable stem separation,
        # even when the content profile would otherwise request it.
        if self.demucs_model == "none":
            p["use_stems"] = False
            p["use_demucs"] = False
        p["use_cache"] = self.use_cache
        p["use_voicefixer"] = self.use_voicefixer and self.experimental_ack
        p["use_deepfilternet"] = self.use_deepfilternet and self.experimental_ack
        p["surround_codec"] = self.surround_codec
        p["lfe_mode"] = self.lfe_mode
        p["use_limiter"] = True
        p["use_loudnorm"] = True
        p.setdefault("lufs_range", 11.0)
        return p

    def save(self):
        save_config({
            "input_path": self.input_path, "input_mode": self.input_mode,
            "output_dir": self.output_dir, "output_format": self.output_format,
            "bitrate": self.bitrate, "album_mode": self.album_mode,
            "parallel_workers": self.parallel_workers,
            "preview_start_s": self.preview_start_s,
            "preview_duration_s": self.preview_duration_s,
            "device_key": self.device_key, "content_key": self.content_key,
            "demucs_model": self.demucs_model, "use_cache": self.use_cache,
            "use_voicefixer": self.use_voicefixer,
            "use_deepfilternet": self.use_deepfilternet,
            "experimental_ack": self.experimental_ack,
            "trim_silence": self.trim_silence,
            "fade_in_ms": self.fade_in_ms, "fade_out_ms": self.fade_out_ms,
            "surround_codec": self.surround_codec, "lfe_mode": self.lfe_mode,
        })

    def cleanup_and_save(self):
        """Called on clean app exit (window closed normally). Only the
        input file/folder selection is retained across sessions — every
        other wizard choice (system/content profile, DSP slider overrides,
        preview window, batch/experimental options, etc.) resets to its
        default the next time the app is launched, rather than silently
        carrying over stale values like a leftover preview duration."""
        save_config({
            "input_path": self.input_path,
            "input_mode": self.input_mode,
        })


# ---------------------------------------------------------------------------
# Step 1 — Source & Output
# ---------------------------------------------------------------------------

class Step1Source(ttk.Frame):
    def __init__(self, parent, state: WizardState):
        super().__init__(parent)
        self.state = state
        self.columnconfigure(1, weight=1)

        r = 0
        self.env_panel = EnvironmentCheckPanel(self)
        self.env_panel.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        r += 1

        self.mode_var = tk.StringVar(value=state.input_mode)
        ttk.Label(self, text="Input:").grid(row=r, column=0, sticky="w")
        mode_frame = ttk.Frame(self)
        mode_frame.grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(mode_frame, text="Single file", variable=self.mode_var, value="file").pack(side="left")
        ttk.Radiobutton(mode_frame, text="Folder (batch)", variable=self.mode_var, value="folder").pack(side="left", padx=(12, 0))
        r += 1

        self.input_var = tk.StringVar(value=state.input_path)
        ttk.Entry(self, textvariable=self.input_var).grid(row=r, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(self, text="Browse…", command=self._browse_input).grid(row=r, column=2, padx=(6, 0))
        r += 1

        ttk.Label(self, text="Output folder:").grid(row=r, column=0, sticky="w", pady=(8, 0))
        r += 1
        self.output_var = tk.StringVar(value=state.output_dir)
        ttk.Entry(self, textvariable=self.output_var).grid(row=r, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(self, text="Browse…", command=self._browse_output).grid(row=r, column=2, padx=(6, 0))
        r += 1

        opts = ttk.Frame(self)
        opts.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Label(opts, text="Format:").grid(row=0, column=0, sticky="w")
        self.fmt_var = tk.StringVar(value=state.output_format)
        ttk.Combobox(opts, textvariable=self.fmt_var, state="readonly", values=["mp3", "flac", "wav"], width=8).grid(row=0, column=1, padx=(4, 16))
        ttk.Label(opts, text="Bitrate:").grid(row=0, column=2, sticky="w")
        self.bitrate_var = tk.StringVar(value=state.bitrate)
        ttk.Entry(opts, textvariable=self.bitrate_var, width=8).grid(row=0, column=3, padx=(4, 0))
        r += 1

        batch_frame = ttk.LabelFrame(self, text="Batch options")
        batch_frame.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(16, 0))
        self.album_var = tk.BooleanVar(value=state.album_mode)
        ttk.Checkbutton(batch_frame, text="Album mode — preserve relative loudness between tracks",
                        variable=self.album_var).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=4)
        ttk.Label(batch_frame, text="Parallel workers:").grid(row=1, column=0, sticky="w", padx=6)
        self.workers_var = tk.IntVar(value=state.parallel_workers)
        ttk.Spinbox(batch_frame, textvariable=self.workers_var, from_=1, to=8, width=6).grid(row=1, column=1, sticky="w", padx=6, pady=4)
        r += 1

        preview_frame = ttk.LabelFrame(self, text="Preview segment")
        preview_frame.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Label(preview_frame, text="Start (seconds):").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.preview_start_var = tk.DoubleVar(value=state.preview_start_s)
        ttk.Spinbox(preview_frame, textvariable=self.preview_start_var, from_=0, to=3600, width=8).grid(row=0, column=1, padx=6)
        ttk.Label(preview_frame, text="Duration (seconds):").grid(row=0, column=2, sticky="w", padx=(12, 6))
        self.preview_dur_var = tk.IntVar(value=state.preview_duration_s)
        ttk.Spinbox(preview_frame, textvariable=self.preview_dur_var, from_=5, to=60, width=6).grid(row=0, column=3, padx=6)

        self.disk_label = ttk.Label(self, text="", foreground="#e5a50a")
        self.disk_label.grid(row=r + 1, column=0, columnspan=3, sticky="w", pady=(10, 0))

        # Recompute the estimate whenever format/bitrate change too, not
        # just on browse — a bitrate change can noticeably move the
        # estimate for a large batch.
        self.fmt_var.trace_add("write", lambda *_: self._update_disk_estimate())
        self.bitrate_var.trace_add("write", lambda *_: self._update_disk_estimate())
        self._update_disk_estimate()

    def _browse_input(self):
        if self.mode_var.get() == "file":
            path = filedialog.askopenfilename(
                title="Select audio file",
                filetypes=[("Audio files", "*.mp3 *.flac *.wav *.m4a *.aac *.ogg")],
            )
        else:
            path = filedialog.askdirectory(title="Select folder of audio files")
        if path:
            self.input_var.set(path)
            self._update_disk_estimate()

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)
            self._update_disk_estimate()

    def _update_disk_estimate(self):
        """Populate the previously-unwired disk_label with an estimate of
        output size and free space, using engine.estimate_disk_usage /
        check_disk_space (both already existed but nothing called them
        from here). This runs before Step 2/3 have set a device or
        content profile, so it assumes plain stereo output at the chosen
        format/bitrate — a conservative baseline; a 5.1 device profile
        chosen later will use more space (extra MKV/AC3 track) than this
        estimate shows.
        """
        input_str = self.input_var.get().strip()
        output_str = self.output_var.get().strip()
        if not input_str or not output_str or core is None:
            self.disk_label.configure(text="")
            return
        p = Path(input_str)
        if not p.exists():
            self.disk_label.configure(text="")
            return
        try:
            files = core.collect_audio_files(p)
            if not files:
                self.disk_label.configure(text="")
                return
            params = {
                "layout": "stereo",
                "output_format": self.fmt_var.get(),
                "bitrate": self.bitrate_var.get().strip() or "320k",
                "also_produce_stereo_mp3": False,
            }
            estimated = core.estimate_disk_usage(files, params)
            out_dir = Path(output_str).expanduser()
            check_dir = out_dir
            while not check_dir.exists() and check_dir != check_dir.parent:
                check_dir = check_dir.parent
            ok, free = core.check_disk_space(check_dir, estimated)
            gb = estimated / 1e9
            note = " (stereo estimate — 5.1 output will use more)"
            if free >= 0:
                free_gb = free / 1e9
                if ok:
                    self.disk_label.configure(
                        text=f"Estimated output size: ~{gb:.2f} GB for {len(files)} file(s), "
                             f"{free_gb:.1f} GB free.{note}",
                        foreground="#555555",
                    )
                else:
                    self.disk_label.configure(
                        text=f"⚠ Estimated output ~{gb:.2f} GB may exceed free disk space "
                             f"({free_gb:.1f} GB free).{note}",
                        foreground="#c01c28",
                    )
            else:
                self.disk_label.configure(
                    text=f"Estimated output size: ~{gb:.2f} GB for {len(files)} file(s).{note}",
                    foreground="#555555",
                )
        except Exception:
            self.disk_label.configure(text="")

    def commit(self):
        self.state.input_mode = self.mode_var.get()
        self.state.input_path = self.input_var.get().strip()
        self.state.output_dir = self.output_var.get().strip()
        self.state.output_format = self.fmt_var.get()
        self.state.bitrate = self.bitrate_var.get().strip() or "320k"
        self.state.album_mode = self.album_var.get()
        self.state.parallel_workers = self.workers_var.get()
        self.state.preview_start_s = self.preview_start_var.get()
        self.state.preview_duration_s = self.preview_dur_var.get()

    def validate(self) -> str | None:
        if not self.input_var.get().strip():
            return "Please select an input file or folder."
        p = Path(self.input_var.get().strip())
        if not p.exists():
            return f"Input path does not exist: {p}"
        return None


# ---------------------------------------------------------------------------
# Step 2 — Target Device
# ---------------------------------------------------------------------------

class Step2Device(ttk.Frame):
    """Device profile picker. Uses a dynamic grid (not a fixed 2x2) so the
    number of device profiles can grow — e.g. new soundbar/home-theatre
    categories — without any change to this layout code.
    """

    def __init__(self, parent, state: WizardState):
        super().__init__(parent)
        self.state = state
        self.cards: dict[str, SelectableCard] = {}
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Label(self, text="What will you play this remastered audio on?",
                 foreground="#555555").grid(row=0, column=0, sticky="w", pady=(0, 12))

        grid = ttk.Frame(self)
        grid.grid(row=1, column=0, sticky="nsew")

        keys = list(DEVICE_PROFILES.keys())
        num_cols = 2 if len(keys) <= 6 else 3
        num_rows = math.ceil(len(keys) / num_cols)
        for c in range(num_cols):
            grid.columnconfigure(c, weight=1)
        for r in range(num_rows):
            grid.rowconfigure(r, weight=1)

        for i, key in enumerate(keys):
            dp = DEVICE_PROFILES[key]
            row, col = i // num_cols, i % num_cols
            card = SelectableCard(grid, dp.display_name, dp.description, self._select, key)
            card.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)
            self.cards[key] = card

        # First time through (no device chosen yet), pre-select the first
        # tile so the step never starts with nothing selected. On later
        # visits, keep whatever was already chosen.
        initial_key = state.device_key or (keys[0] if keys else "")
        if initial_key:
            self._select(initial_key)

    def _select(self, key: str):
        self.state.device_key = key
        for k, card in self.cards.items():
            card._set_style(k == key)

    def commit(self):
        pass  # already committed via _select

    def validate(self) -> str | None:
        if not self.state.device_key:
            return "Please select a target device before continuing."
        return None


# ---------------------------------------------------------------------------
# Step 3 — Content Type
# ---------------------------------------------------------------------------

class Step3Content(ttk.Frame):
    def __init__(self, parent, state: WizardState):
        super().__init__(parent)
        self.state = state
        self.cards: dict[str, SelectableCard] = {}
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        ttk.Label(self, text="What kind of recording is this? (auto-detected — override if needed)",
                 foreground="#555555").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.badge = ConfidenceBadge(self)
        self.badge.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

        grid = ttk.Frame(self)
        grid.grid(row=2, column=0, columnspan=2, sticky="nsew")

        # Content profiles are keyed by recording era/style (not language),
        # so there's no natural pairing to group by — a flat grid in dict
        # order is enough. Same dynamic-grid approach as Step2Device, so
        # adding/removing a content profile never requires touching layout
        # code here.
        all_keys = list(CONTENT_PROFILES.keys())
        num_cols = 2 if len(all_keys) <= 6 else 3
        num_rows = math.ceil(len(all_keys) / num_cols)
        for c in range(num_cols):
            grid.columnconfigure(c, weight=1)
        for r in range(num_rows):
            grid.rowconfigure(r, weight=1)

        for i, key in enumerate(all_keys):
            cp = CONTENT_PROFILES[key]
            row, col = i // num_cols, i % num_cols
            card = SelectableCard(grid, cp.display_name, cp.description, self._select, key)
            card.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)
            self.cards[key] = card

        self._detected = False
        self._detected_input: str | None = None

    def run_detection(self, log_fn=None):
        """Run auto-detection on the first input file. Called when the
        wizard advances into this step.

        Re-runs whenever the input path/mode has changed since the last
        detection (e.g. the user went back to Step 1 and picked a
        different file or folder) rather than only ever running once per
        app session — otherwise the badge and pre-selected content type
        keep reflecting whatever file was loaded first, even after the
        input changes.
        """
        input_signature = f"{self.state.input_mode}:{self.state.input_path}"
        if self._detected and self._detected_input == input_signature:
            return
        self._detected = True
        self._detected_input = input_signature
        try:
            p = Path(self.state.input_path)
            files = core.collect_audio_files(p)
            if not files:
                return
            seg = core.load_audio_file(files[0])
            if core.CAP.numpy:
                audio, sr = core.pydub_to_float32(seg)
                key, conf = core.detect_content_type(audio, sr, files[0])
            else:
                key, conf = "early_digital", 0.3
            self.state.content_auto_detected = key
            self.state.content_confidence = conf
            self.badge.set_confidence(CONTENT_PROFILES[key].display_name, conf)
            if not self.state.content_key:  # only auto-select if user hasn't chosen
                self._select(key)
        except Exception as exc:
            if log_fn:
                log_fn(f"Content auto-detection failed: {exc}")

    def _select(self, key: str):
        self.state.content_key = key
        for k, card in self.cards.items():
            card._set_style(k == key)

    def commit(self):
        pass

    def validate(self) -> str | None:
        if not self.state.content_key:
            return "Please select a content type."
        return None


# ---------------------------------------------------------------------------
# Step 4 — Enhancement
# ---------------------------------------------------------------------------

class Step4Enhancement(ttk.Frame):
    def __init__(self, parent, state: WizardState):
        super().__init__(parent)
        self.state = state
        self.columnconfigure(0, weight=1)

        stem_frame = ttk.LabelFrame(self, text="Stem Separation (Demucs)")
        stem_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(stem_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.model_var = tk.StringVar(value=state.demucs_model)
        ttk.Combobox(stem_frame, textvariable=self.model_var, state="readonly",
                    values=["htdemucs", "htdemucs_ft", "none"], width=14).grid(row=0, column=1, padx=6)
        self.cache_var = tk.BooleanVar(value=state.use_cache)
        ttk.Checkbutton(stem_frame, text="Use stem cache (faster repeated runs)",
                        variable=self.cache_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        if not core.CAP.demucs:
            warn = ttk.Label(stem_frame, text="⚠ Demucs not installed — stem processing will be skipped.",
                            foreground="#e5a50a")
            warn.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))

        exp_frame = ttk.LabelFrame(self, text="Experimental AI Tools (collapsed by default)")
        exp_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        self.exp_expanded = tk.BooleanVar(value=False)
        self.exp_toggle = ttk.Checkbutton(
            exp_frame, text="▶ Show experimental tools (may degrade musical quality)",
            variable=self.exp_expanded, command=self._toggle_experimental,
        )
        self.exp_toggle.grid(row=0, column=0, sticky="w", padx=6, pady=4)

        self.exp_body = ttk.Frame(exp_frame)

        warn_label = tk.Label(
            self.exp_body,
            text="These tools were designed for speech, not music. VoiceFixer can strip "
                 "musical qualities from singing; DeepFilterNet can suppress instruments "
                 "it misclassifies as noise. Use only if you understand the risk.",
            fg="#c01c28", wraplength=500, justify="left",
        )
        warn_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 8))

        self.ack_var = tk.BooleanVar(value=state.experimental_ack)
        ttk.Checkbutton(self.exp_body, text="I understand this may degrade quality — enable anyway",
                        variable=self.ack_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 8))

        self.vf_var = tk.BooleanVar(value=state.use_voicefixer)
        ttk.Checkbutton(self.exp_body, text="VoiceFixer (AI vocal restoration)",
                        variable=self.vf_var).grid(row=2, column=0, sticky="w", padx=6)
        if not core.CAP.voicefixer:
            ttk.Label(self.exp_body, text="not installed", foreground="#888888").grid(row=2, column=1, sticky="w")

        self.dfn_var = tk.BooleanVar(value=state.use_deepfilternet)
        ttk.Checkbutton(self.exp_body, text="DeepFilterNet (AI noise reduction)",
                        variable=self.dfn_var).grid(row=3, column=0, sticky="w", padx=6, pady=(0, 6))
        if not core.CAP.deepfilternet:
            ttk.Label(self.exp_body, text="not installed", foreground="#888888").grid(row=3, column=1, sticky="w")

    def _toggle_experimental(self):
        if self.exp_expanded.get():
            self.exp_body.grid(row=1, column=0, sticky="ew")
            self.exp_toggle.configure(text="▼ Show experimental tools (may degrade musical quality)")
        else:
            self.exp_body.grid_remove()
            self.exp_toggle.configure(text="▶ Show experimental tools (may degrade musical quality)")

    def commit(self):
        self.state.demucs_model = self.model_var.get()
        self.state.use_cache = self.cache_var.get()
        self.state.experimental_ack = self.ack_var.get()
        self.state.use_voicefixer = self.vf_var.get()
        self.state.use_deepfilternet = self.dfn_var.get()

    def validate(self) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Step 5 — DSP Review & Preview
# ---------------------------------------------------------------------------

class CollapsibleSection(ttk.Frame):
    """A titled section that can be expanded/collapsed via a toggle button."""

    def __init__(self, parent, title: str, expanded: bool = True):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self._expanded = tk.BooleanVar(value=expanded)
        self.toggle_btn = ttk.Checkbutton(
            self, text=("▼ " if expanded else "▶ ") + title,
            variable=self._expanded, command=self._on_toggle,
            style="Toolbutton",
        )
        self.toggle_btn.grid(row=0, column=0, sticky="w", pady=(6, 2))
        self.title = title
        self.body = ttk.Frame(self)
        if expanded:
            self.body.grid(row=1, column=0, sticky="ew", padx=(16, 0))

    def _on_toggle(self):
        if self._expanded.get():
            self.body.grid(row=1, column=0, sticky="ew", padx=(16, 0))
            self.toggle_btn.configure(text="▼ " + self.title)
        else:
            self.body.grid_remove()
            self.toggle_btn.configure(text="▶ " + self.title)


class Step5DSP(ttk.Frame):
    def __init__(self, parent, state: WizardState):
        super().__init__(parent)
        self.state = state
        self.columnconfigure(0, weight=1)
        self.vars: dict[str, tk.Variable] = {}
        self.ideal_labels: dict[str, tk.Label] = {}
        self.last_recommendations: dict = {}
        self.last_measurements: dict = {}
        self._preview_playing = False
        self._active = True  # False once the user navigates away from this step

        # Analysis report and the Device/Content label are no longer shown
        # at the top of the screen — the widgets still exist (and are kept
        # up to date by refresh_profile_label()/_analyze_track() etc. below)
        # so none of that logic needs to change, they're just never gridded.
        self.analysis_status = ttk.Label(self, text="", foreground="#555555", wraplength=700, justify="left")
        self.profile_label = ttk.Label(self, text="", foreground="#555555")

        scrollable = ScrollableFrame(self)
        scrollable.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        body = scrollable.inner
        body.columnconfigure(0, weight=1)

        # --- EQ section (with live curve) ---
        eq_section = CollapsibleSection(body, "Tonal EQ", expanded=True)
        eq_section.grid(row=0, column=0, sticky="ew")
        self.eq_canvas = EQCurveCanvas(eq_section.body)
        self.eq_canvas.grid(row=0, column=0, columnspan=4, pady=(4, 8))
        ttk.Label(
            eq_section.body,
            text="Shapes the final master's overall tone. Applies to every profile, "
                 "regardless of whether stem separation is used.",
            foreground="#888888", font=("TkDefaultFont", 8), wraplength=520, justify="left",
        ).grid(row=5, column=0, columnspan=4, sticky="w", padx=6, pady=(4, 0))

        self._eq_slider("bass_shelf_db", "Bass shelf (dB)", eq_section.body, 1, -12, 12)
        self._eq_slider("treble_shelf_db", "Treble shelf (dB)", eq_section.body, 2, -12, 12)
        self._eq_slider("presence_db", "Presence peak (dB)", eq_section.body, 3, -12, 12)
        self._eq_slider("notch_db", "Notch cut (dB)", eq_section.body, 4, -12, 12)

        # --- Dynamics section ---
        dyn_section = CollapsibleSection(body, "Dynamics", expanded=True)
        dyn_section.grid(row=1, column=0, sticky="ew")
        self._bool_row("multiband_compress", "Multiband compression", dyn_section.body, 0)
        self._bool_row("deesser", "De-esser (vocal sibilance)", dyn_section.body, 1)
        self._float_row("final_lufs", "Target loudness (LUFS)", dyn_section.body, 2, -30, -5)
        self._bool_row("auto_loudness", "Auto loudness (let ffmpeg two-pass loudnorm decide)", dyn_section.body, 3)
        self._float_row("headroom_db", "Peak headroom (dB)", dyn_section.body, 4, 0, 3)
        ttk.Label(
            dyn_section.body,
            text="Controls compression and overall output loudness. Target loudness sets "
                 "the final master's level (applies always); multiband compression and the "
                 "de-esser shape dynamics before that final level is set.",
            foreground="#888888", font=("TkDefaultFont", 8), wraplength=520, justify="left",
        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 0))

        # --- Vocal Chain section (stem-based vocal processing) ---
        vocal_section = CollapsibleSection(body, "Vocal Chain (stem-based)", expanded=True)
        vocal_section.grid(row=2, column=0, sticky="ew")
        # Dynamic status -- updated in refresh_profile_label() to clearly show
        # whether these controls will have ANY effect for the current
        # system+content selection. This is the single most common source of
        # "I changed the slider and nothing happened" confusion: several
        # content profiles (Tamil Classic, Hindi Classic, Carnatic, BGM)
        # never use stem separation at all, and the quick Preview always
        # skips it for speed regardless of profile.
        self.vocal_stems_status = ttk.Label(
            vocal_section.body, text="", font=("TkDefaultFont", 8, "bold"),
            wraplength=520, justify="left",
        )
        self.vocal_stems_status.grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 2))
        ttk.Label(
            vocal_section.body,
            text="To make vocals louder or quieter relative to the instruments, use "
                 "'Vocal volume' below — the tone sliders (presence/air/mud) only reshape "
                 "the vocal's EQ curve, they don't change its overall level (it gets "
                 "re-normalised to the volume target regardless of EQ).",
            foreground="#888888", font=("TkDefaultFont", 8), wraplength=520, justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))
        self._float_row("vocal_lufs", "Vocal volume (LUFS, lower = quieter)", vocal_section.body, 2, -30, -5)
        self._eq_slider("vocal_presence_db", "Vocal presence tone (3kHz peak, dB)", vocal_section.body, 3, -12, 12)
        self._eq_slider("vocal_air_db", "Vocal air tone (10kHz shelf, dB)", vocal_section.body, 4, -12, 12)
        self._eq_slider("vocal_mud_cut_db", "Vocal mud-cut tone (200Hz shelf, dB)", vocal_section.body, 5, -12, 12)

        # --- Space section ---
        space_section = CollapsibleSection(body, "Space (stereo / surround)", expanded=True)
        space_section.grid(row=3, column=0, sticky="ew")
        self._bool_row("width_bands", "Per-band stereo width", space_section.body, 0)
        self._bool_row("saturation", "Tape saturation", space_section.body, 1)
        self._bool_row("crystalizer", "Crystalizer (clarity)", space_section.body, 2)
        self._bool_row("crossfeed", "Headphone crossfeed", space_section.body, 3)
        self._float_row("width_mid", "Stereo width — mid band", space_section.body, 4, 0.5, 3.0)
        self._float_row("width_treble", "Stereo width — treble band", space_section.body, 5, 0.5, 3.0)
        ttk.Label(
            space_section.body,
            text="Stereo width, warmth, and clarity effects on the final master. Crossfeed "
                 "is meant for headphone listening and will sound odd on speakers. Width "
                 "sliders only take effect when \"Per-band stereo width\" above is on.",
            foreground="#888888", font=("TkDefaultFont", 8), wraplength=520, justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 0))

        # --- 5.1 speaker mapping (only meaningful for a 5.1 device
        # profile's surround output; hidden/shown by refresh_profile_label()) ---
        self.speaker_bias_vars: dict[str, tuple] = {}
        self.speaker_bias_frame = ttk.Frame(space_section.body)
        self.speaker_bias_frame.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Label(
            self.speaker_bias_frame, text="5.1 Speaker Mapping",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=6, pady=(4, 0))
        ttk.Label(
            self.speaker_bias_frame,
            text="Blend extra content into a speaker group on top of its normal mix. The "
                 "dB slider sets the blend's loudness relative to that speaker's own level "
                 "(0dB = as loud as the channel already is). Only affects the 5.1 MKV output.",
            foreground="#888888", font=("TkDefaultFont", 8), wraplength=520, justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 6))
        self._speaker_bias_row("front", "Front L/R", self.speaker_bias_frame, 2)
        self._speaker_bias_row("centre", "Centre", self.speaker_bias_frame, 3)
        self._speaker_bias_row("rear", "Rear L/R", self.speaker_bias_frame, 4)

        # --- Restoration section ---
        rest_section = CollapsibleSection(body, "Restoration", expanded=True)
        rest_section.grid(row=4, column=0, sticky="ew")
        ttk.Label(
            rest_section.body,
            text="Cleans up the source before any EQ/dynamics are applied. Declick removes "
                 "pops/clicks; Denoise reduces hiss (aggressive settings can also dull real "
                 "high-frequency detail — use \"Analyze Track\" above for a level tuned to "
                 "this specific recording instead of guessing).",
            foreground="#888888", font=("TkDefaultFont", 8), wraplength=520, justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 0))
        self._bool_row("declick", "Declick / declip", rest_section.body, 0)
        self._bool_row("denoise", "Denoise", rest_section.body, 1)
        self._float_row("denoise_amount", "Denoise amount (afftdn nr, 0-20)", rest_section.body, 2, 0, 20)
        self.trim_var = tk.BooleanVar(value=state.trim_silence)
        ttk.Checkbutton(rest_section.body, text="Trim silence", variable=self.trim_var).grid(row=3, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(rest_section.body, text="Fade in (ms):").grid(row=4, column=0, sticky="w", padx=6)
        self.fade_in_var = tk.IntVar(value=state.fade_in_ms)
        ttk.Spinbox(rest_section.body, textvariable=self.fade_in_var, from_=0, to=10000, width=8).grid(row=4, column=1, sticky="w")
        ttk.Label(rest_section.body, text="Fade out (ms):").grid(row=5, column=0, sticky="w", padx=6)
        self.fade_out_var = tk.IntVar(value=state.fade_out_ms)
        ttk.Spinbox(rest_section.body, textvariable=self.fade_out_var, from_=0, to=10000, width=8).grid(row=5, column=1, sticky="w")

        # --- Analyze Track button group (sits directly above the Preview
        # pane below; bordered like the Preview pane, items left-aligned) ---
        title_row = ttk.LabelFrame(self, text="Analyze Track")
        title_row.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(title_row, text="🔍 Analyze Track", command=self._analyze_track).pack(
            side="left", padx=(6, 4), pady=(6, 6)
        )
        self.thorough_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            title_row, text="Thorough (full track + Demucs, slower)", variable=self.thorough_var,
        ).pack(side="left", padx=(4, 8), pady=(6, 6))
        self.analysis_info_btn = ttk.Button(
            title_row, text="📋 Analysis Info", command=self._show_analysis_info, state="disabled",
        )
        self.analysis_info_btn.pack(side="left", padx=(0, 4), pady=(6, 6))
        self.apply_ideal_btn = ttk.Button(
            title_row, text="Apply Ideal Values", command=self._apply_ideal_values, state="disabled",
        )
        self.apply_ideal_btn.pack(side="left", padx=(0, 4), pady=(6, 6))
        ttk.Button(title_row, text="↺ Reset to Defaults", command=self._reset_to_defaults).pack(
            side="left", padx=(0, 4), pady=(6, 6)
        )

        # --- Preview controls ---
        preview_frame = ttk.LabelFrame(self, text="Preview")
        preview_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        preview_frame.columnconfigure(3, weight=1)
        self.preview_status = ttk.Label(preview_frame, text="Not previewed yet.")
        self.preview_status.grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        self.preview_progress = ttk.Progressbar(preview_frame, mode="indeterminate", length=140)
        self.preview_progress.grid(row=0, column=3, sticky="e", padx=6, pady=4)
        self.preview_progress.grid_remove()  # hidden until a processing job starts
        self.play_original_btn = ttk.Button(preview_frame, text="▶ Play Original", command=self._preview_original)
        self.play_original_btn.grid(row=1, column=0, padx=6, pady=6)
        self.play_processed_btn = ttk.Button(preview_frame, text="▶ Play Processed", command=self._preview_processed)
        self.play_processed_btn.grid(row=1, column=1, padx=6, pady=6)
        self.stop_btn = ttk.Button(preview_frame, text="■ Stop", command=self._preview_stop)
        self.stop_btn.grid(row=1, column=2, padx=6, pady=6)

        vol_frame = ttk.Frame(preview_frame)
        vol_frame.grid(row=1, column=3, sticky="e", padx=6)
        ttk.Label(vol_frame, text="🔊").pack(side="left", padx=(0, 4))
        self.volume_var = tk.DoubleVar(value=100.0)
        ttk.Scale(
            vol_frame, from_=0, to=150, variable=self.volume_var, orient="horizontal",
            length=110, command=lambda _v: self._update_volume_label(),
        ).pack(side="left")
        self.volume_label = ttk.Label(vol_frame, text="100%", width=5)
        self.volume_label.pack(side="left", padx=(4, 0))

        ttk.Label(
            preview_frame,
            text="Preview applies the full EQ/dynamics/space chain, but skips AI stem "
                 "separation (vocal/instrument isolation) for speed — Vocal Chain sliders "
                 "and per-stem enhancement only apply to the final export in Step 6.",
            foreground="#888888", font=("TkDefaultFont", 8), wraplength=520, justify="left",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 4))

        self._update_eq_canvas()

    def _eq_slider(self, key, label, parent, row, lo, hi):
        default = self.state.params.get(key, 0.0)
        var = tk.DoubleVar(value=default)
        self.vars[key] = var
        ttk.Label(parent, text=label, width=36, anchor="w").grid(row=row, column=0, sticky="w", padx=6)
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal", length=130,
                          command=lambda _v: self._update_eq_canvas())
        scale.grid(row=row, column=1, sticky="w", padx=6)
        val_label = ttk.Label(parent, text=f"{default:+.1f}", width=6)
        val_label.grid(row=row, column=2)
        self._add_ideal_label(key, parent, row, column=3)

        # Slider drags produce raw floats like 3.478291023841 -- show a
        # clean, fixed-precision value instead of that directly, updated
        # on both drag and programmatic changes (reset, profile reload).
        def _sync_label(*_args, _var=var, _lbl=val_label):
            try:
                _lbl.configure(text=f"{_var.get():+.1f}")
            except tk.TclError:
                pass
        var.trace_add("write", _sync_label)

    def _bool_row(self, key, label, parent, row):
        default = self.state.params.get(key, False)
        var = tk.BooleanVar(value=default)
        self.vars[key] = var
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        self._add_ideal_label(key, parent, row, column=1)

    def _add_ideal_label(self, key: str, parent, row: int, column: int):
        """Small 'Ideal: —' label shown next to a control, filled in by
        _analyze_track() and left blank until analysis has run. Kept in
        self.ideal_labels so _analyze_track()/_apply_ideal_values() can
        find it by parameter key regardless of which control type it's
        attached to (slider, checkbox, spinbox)."""
        lbl = tk.Label(parent, text="", font=("TkDefaultFont", 8), fg="#1a5fb4")
        lbl.grid(row=row, column=column, sticky="w", padx=(8, 0))
        self.ideal_labels[key] = lbl

    def _speaker_bias_row(self, group_key: str, label: str, parent, row: int):
        """One speaker-mapping row: a source dropdown (Off/Vocals/Band1-5)
        and a loudness bias slider (-24dB..+6dB), for one 5.1 speaker group."""
        label_by_key = dict(SPEAKER_BIAS_SOURCE_LABELS)
        key_by_label = {lbl: k for k, lbl in SPEAKER_BIAS_SOURCE_LABELS}
        display_values = [lbl for _, lbl in SPEAKER_BIAS_SOURCE_LABELS]

        current_key = self.state.params.get(f"speaker_bias_{group_key}", "off")
        current_db = self.state.params.get(f"speaker_bias_{group_key}_db", 0.0)

        ttk.Label(parent, text=label, width=12, anchor="w").grid(row=row, column=0, sticky="w", padx=6, pady=3)

        source_var = tk.StringVar(value=label_by_key.get(current_key, "Off"))
        ttk.Combobox(
            parent, textvariable=source_var, state="readonly", values=display_values, width=24,
        ).grid(row=row, column=1, sticky="w", padx=(0, 10))

        db_var = tk.DoubleVar(value=current_db)
        ttk.Scale(
            parent, from_=-24, to=6, variable=db_var, orient="horizontal", length=120,
        ).grid(row=row, column=2, sticky="w", padx=(0, 6))
        db_label = ttk.Label(parent, text=f"{current_db:+.1f} dB", width=8)
        db_label.grid(row=row, column=3, sticky="w")

        def _sync_db_label(*_args, _var=db_var, _lbl=db_label):
            try:
                _lbl.configure(text=f"{_var.get():+.1f} dB")
            except tk.TclError:
                pass
        db_var.trace_add("write", _sync_db_label)

        self.speaker_bias_vars[group_key] = (source_var, db_var, key_by_label)

    def _float_row(self, key, label, parent, row, lo, hi):
        default = self.state.params.get(key, (lo + hi) / 2)
        var = tk.DoubleVar(value=default)
        self.vars[key] = var
        ttk.Label(parent, text=label, width=36, anchor="w").grid(row=row, column=0, sticky="w", padx=6)
        ttk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=0.5, width=8).grid(row=row, column=1, sticky="w")
        self._add_ideal_label(key, parent, row, column=2)

    def _update_eq_canvas(self):
        preview_params = {
            "bass_shelf_hz": self.state.params.get("bass_shelf_hz", 100),
            "bass_shelf_db": self.vars["bass_shelf_db"].get(),
            "treble_shelf_hz": self.state.params.get("treble_shelf_hz", 8000),
            "treble_shelf_db": self.vars["treble_shelf_db"].get(),
            "presence_hz": self.state.params.get("presence_hz", 2800),
            "presence_db": self.vars["presence_db"].get(),
            "presence_q": self.state.params.get("presence_q", 1.2),
            "notch_hz": self.state.params.get("notch_hz", 450),
            "notch_db": self.vars["notch_db"].get(),
            "notch_q": self.state.params.get("notch_q", 2.0),
        }
        self.eq_canvas.redraw(preview_params)

    def _resolved_content_key(self) -> str:
        content_key = self.state.content_key or self.state.content_auto_detected or "?"
        return content_key if content_key in CONTENT_PROFILES else "early_digital"

    def refresh_profile_label(self):
        device_name = DEVICE_PROFILES[self.state.device_key].display_name
        content_key = self._resolved_content_key()
        content_name = CONTENT_PROFILES[content_key].display_name
        self.profile_label.configure(text=f"Device: {device_name}  |  Content: {content_name}")

        resolved = resolve_profile(self.state.device_key, content_key)
        profile_signature = (self.state.device_key, content_key)
        profile_changed = getattr(self, "_last_profile_signature", None) != profile_signature
        self._last_profile_signature = profile_signature

        if profile_changed:
            # A different device/content selection than last time Step 5
            # was shown (e.g. user went back and picked a different
            # device or content type) -- start from this profile's actual
            # defaults, since prior slider tweaks belonged to a different
            # profile and wouldn't necessarily make sense here.
            self.state.params = dict(resolved)
            for key, var in self.vars.items():
                if key in resolved:
                    var.set(resolved[key])
            self._reset_speaker_bias_vars()
        else:
            # Same profile as last time -- preserve whatever the user had
            # on the sliders (already saved into state.params by commit()
            # when they left this step) instead of clobbering it back to
            # the profile defaults on every revisit.
            self.state.params = {**resolved, **self.state.params}
            for key, var in self.vars.items():
                if key in self.state.params:
                    var.set(self.state.params[key])
            self._sync_speaker_bias_vars()

        # Speaker mapping only affects 5.1 surround output -- hide it
        # entirely for every other target device, regardless of which
        # 5.1-layout device profile is selected.
        if resolved.get("layout") == "5.1":
            self.speaker_bias_frame.grid()
        else:
            self.speaker_bias_frame.grid_remove()

        self._update_stems_status(content_key, resolved)
        self._update_eq_canvas()

    def _reset_speaker_bias_vars(self):
        """Reset the speaker-mapping UI (and state.params) to 'Off' / 0dB —
        used when switching to a different device/content profile."""
        label_off = dict(SPEAKER_BIAS_SOURCE_LABELS)["off"]
        for group_key, (source_var, db_var, _key_by_label) in self.speaker_bias_vars.items():
            source_var.set(label_off)
            db_var.set(0.0)
            self.state.params[f"speaker_bias_{group_key}"] = "off"
            self.state.params[f"speaker_bias_{group_key}_db"] = 0.0

    def _sync_speaker_bias_vars(self):
        """Reflect any speaker-mapping values already in state.params onto
        the UI controls — used when revisiting Step 5 on the same profile."""
        label_by_key = dict(SPEAKER_BIAS_SOURCE_LABELS)
        for group_key, (source_var, db_var, _key_by_label) in self.speaker_bias_vars.items():
            key = self.state.params.get(f"speaker_bias_{group_key}", "off")
            db = self.state.params.get(f"speaker_bias_{group_key}_db", 0.0)
            source_var.set(label_by_key.get(key, "Off"))
            db_var.set(db)

    def _update_stems_status(self, content_key: str, resolved: dict):
        content_name = CONTENT_PROFILES[content_key].display_name
        if not resolved.get("use_stems", False):
            self.vocal_stems_status.configure(
                text=f"✘ Not active: the \"{content_name}\" content profile never uses stem "
                     f"separation, so Vocal Chain controls below have no effect. Pick a "
                     f"different content type in Step 3 to use them.",
                foreground="#c01c28",
            )
        elif self.state.demucs_model == "none":
            self.vocal_stems_status.configure(
                text="✘ Not active: stem separation model is set to \"none\" in Step 4. "
                     "Choose htdemucs or htdemucs_ft there to enable Vocal Chain controls.",
                foreground="#c01c28",
            )
        else:
            self.vocal_stems_status.configure(
                text=f"✔ Active for \"{content_name}\" — but only in the final export "
                     f"(Step 6). The quick Preview below always skips stem separation for speed.",
                foreground="#1b7a43",
            )

    def _reset_to_defaults(self):
        """Reset every Step 5 control back to the current device+content
        profile's defaults, discarding any manual slider/toggle overrides."""
        resolved = resolve_profile(self.state.device_key, self._resolved_content_key())
        for key, var in self.vars.items():
            if key in resolved:
                var.set(resolved[key])
        self.trim_var.set(False)
        self.fade_in_var.set(0)
        self.fade_out_var.set(0)
        self.state.params = dict(resolved)  # drop prior overrides entirely
        self._reset_speaker_bias_vars()
        self._update_eq_canvas()
        self.preview_status.configure(text="Reset to profile defaults.")

    def _analyze_track(self):
        """Analyze the input file and populate the "Ideal: X" labels next
        to each affected control — mirroring this project's earlier
        CLI+GUI tool (analyze_track()/compute_recommendations() + the
        "Apply Ideal Values" button). Analysis only fills in the
        ideal-value labels; nothing is applied to the actual sliders
        until the user reviews the recommendations and clicks "Apply
        Ideal Values" — same two-step review flow as before, rather than
        silently overwriting the sliders the moment analysis finishes.

        The "Thorough" checkbox trades speed for depth: unchecked, this
        analyzes at most the first 120 seconds and only runs Demucs
        vocal/instrument separation if a stem model happens to be
        selected in Step 4 already — fast, seconds at most. Checked, it
        analyzes the FULL track, always attempts Demucs (if installed)
        regardless of the Step 4 selection, and computes sturdier
        band/loudness measurements (windowed medians and an approximate
        loudness range instead of one aggregate number) — this can take
        anywhere from tens of seconds to a few minutes depending on track
        length and whether Demucs runs.

        This is separate from Step 3's content-type auto-detection: that
        picks a discrete era/style BUCKET (e.g. "Cassette Era") whose
        defaults are shared across every recording in that bucket. This
        looks at the specific file's own measured levels/tonal balance and
        recommends values tuned to that one file.
        """
        if not self.state.input_path:
            self.analysis_status.configure(text="No input file selected.")
            return
        self.commit()
        thorough = self.thorough_var.get()
        busy_text = (
            "Analyzing full track — this can take a while (Demucs stem separation)…"
            if thorough else "Analyzing track…"
        )
        self._set_preview_busy(True, busy_text)
        input_path = self.state.input_path
        # Reuse whatever Step 4 chose for stem separation — matches the
        # earlier tool's behaviour of reusing its "Use True Demucs
        # Separation" checkbox for analysis rather than a separate toggle.
        # Thorough mode always attempts Demucs (if installed) regardless
        # of the Step 4 selection, since that's the biggest lever for a
        # more complete analysis and worth running independent of the
        # export choice.
        want_demucs = thorough or self.state.demucs_model not in ("", "none")
        demucs_model = self.state.demucs_model if self.state.demucs_model not in ("", "none") else "htdemucs"
        music_lufs_ref = float(self.state.params.get("music_lufs", -18.0))

        def _work():
            try:
                files = core.collect_audio_files(Path(input_path))
                if not files:
                    self.after(0, lambda: self._set_preview_busy(False, "No audio files found to analyze."))
                    return
                if not core.CAP.numpy:
                    self.after(0, lambda: self._set_preview_busy(
                        False, "Analysis needs numpy, which isn't installed."))
                    return
                first_file = files[0]
                seg = core.load_audio_file(first_file)
                if thorough:
                    # Full track, no cap — this is the whole point of
                    # thorough mode: give the analysis every second of
                    # audio instead of just the first couple of minutes.
                    seg_for_measurement = seg
                else:
                    # Cap the analysis window so a very long track doesn't
                    # make this feel slow — a couple of minutes is plenty
                    # of signal for level/tonal-balance estimates.
                    max_ms = 120_000
                    seg_for_measurement = seg[:max_ms] if len(seg) > max_ms else seg
                audio, sr = core.pydub_to_float32(seg_for_measurement)
                result = core.analyze_track_for_optimal_params(
                    audio, sr,
                    input_path=first_file,
                    use_demucs=want_demucs,
                    demucs_model=demucs_model,
                    music_lufs_reference=music_lufs_ref,
                    elaborate=thorough,
                )

                def _finish():
                    # Analysis reused the Preview section's busy indicator
                    # (progress bar + disabled buttons) while it ran — that
                    # left the Preview status label stuck on "Analyzing
                    # full track…" forever, since the plain False-only call
                    # here never supplied replacement text. Explicitly hand
                    # it back to a neutral preview message so it doesn't
                    # look like analysis is still (or always) running.
                    self._set_preview_busy(False, "Not previewed yet.")
                    self.last_recommendations = result.get("recommendations", {})
                    self.last_measurements = result.get("measurements", {})
                    self._update_ideal_labels()
                    self.apply_ideal_btn.configure(
                        state="normal" if self.last_recommendations else "disabled"
                    )
                    self.analysis_info_btn.configure(
                        state="normal" if (self.last_recommendations or self.last_measurements)
                        else "disabled"
                    )
                    m = dict(self.last_measurements)
                    # vocal_analysis_status is pulled out and shown separately
                    # (rather than lost in the generic k=v dump) since it's
                    # the one thing that explains why analysis might have
                    # finished unusually fast: the Demucs vocal/instrument
                    # step is by far the slowest part when it runs, and
                    # silently doesn't when it can't.
                    vocal_status = m.pop("vocal_analysis_status", None)
                    parts = [f"{k}={v}" for k, v in m.items()]
                    status_line = "Analysis complete — " + ", ".join(parts) + "."
                    if vocal_status and vocal_status != "ok":
                        status_line += f"  Vocal/instrument analysis: {vocal_status}."
                    status_line += (
                        "  Review the \"Ideal\" values below, then click "
                        "\"Apply Ideal Values\" to use them."
                    )
                    self.analysis_status.configure(text=status_line)
                self.after(0, _finish)
            except Exception as exc:
                self.after(0, lambda: self._set_preview_busy(False, f"Analysis failed: {exc}"))

        threading.Thread(target=_work, daemon=True).start()

    def _update_ideal_labels(self):
        """Refresh every 'Ideal: X' label from self.last_recommendations."""
        for key, lbl in self.ideal_labels.items():
            if key not in self.last_recommendations:
                lbl.configure(text="")
                continue
            value = self.last_recommendations[key]
            if isinstance(value, bool):
                lbl.configure(text=f"Ideal: {'On' if value else 'Off'}")
            elif isinstance(value, float):
                lbl.configure(text=f"Ideal: {value:+.1f}")
            else:
                lbl.configure(text=f"Ideal: {value}")

    def _show_analysis_info(self):
        """Show the full analysis log (every measurement and every
        recommendation, not just the one-line summary above the sliders)
        in a pretty-printed, read-only popup. Disabled until an analysis
        has actually produced something to show."""
        if not self.last_measurements and not self.last_recommendations:
            return

        def _fmt_value(v):
            if isinstance(v, bool):
                return "On" if v else "Off"
            if isinstance(v, float):
                return f"{v:+.2f}"
            return str(v)

        def _section(title: str, data: dict) -> list[str]:
            lines = [title, "-" * len(title)]
            if not data:
                lines.append("  (none)")
            else:
                key_width = max(len(k) for k in data)
                for k, v in data.items():
                    lines.append(f"  {k.ljust(key_width)} : {_fmt_value(v)}")
            return lines

        lines: list[str] = []
        lines += _section("Measurements", self.last_measurements)
        lines.append("")
        lines += _section("Recommendations (\"Ideal\" values)", self.last_recommendations)

        win = tk.Toplevel(self)
        win.title("Analysis Info")
        win.geometry("560x480")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        text_frame = ttk.Frame(win, padding=8)
        text_frame.grid(row=0, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        txt = tk.Text(text_frame, wrap="word", font=("TkFixedFont", 10))
        txt.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=txt.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        txt.configure(yscrollcommand=scroll.set)

        txt.insert("1.0", "\n".join(lines))
        txt.configure(state="disabled")

        ttk.Button(win, text="Close", command=win.destroy).grid(row=1, column=0, pady=(0, 8))

    def _apply_ideal_values(self):
        """Copy the last analysis's recommendations onto the actual
        controls. Disabled until an analysis has produced at least one
        recommendation. Applied values become ordinary slider overrides —
        "Reset to Defaults" discards them same as any manual tweak."""
        if not self.last_recommendations:
            return
        for key, value in self.last_recommendations.items():
            if key in self.vars:
                self.vars[key].set(value)
            self.state.params[key] = value
        self._update_eq_canvas()
        self.analysis_status.configure(text="Ideal values applied to the controls below.")

    def commit(self):
        for key, var in self.vars.items():
            self.state.params[key] = var.get()
        self.state.trim_silence = self.trim_var.get()
        self.state.fade_in_ms = self.fade_in_var.get()
        self.state.fade_out_ms = self.fade_out_var.get()
        for group_key, (source_var, db_var, key_by_label) in self.speaker_bias_vars.items():
            self.state.params[f"speaker_bias_{group_key}"] = key_by_label.get(source_var.get(), "off")
            self.state.params[f"speaker_bias_{group_key}_db"] = db_var.get()

    def validate(self) -> str | None:
        return None

    def _get_preview_clip(self):
        files = core.collect_audio_files(Path(self.state.input_path))
        if not files:
            return None, None
        seg = core.load_audio_file(files[0])
        start_ms = int(self.state.preview_start_s * 1000)
        dur_ms = int(self.state.preview_duration_s * 1000)
        clip = seg[start_ms: start_ms + dur_ms]
        if not core.CAP.numpy:
            return None, None
        audio, sr = core.pydub_to_float32(clip)
        return audio, sr

    def _set_preview_busy(self, busy: bool, status_text: str | None = None):
        """Toggle the loading indicator + preview buttons. Safe to call
        from the main thread only — background threads must marshal
        through self.after(0, ...)."""
        if status_text is not None:
            self.preview_status.configure(text=status_text)
        if busy:
            self.preview_progress.grid()
            self.preview_progress.start(12)
            self.play_original_btn.configure(state="disabled")
            self.play_processed_btn.configure(state="disabled")
        else:
            self.preview_progress.stop()
            self.preview_progress.grid_remove()
            self.play_original_btn.configure(state="normal")
            self.play_processed_btn.configure(state="normal")

    def _update_volume_label(self):
        self.volume_label.configure(text=f"{int(round(self.volume_var.get()))}%")

    def _apply_volume(self, audio):
        """Scale a float32 preview clip by the volume slider, clipped to
        [-1, 1] so values above 100% don't hard-clip in the audio driver."""
        gain = max(0.0, self.volume_var.get()) / 100.0
        if gain == 1.0:
            return audio
        return (audio * gain).clip(-1.0, 1.0)

    def _preview_original(self):
        self.commit()
        audio, sr = self._get_preview_clip()
        if audio is None:
            self.preview_status.configure(text="No input file selected.")
            return
        audio = self._apply_volume(audio)
        self.preview_status.configure(text="Playing original…")
        core.play_audio_preview(
            audio, sr,
            on_finish=lambda: self.preview_status.configure(text="Original playback finished."),
            on_error=lambda exc: self.preview_status.configure(text=f"Playback failed: {exc}"),
        )

    def _preview_processed(self):
        self.commit()
        audio, sr = self._get_preview_clip()
        if audio is None:
            self.preview_status.configure(text="No input file selected.")
            return
        self._set_preview_busy(True, "Processing preview…")

        def _process_and_play():
            try:
                params = self.state.resolve()
                params["use_stems"] = False  # skip slow Demucs for quick preview
                params["use_loudnorm"] = False  # peak-normalise instead (pyloudnorm unreliable <10s here anyway for very short clips)
                # NOTE: this passes the FULL resolved parameter set (not a
                # subset) so the preview actually reflects the tuned system/
                # content profile — crossover frequencies, per-band
                # compression ratios, per-band width, presence/notch Q, and
                # crossfeed strength/range all matter and previously fell
                # back to apply_mastering_chain's generic defaults instead
                # of the profile's real values, making the preview sound
                # less enhanced than the actual export.
                processed = core.apply_mastering_chain(
                    audio, sr,
                    bass_shelf_hz=params.get("bass_shelf_hz", 100), bass_shelf_db=params.get("bass_shelf_db", 0),
                    treble_shelf_hz=params.get("treble_shelf_hz", 8000), treble_shelf_db=params.get("treble_shelf_db", 0),
                    presence_hz=params.get("presence_hz", 0), presence_db=params.get("presence_db", 0),
                    presence_q=params.get("presence_q", 1.2),
                    notch_hz=params.get("notch_hz", 0), notch_db=params.get("notch_db", 0),
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

                def _on_ready():
                    if not self._active:
                        # User already navigated away — don't start playback
                        # on a step they're no longer viewing.
                        self._set_preview_busy(False)
                        return
                    self._set_preview_busy(False, "Playing processed…")
                    playback_audio = self._apply_volume(processed)
                    core.play_audio_preview(
                        playback_audio, sr,
                        on_finish=lambda: self.preview_status.configure(text="Processed playback finished."),
                        on_error=lambda exc: self.preview_status.configure(text=f"Playback failed: {exc}"),
                    )

                self.after(0, _on_ready)
            except Exception as exc:
                self.after(0, lambda: self._set_preview_busy(False, f"Preview failed: {exc}"))

        threading.Thread(target=_process_and_play, daemon=True).start()

    def _preview_stop(self):
        core.stop_audio_preview()
        self._set_preview_busy(False, "Stopped.")

    def on_leave(self):
        """Called by the wizard controller when navigating away from this
        step. Stops any in-progress playback (and prevents a still-running
        background processing job from starting playback afterwards)."""
        self._active = False
        core.stop_audio_preview()
        self._set_preview_busy(False)

    def on_enter(self):
        """Called by the wizard controller when this step becomes visible."""
        self._active = True


def _fmt_mmss(seconds: float) -> str:
    """Format a duration in seconds as M:SS for the player's time label."""
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class AudioPlayerPanel(ttk.LabelFrame):
    """Play / pause / stop / seek preview for an audio file — used for
    both the original input ("Play Original") and the finished output
    ("Play Converted") on Step 6.

    Streams the loaded track through sounddevice (via engine's `core.sd`,
    the same backend `play_audio_preview` uses) with a manually-tracked
    sample position so pausing and dragging the seek bar to a new spot
    both work — `core.play_audio_preview` only supports uninterrupted
    play/stop, not pause or mid-playback seeking.
    """

    def __init__(self, parent, title: str = "Play Converted",
                 empty_status_text: str = "Run a conversion to preview the output here."):
        super().__init__(parent, text=title)
        self._empty_status_text = empty_status_text
        self.columnconfigure(3, weight=1)

        self.audio: "np.ndarray | None" = None
        self.sr: int = 44100
        self.duration_s: float = 0.0
        self.pos: int = 0             # current playhead, in samples
        self.playing: bool = False
        self._stream = None
        self._seeking = False
        self._poll_id = None

        self.play_btn = ttk.Button(self, text="▶ Play", command=self._play, state="disabled")
        self.play_btn.grid(row=0, column=0, padx=(6, 4), pady=6)
        self.pause_btn = ttk.Button(self, text="⏸ Pause", command=self._pause, state="disabled")
        self.pause_btn.grid(row=0, column=1, padx=4, pady=6)
        self.stop_btn = ttk.Button(self, text="■ Stop", command=self._stop, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=4, pady=6)

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = ttk.Scale(
            self, from_=0, to=100, variable=self.seek_var, orient="horizontal", state="disabled",
        )
        self.seek_scale.grid(row=0, column=3, sticky="ew", padx=6, pady=6)
        self.seek_scale.bind("<ButtonPress-1>", self._seek_start)
        self.seek_scale.bind("<ButtonRelease-1>", self._seek_end)

        self.time_label = ttk.Label(self, text="0:00 / 0:00", width=12)
        self.time_label.grid(row=0, column=4, padx=(0, 6))

        self.status_label = ttk.Label(
            self, text=empty_status_text,
            foreground="#888888", font=("TkDefaultFont", 8),
        )
        self.status_label.grid(row=1, column=0, columnspan=5, sticky="w", padx=6, pady=(0, 4))

    # -- loading -------------------------------------------------------

    def load(self, path: Path):
        """Load a finished output file for preview playback."""
        self._stop()
        self.audio = None
        self.duration_s = 0.0
        self.pos = 0
        self.seek_var.set(0.0)
        self._update_time_label()

        if core is None or not core.CAP.numpy:
            self.status_label.configure(text="Preview unavailable (numpy not installed).")
            self._set_controls_enabled(False)
            return
        try:
            seg = core.load_audio_file(path)
            audio, sr = core.pydub_to_float32(seg)
        except Exception as exc:
            self.status_label.configure(text=f"Could not load output for preview: {exc}")
            self._set_controls_enabled(False)
            return

        self.audio = audio
        self.sr = sr
        self.duration_s = (len(audio) / sr) if sr else 0.0
        self.seek_scale.configure(to=max(self.duration_s, 0.1))
        self._update_time_label()

        if not core.CAP.sounddevice:
            self.status_label.configure(
                text=f"Loaded {path.name} — install 'sounddevice' to enable preview playback."
            )
            self._set_controls_enabled(False)
            return

        self.status_label.configure(text=f"Previewing: {path.name}")
        self._set_controls_enabled(True)

    def clear(self):
        """Reset to the empty/no-output state (e.g. before a new run starts)."""
        self._stop()
        self.audio = None
        self.duration_s = 0.0
        self.pos = 0
        self.seek_var.set(0.0)
        self.seek_scale.configure(to=100)
        self._update_time_label()
        self.status_label.configure(text=self._empty_status_text)
        self._set_controls_enabled(False)

    def _set_controls_enabled(self, enabled: bool):
        self.play_btn.configure(state="normal" if enabled else "disabled")
        self.pause_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        self.seek_scale.configure(state="normal" if enabled else "disabled")

    # -- transport -------------------------------------------------------

    def _play(self):
        if self.audio is None or core is None or not core.CAP.sounddevice:
            return
        if self.playing:
            return
        if self.pos >= len(self.audio):
            self.pos = 0
        self._start_stream()
        self.playing = True
        self.play_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal")
        self.stop_btn.configure(state="normal")
        self._poll()

    def _pause(self):
        if not self.playing:
            return
        self._stop_stream()
        self.playing = False
        self.play_btn.configure(state="normal")
        self.pause_btn.configure(state="disabled")
        if self._poll_id:
            self.after_cancel(self._poll_id)
            self._poll_id = None

    def _stop(self):
        self._stop_stream()
        self.playing = False
        self.pos = 0
        self.seek_var.set(0.0)
        self._update_time_label()
        has_audio = self.audio is not None and core is not None and core.CAP.sounddevice
        self.play_btn.configure(state="normal" if has_audio else "disabled")
        self.pause_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        if self._poll_id:
            self.after_cancel(self._poll_id)
            self._poll_id = None

    def _start_stream(self):
        sd = core.sd
        audio = self.audio
        channels = audio.shape[1] if audio.ndim == 2 else 1

        def callback(outdata, frames, time_info, status):
            start = self.pos
            end = min(start + frames, len(audio))
            chunk = audio[start:end]
            if audio.ndim == 1:
                chunk = chunk.reshape(-1, 1)
            if len(chunk) < frames:
                outdata[:len(chunk)] = chunk
                outdata[len(chunk):] = 0
                self.pos = len(audio)
                raise sd.CallbackStop()
            outdata[:] = chunk
            self.pos = end

        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=channels, callback=callback, dtype="float32",
            finished_callback=self._on_stream_finished,
        )
        self._stream.start()

    def _on_stream_finished(self):
        # Fires on a sounddevice-managed thread — marshal back to the Tk
        # main thread before touching any widgets.
        self.after(0, self._handle_stream_finished)

    def _handle_stream_finished(self):
        if self.playing and self.audio is not None and self.pos >= len(self.audio):
            self._stop()

    def _stop_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # -- seeking -------------------------------------------------------

    def _seek_start(self, _event):
        self._seeking = True

    def _seek_end(self, _event):
        self._seeking = False
        if self.audio is None:
            return
        target_s = self.seek_var.get()
        self.pos = max(0, min(int(target_s * self.sr), len(self.audio)))
        self._update_time_label()
        if self.playing:
            self._stop_stream()
            self._start_stream()

    # -- polling / display -------------------------------------------------------

    def _poll(self):
        if self.playing and not self._seeking:
            self.seek_var.set(self.pos / self.sr if self.sr else 0.0)
            self._update_time_label()
        if self.playing:
            self._poll_id = self.after(200, self._poll)

    def _update_time_label(self):
        cur = self.pos / self.sr if self.sr else 0.0
        self.time_label.configure(text=f"{_fmt_mmss(cur)} / {_fmt_mmss(self.duration_s)}")

    def on_leave(self):
        """Stop any playback in progress (called when navigating away)."""
        self._stop()


class _QueueLogHandler(logging.Handler):
    """Logging handler that forwards formatted INFO+ log records into a
    wizard step's message queue, the same way `_log()` already forwards
    hand-written progress/status lines from run_batch()/remaster_file().

    Without this, only those explicit log_callback() lines show up in the
    GUI's "Progress & Log" text area — every other log.info()/warning()
    call scattered through engine.py (e.g. "Stereo output: …", "MKV
    muxed: …") would only ever be visible on the CLI console or in the
    rotating log file, never in the GUI.
    """

    def __init__(self, msg_queue: "queue.Queue", level: int = logging.INFO):
        super().__init__(level=level)
        self.msg_queue = msg_queue
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.msg_queue.put_nowait(("log", self.format(record)))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Step 6 — Output & Run
# ---------------------------------------------------------------------------

class Step6Run(ttk.Frame):
    def __init__(self, parent, state: WizardState, on_run_complete=None, on_back=None):
        super().__init__(parent)
        self.state = state
        self.on_run_complete = on_run_complete
        self.on_back = on_back
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.msg_queue: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False

        # --- Selections summary: every choice made in Steps 1-5, as
        # scrollable key/value rows (change #2). ---
        summary_frame = ttk.LabelFrame(self, text="Your Selections")
        summary_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        summary_frame.columnconfigure(0, weight=1)

        summary_canvas = tk.Canvas(summary_frame, height=220, highlightthickness=0, bg="white")
        summary_scroll = ttk.Scrollbar(summary_frame, orient="vertical", command=summary_canvas.yview)
        self.summary_inner = tk.Frame(summary_canvas, bg="white")
        self.summary_inner.bind(
            "<Configure>", lambda e: summary_canvas.configure(scrollregion=summary_canvas.bbox("all"))
        )
        summary_window = summary_canvas.create_window((0, 0), window=self.summary_inner, anchor="nw")
        summary_canvas.configure(yscrollcommand=summary_scroll.set)

        def _resize_summary(event, _c=summary_canvas, _w=summary_window):
            _c.itemconfig(_w, width=event.width)
        summary_canvas.bind("<Configure>", _resize_summary)

        def _summary_wheel(event, _c=summary_canvas):
            _c.yview_scroll(int(-1 * (event.delta / 120)), "units")
        summary_canvas.bind("<Enter>", lambda e: summary_canvas.bind_all("<MouseWheel>", _summary_wheel))
        summary_canvas.bind("<Leave>", lambda e: summary_canvas.unbind_all("<MouseWheel>"))

        summary_canvas.grid(row=0, column=0, sticky="nsew")
        summary_scroll.grid(row=0, column=1, sticky="ns")

        self.surround_frame = ttk.LabelFrame(self, text="Surround Output (5.1)")
        self.surround_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(self.surround_frame, text="Codec:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.codec_var = tk.StringVar(value=state.surround_codec)
        ttk.Combobox(self.surround_frame, textvariable=self.codec_var, state="readonly",
                    values=["ac3", "pcm"], width=8).grid(row=0, column=1, padx=6)
        ttk.Label(self.surround_frame, text="LFE mode:").grid(row=0, column=2, sticky="w", padx=(16, 6))
        self.lfe_var = tk.StringVar(value=state.lfe_mode)
        ttk.Combobox(self.surround_frame, textvariable=self.lfe_var, state="readonly",
                    values=["silent", "gentle", "full"], width=8).grid(row=0, column=3, padx=6)

        log_frame = ttk.LabelFrame(self, text="Progress & Log")
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)

        self.progress = ttk.Progressbar(log_frame, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        # height reduced to 75% of the original 10 rows
        self.log_text = tk.Text(log_frame, height=7, wrap="word", state="disabled")
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        # Play Original — sits above Play Converted, populated whenever the
        # step is entered with an input file selected (see on_enter() /
        # _load_original_preview() below).
        self.original_player = AudioPlayerPanel(
            self, title="Play Original",
            empty_status_text="Select an input file in Step 1 to preview the original here.",
        )
        self.original_player.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self._original_loaded_signature: str | None = None

        # Play Converted — populated once a run finishes (see
        # _load_preview_from_report()).
        self.player = AudioPlayerPanel(self, title="Play Converted")
        self.player.grid(row=4, column=0, sticky="ew", pady=(8, 0))

        btn_row = ttk.Frame(self)
        btn_row.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        self.back_btn = ttk.Button(btn_row, text="< Back", command=self._on_back_clicked)
        self.back_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btn_row, text="Cancel", command=self._cancel, state="disabled")
        self.open_btn = ttk.Button(btn_row, text="Open Output Folder", command=self._open_output, state="disabled")
        self.run_btn = ttk.Button(btn_row, text="Run", command=self._start_run)
        self.run_btn.pack(side="right")
        self.open_btn.pack(side="right", padx=(0, 6))
        self.cancel_btn.pack(side="right", padx=(0, 6))

        # Forward the same INFO+ log messages that already appear on the
        # CLI/console (and in the rotating log file) into this step's log
        # text area, in addition to the hand-written progress/status lines
        # already sent here via log_callback in _start_run()'s worker.
        if core is not None:
            self._log_handler = _QueueLogHandler(self.msg_queue)
            core.log.addHandler(self._log_handler)

        self.after(200, self._poll_queue)

    def _on_back_clicked(self):
        if self.on_back:
            self.on_back()

    # Friendly labels for the flat DSP/speaker-bias params dict, in the
    # order they should appear under "Mastering (DSP)" / "Surround Speaker Mapping".
    _DSP_SUMMARY_FIELDS = [
        ("bass_shelf_db", "Bass shelf (dB)"),
        ("treble_shelf_db", "Treble shelf (dB)"),
        ("presence_db", "Presence peak (dB)"),
        ("notch_db", "Notch cut (dB)"),
        ("multiband_compress", "Multiband compression"),
        ("deesser", "De-esser"),
        ("final_lufs", "Target loudness (LUFS)"),
        ("vocal_lufs", "Vocal volume (LUFS)"),
        ("vocal_presence_db", "Vocal presence tone (dB)"),
        ("vocal_air_db", "Vocal air tone (dB)"),
        ("vocal_mud_cut_db", "Vocal mud-cut tone (dB)"),
        ("width_bands", "Per-band stereo width"),
        ("saturation", "Tape saturation"),
        ("crystalizer", "Crystalizer"),
        ("crossfeed", "Headphone crossfeed"),
        ("declick", "Declick / declip"),
        ("denoise", "Denoise"),
    ]

    _SPEAKER_BIAS_SUMMARY_FIELDS = [
        ("front", "Front L/R"),
        ("centre", "Centre"),
        ("rear", "Rear L/R"),
    ]

    def _summary_section(self, row: int, title: str) -> int:
        lbl = tk.Label(
            self.summary_inner, text=title, font=("TkDefaultFont", 9, "bold"),
            fg="#1a4d8f", bg="white", anchor="w",
        )
        lbl.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 2))
        return row + 1

    def _summary_kv(self, row: int, key: str, value) -> int:
        k = tk.Label(
            self.summary_inner, text=f"{key}:", font=("TkDefaultFont", 8),
            fg="#444444", bg="white", anchor="w", width=28, justify="left",
        )
        k.grid(row=row, column=0, sticky="nw", padx=(14, 6))
        v = tk.Label(
            self.summary_inner, text=str(value), font=("TkDefaultFont", 8),
            fg="#000000", bg="white", anchor="w", justify="left", wraplength=320,
        )
        v.grid(row=row, column=1, sticky="w")
        return row + 1

    def refresh_summary(self):
        for child in self.summary_inner.winfo_children():
            child.destroy()

        params = self.state.resolve()
        content_key = self.state.content_key or self.state.content_auto_detected or "early_digital"
        is_5_1 = params.get("layout") == "5.1"

        # Surround output only applies to a 5.1 device profile -- hide the
        # whole section for every other target device rather than showing
        # controls that do nothing.
        if is_5_1:
            self.surround_frame.grid()
        else:
            self.surround_frame.grid_remove()

        r = 0
        r = self._summary_section(r, "Source & Output")
        r = self._summary_kv(r, "Input", self.state.input_path or "(not set)")
        r = self._summary_kv(r, "Input mode", "Folder (batch)" if self.state.input_mode == "folder" else "Single file")
        r = self._summary_kv(r, "Output folder", self.state.output_dir)
        r = self._summary_kv(r, "Format", f"{self.state.output_format} @ {self.state.bitrate}")
        r = self._summary_kv(r, "Album mode", "On" if self.state.album_mode else "Off")
        r = self._summary_kv(r, "Parallel workers", self.state.parallel_workers)
        r = self._summary_kv(r, "Preview segment",
                             f"{self.state.preview_start_s:g}s start, {self.state.preview_duration_s}s duration")

        r = self._summary_section(r, "Target Device")
        r = self._summary_kv(r, "Device", DEVICE_PROFILES[self.state.device_key].display_name)

        r = self._summary_section(r, "Content Type")
        r = self._summary_kv(r, "Content", CONTENT_PROFILES.get(content_key, CONTENT_PROFILES["early_digital"]).display_name)
        if self.state.content_confidence:
            r = self._summary_kv(r, "Detection confidence", f"{self.state.content_confidence * 100:.0f}%")

        r = self._summary_section(r, "Enhancement")
        r = self._summary_kv(r, "Demucs model", self.state.demucs_model)
        r = self._summary_kv(r, "Stem cache", "On" if self.state.use_cache else "Off")
        r = self._summary_kv(r, "Experimental tools enabled", "Yes" if self.state.experimental_ack else "No")
        r = self._summary_kv(r, "VoiceFixer", "On" if params.get("use_voicefixer") else "Off")
        r = self._summary_kv(r, "DeepFilterNet", "On" if params.get("use_deepfilternet") else "Off")

        r = self._summary_section(r, "Mastering (DSP)")
        for key, label in self._DSP_SUMMARY_FIELDS:
            if key not in self.state.params and key not in params:
                continue
            value = self.state.params.get(key, params.get(key))
            if isinstance(value, bool):
                value = "On" if value else "Off"
            elif isinstance(value, float):
                value = f"{value:+.1f}" if "db" in key or key == "final_lufs" or key == "vocal_lufs" else f"{value:.2f}"
            r = self._summary_kv(r, label, value)
        r = self._summary_kv(r, "Trim silence", "On" if self.state.trim_silence else "Off")
        r = self._summary_kv(r, "Fade in / out", f"{self.state.fade_in_ms}ms / {self.state.fade_out_ms}ms")

        if is_5_1:
            r = self._summary_section(r, "Surround Speaker Mapping")
            for group_key, group_label in self._SPEAKER_BIAS_SUMMARY_FIELDS:
                source = self.state.params.get(f"speaker_bias_{group_key}", "off")
                db = self.state.params.get(f"speaker_bias_{group_key}_db", 0.0)
                if source == "off":
                    r = self._summary_kv(r, group_label, "Off (default mix only)")
                else:
                    label_map = dict(SPEAKER_BIAS_SOURCE_LABELS)
                    r = self._summary_kv(r, group_label, f"{label_map.get(source, source)}  ({db:+.1f} dB)")

            r = self._summary_section(r, "Output & Run")
            r = self._summary_kv(r, "Surround codec", self.codec_var.get().upper())
            r = self._summary_kv(r, "LFE mode", self.lfe_var.get().capitalize())

    def _start_run(self):
        if self.busy:
            return
        self.state.surround_codec = self.codec_var.get()
        self.state.lfe_mode = self.lfe_var.get()

        # Resolve to an absolute path up front so the folder we write to
        # and the folder "Open Output Folder" later opens are guaranteed
        # to be the same location, regardless of the process's working
        # directory at launch time.
        output_dir = Path(self.state.output_dir or str(Path.home() / "RemasterStudio" / "output")).expanduser()
        try:
            output_dir = output_dir.resolve()
        except Exception:
            pass
        self.state.output_dir = str(output_dir)
        self.state.save()

        files = core.collect_audio_files(Path(self.state.input_path))
        if not files:
            messagebox.showerror("No files", "No audio files found at the selected input path.")
            return

        self.cancel_event.clear()
        self.busy = True
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.open_btn.configure(state="disabled")
        self.back_btn.configure(state="disabled")
        self.progress.configure(value=0, maximum=len(files))
        self._clear_log()
        self.player.clear()

        params = self.state.resolve()

        def _worker():
            def _progress(done, total):
                self.msg_queue.put(("progress", done, total))
            def _log(msg):
                self.msg_queue.put(("log", msg))
            try:
                report = core.run_batch(
                    files, output_dir, params,
                    parallel_workers=self.state.parallel_workers,
                    album_mode=self.state.album_mode,
                    progress_callback=_progress,
                    log_callback=_log,
                    cancel_event=self.cancel_event,
                )
                self.msg_queue.put(("done", report))
            except Exception:
                self.msg_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=_worker, daemon=True).start()

    def _cancel(self):
        self.cancel_event.set()
        self._append_log("Cancelling…\n")

    def _open_output(self):
        import subprocess as sp, sys as _sys
        import os as _os
        raw_path = self.state.output_dir or str(Path.home() / "RemasterStudio" / "output")
        resolved = Path(raw_path).expanduser()
        try:
            resolved = resolved.resolve()
        except Exception:
            pass  # keep the expanduser()'d path if resolve() itself fails

        try:
            if not resolved.exists():
                # Self-heal: the folder may not exist yet (e.g. relative
                # path resolved differently than at run time, or nothing
                # has been exported here yet) — create it rather than
                # erroring out with a raw OS "file not found".
                resolved.mkdir(parents=True, exist_ok=True)

            path_str = str(resolved)
            if _sys.platform == "win32":
                _os.startfile(path_str)
            elif _sys.platform == "darwin":
                sp.run(["open", path_str])
            else:
                sp.run(["xdg-open", path_str])
        except Exception as exc:
            messagebox.showinfo("Output folder", f"Output saved to:\n{resolved}\n\n({exc})")

    def _poll_queue(self):
        try:
            while True:
                item = self.msg_queue.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _, done, total = item
                    self.progress.configure(maximum=total, value=done)
                elif kind == "log":
                    self._append_log(item[1] + "\n")
                elif kind == "done":
                    report = item[1]
                    self._append_log(f"\nDone: {report.succeeded} succeeded, {report.failed} failed.\n")
                    self._load_preview_from_report(report)
                    self._finish(success=True)
                elif kind == "error":
                    self._append_log(f"\nERROR:\n{item[1]}\n")
                    self._finish(success=False)
        except queue.Empty:
            pass
        self.after(200, self._poll_queue)

    def _finish(self, success: bool):
        self.busy = False
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.open_btn.configure(state="normal")
        self.back_btn.configure(state="normal")
        if self.on_run_complete:
            self.on_run_complete(success)

    def _load_preview_from_report(self, report):
        """Point the preview player at the first successfully-produced
        output file (preferring a directly-playable stereo file over a
        5.1 MKV, which pydub/ffmpeg can still often open, but the stereo
        file is the more universally reliable preview source)."""
        for fr in report.files:
            if fr.success and fr.output_paths:
                candidates = [Path(p) for p in fr.output_paths]
                preferred = next((p for p in candidates if p.suffix.lower() != ".mkv"), candidates[0])
                self.player.load(preferred)
                return
        self.player.status_label.configure(text="No output was produced to preview.")

    def _load_original_preview(self):
        """Point the Play Original player at the first input file, if any.
        Cheap no-op if the input selection hasn't changed since the last
        time this ran (see on_enter())."""
        if not self.state.input_path or core is None:
            self.original_player.clear()
            return
        try:
            files = core.collect_audio_files(Path(self.state.input_path))
        except Exception:
            files = []
        if not files:
            self.original_player.clear()
            self.original_player.status_label.configure(text="No input file found to preview.")
            return
        self.original_player.load(files[0])

    def on_enter(self):
        """Called by the wizard controller when this step becomes visible.
        Reloads the Play Original preview only if the input selection
        actually changed since the last visit (e.g. the user went back to
        Step 1 and picked a different file/folder)."""
        signature = f"{self.state.input_mode}:{self.state.input_path}"
        if signature != self._original_loaded_signature:
            self._original_loaded_signature = signature
            self._load_original_preview()

    def on_leave(self):
        """Called by the wizard controller when navigating away from this
        step. Stops any in-progress preview playback."""
        self.original_player.on_leave()
        self.player.on_leave()

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def commit(self):
        self.state.surround_codec = self.codec_var.get()
        self.state.lfe_mode = self.lfe_var.get()

    def validate(self) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Wizard controller
# ---------------------------------------------------------------------------

class WizardController:
    STEP_NAMES = [
        "Source & Output", "Target Device", "Content Type",
        "Enhancement", "DSP Review", "Output & Run",
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.state = WizardState()
        self.current_step = 0

        root.title("OpenRemaster")
        self._size_window()
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        self._apply_style()

        # Step indicator
        self.step_label = ttk.Label(root, text="", font=("TkDefaultFont", 10, "bold"))
        self.step_label.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))

        # Page container
        self.container = ttk.Frame(root)
        self.container.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        self.container.columnconfigure(0, weight=1)
        self.container.rowconfigure(0, weight=1)

        self.pages: list[ttk.Frame] = [
            Step1Source(self.container, self.state),
            Step2Device(self.container, self.state),
            Step3Content(self.container, self.state),
            Step4Enhancement(self.container, self.state),
            Step5DSP(self.container, self.state),
            Step6Run(self.container, self.state, on_run_complete=self._on_run_complete, on_back=self._go_back),
        ]
        for page in self.pages:
            page.grid(row=0, column=0, sticky="nsew")

        # Nav bar
        nav = ttk.Frame(root)
        nav.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))
        self.back_btn = ttk.Button(nav, text="< Back", command=self._go_back)
        self.back_btn.pack(side="left")
        self.next_btn = ttk.Button(nav, text="Next >", command=self._go_next)
        self.next_btn.pack(side="right")

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_step(0)

    def _size_window(self):
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()
        win_w, win_h = int(w * 0.6), int(h * 0.75)
        x, y = (w - win_w) // 2, (h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.minsize(760, 560)

    def _apply_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Blue-based palette. "clam"'s defaults (esp. Progressbar) render
        # nearly invisible against the window background — every colour
        # below is set explicitly rather than left to theme defaults.
        BLUE_DARK   = "#1a4d8f"
        BLUE_MID    = "#2f6fbf"
        BLUE_LIGHT  = "#e8f0fb"
        BLUE_ACCENT = "#3f7fd6"
        BG          = "#f4f7fc"

        self.root.configure(bg=BG)

        style.configure(".", background=BG)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG)
        style.configure("TLabelframe", background=BG, bordercolor=BLUE_MID)
        style.configure("TLabelframe.Label", background=BG, foreground=BLUE_DARK, font=("TkDefaultFont", 9, "bold"))

        style.configure("TButton", background=BLUE_MID, foreground="white", padding=6, borderwidth=0)
        style.map("TButton",
                  background=[("disabled", "#a9bcd6"), ("active", BLUE_DARK), ("pressed", BLUE_DARK)],
                  foreground=[("disabled", "#eef2f8")])

        style.configure("TCheckbutton", background=BG)
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("Toolbutton", background=BG)
        style.map("Toolbutton", background=[("active", BLUE_LIGHT), ("selected", BLUE_LIGHT)])

        style.configure("TRadiobutton", background=BG)
        style.configure("TCombobox", fieldbackground="white", background="white")
        style.configure("TEntry", fieldbackground="white")

        style.configure("TNotebook", background=BG)
        style.configure("Selected.TFrame", background=BLUE_LIGHT)

        # Progress bar: explicit trough + bar colours so it's clearly
        # visible instead of blending into the background (change #1).
        style.configure(
            "TProgressbar",
            troughcolor="#d6e2f2",
            background=BLUE_ACCENT,
            bordercolor="#d6e2f2",
            lightcolor=BLUE_ACCENT,
            darkcolor=BLUE_ACCENT,
            thickness=16,
        )

    def _show_step(self, index: int):
        self.current_step = index
        self.pages[index].tkraise()
        self.step_label.configure(text=f"Step {index + 1} of 6 — {self.STEP_NAMES[index]}")
        if index == len(self.pages) - 1:
            # Step 6 renders its own Back/Cancel/Open Output Folder/Run row,
            # so the global nav bar's Back and Run ">" buttons are both
            # redundant there — hide them entirely.
            self.back_btn.pack_forget()
            self.next_btn.pack_forget()
        else:
            self.back_btn.configure(state="disabled" if index == 0 else "normal")
            self.back_btn.pack(side="left")
            self.next_btn.configure(text="Next >", state="normal")
            self.next_btn.pack(side="right")

        on_enter = getattr(self.pages[index], "on_enter", None)
        if on_enter:
            on_enter()

        # Step-specific refresh hooks
        if index == 2:
            self.pages[2].run_detection()
        if index == 4:
            self.pages[4].refresh_profile_label()
        if index == 5:
            self.pages[5].refresh_summary()

    def _leave_current_step(self):
        """Call the current page's on_leave hook, if it defines one (e.g.
        Step5DSP stops any playing/processing audio preview)."""
        on_leave = getattr(self.pages[self.current_step], "on_leave", None)
        if on_leave:
            on_leave()

    def _go_next(self):
        page = self.pages[self.current_step]
        err = page.validate()
        if err:
            messagebox.showerror("Cannot continue", err)
            return
        page.commit()
        self._leave_current_step()
        self.state.save()
        if self.current_step < len(self.pages) - 1:
            self._show_step(self.current_step + 1)

    def _go_back(self):
        page = self.pages[self.current_step]
        page.commit()
        self._leave_current_step()
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _on_run_complete(self, success: bool):
        pass  # Step6 handles its own UI state; hook available for future use

    def _on_close(self):
        try:
            self.pages[self.current_step].commit()
            self._leave_current_step()
        except Exception:
            pass
        self.state.cleanup_and_save()
        core.stop_audio_preview()
        self.root.destroy()


def _set_app_icon(root: tk.Tk) -> None:
    """Set the window/title-bar/taskbar icon to a small flat cassette-tape
    graphic, decoded from the embedded base64 PNG data (_APP_ICON_PNG_B64)
    above. Multiple sizes are supplied so Tk can pick the best match for
    the title bar, alt-tab switcher, and taskbar respectively. Never
    raises — a missing/broken icon should never stop the app from
    starting.
    """
    try:
        images = [
            tk.PhotoImage(data=b64, format="png", master=root)
            for b64 in _APP_ICON_PNG_B64.values()
        ]
        if images:
            root.iconphoto(True, *images)
            # Keep references alive — Tk drops the image if it's garbage
            # collected, which would silently blank the icon.
            root._app_icon_images = images
    except Exception as exc:
        try:
            log_fn = core.log.debug if core is not None else None
        except Exception:
            log_fn = None
        if log_fn:
            log_fn("Could not set app icon: %s", exc)


def _increase_default_fonts(delta: int = 1):
    for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont"):
        try:
            f = tkfont.nametofont(name)
            f.configure(size=f.cget("size") + delta)
        except tk.TclError:
            pass


def _load_core_and_launch(root: tk.Tk, splash: ttk.Frame, bar: ttk.Progressbar):
    """Import engine (dependency probing + numpy/scipy/etc
    imports) on a background thread so the loading screen stays animated
    and the window never appears frozen, then swap it for the real wizard.
    """
    result: dict = {}

    def _import_core():
        import engine as _core_module
        result["core"] = _core_module

    thread = threading.Thread(target=_import_core, daemon=True)
    thread.start()

    def _poll():
        if thread.is_alive():
            root.after(50, _poll)
            return
        global core
        core = result.get("core")
        bar.stop()
        splash.destroy()
        WizardController(root)

    root.after(50, _poll)


def main():
    root = tk.Tk()
    root.title("OpenRemaster")
    _set_app_icon(root)
    _increase_default_fonts(1)

    # Loading screen: shown immediately, before the (potentially slow)
    # core engine import runs, so the app never appears to hang on launch.
    splash = ttk.Frame(root, padding=40)
    splash.pack(expand=True, fill="both")
    ttk.Label(splash, text="OpenRemaster", font=("TkDefaultFont", 16, "bold")).pack(pady=(20, 8))
    ttk.Label(splash, text="Checking installed audio dependencies…").pack(pady=(0, 14))
    bar = ttk.Progressbar(splash, mode="indeterminate", length=240)
    bar.pack()
    bar.start(12)
    root.update()  # force the loading screen to paint before we start the import

    _load_core_and_launch(root, splash, bar)
    root.mainloop()


if __name__ == "__main__":
    main()