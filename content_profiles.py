"""content_profiles.py — Source-material profiles for OpenRemaster.

Pure data module: no executable top-level code, no imports from other
project files.

A ContentProfile describes the RECORDING'S era and technology, not its
language or genre. This is a deliberate choice: what actually drives the
DSP decisions (how much hiss to expect, whether tape rolloff needs
restoring, whether the mix is already loudness-maximised) is the
recording/mastering technology of the period, not what language is being
sung. A 1970s analogue tape recording needs essentially the same
restoration whether the vocal is in Tamil, Hindi, Bengali, or Punjabi —
which is exactly why, in earlier versions of this project, the
language-keyed "classic" profiles for different languages ended up as
near-duplicates of each other. Keying by era instead removes that
duplication and scales to any language without extra profiles.

Two profiles exist outside the era ladder because they're defined by
performance style rather than period — Carnatic/Hindustani classical
(minimal processing, preserve the performance) and devotional/folk music
(often live-recorded, group vocals, harmonium/tabla) — since these can be
recorded in any era but need consistently different treatment than
mainstream film/popular music of the same period.

If you work with recordings from a region or era whose noise floor, tape
character, or mastering convention doesn't fit well here, please see
CONTRIBUTING.md — profile tuning is meant to grow via real calibration
data, not guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ContentKey = Literal[
    "classic_analogue",
    "cassette_era",
    "early_digital",
    "modern_mastered",
    "carnatic_hindustani",
    "devotional_folk",
    "bgm",
]


# ---------------------------------------------------------------------------
# ContentProfile
# ---------------------------------------------------------------------------

@dataclass
class ContentProfile:
    """Describes the source material's era, technology, and processing needs.

    Era-based processing:
      - ClassicAnalogue (pre-1980): analogue tape sources — high noise
        floor, compressed/rolled-off highs, genuine dynamic range worth
        preserving. Restoration first, then gentle enhancement.
      - CassetteEra (1980s-1990s): cassette/early-digital transfer —
        tape hiss and wow/flutter character rather than vinyl/reel
        artefacts, cleaner vocals than the analogue era but not yet
        digitally polished.
      - EarlyDigital (mid-1980s-2000s): early digital or
        analogue-to-digital transfer, more dynamic range, occasional
        noise, mixed high-frequency response.
      - ModernMastered (post-2000s): already well-mastered, potentially
        already loudness-maximised. Less aggressive processing to avoid
        over-processing what's already been mastered once.
      - CarnaticHindustani: Indian classical (Carnatic or Hindustani).
        Preserve dynamics and timbre above all — minimal processing,
        the recording is the performance.
      - DevotionalFolk: bhajans, qawwali, kirtan, folk — often
        live/group-recorded with harmonium, tabla, or acoustic
        instruments. Light restoration, preserve natural dynamics.
      - BGM/Instrumental: film background scores and instrumental
        tracks. Orchestral warmth, wide stereo image, no vocal-specific
        processing needed.
    """
    key: ContentKey
    display_name: str
    description: str

    # --- Restoration ---
    declick: bool = False
    declick_threshold: float = 2.0
    denoise: bool = False
    denoise_amount: float = 8.0
    denoise_floor: float = -50.0
    denoise_type: Literal["white", "vinyl", "shellac"] = "white"

    # --- Stem processing ---
    use_stems: bool = False         # requires Demucs
    use_demucs: bool = False
    demucs_model: str = "htdemucs"
    vocal_lufs: float = -18.0
    music_lufs: float = -18.0

    # --- Vocal chain (when stems available) ---
    vocal_presence_db: float = 0.0
    vocal_air_db: float = 0.0
    vocal_mud_cut_db: float = 0.0
    deesser: bool = False
    deesser_threshold_db: float = -24.0
    deesser_ratio: float = 4.0
    deesser_freq: int = 5000

    # --- Instrument chain ---
    inst_bass_shelf_db: float = 0.0
    inst_air_shelf_db: float = 0.0
    inst_harmonic_exciter: bool = False
    inst_exciter_freq: int = 3000
    inst_exciter_amount: float = 0.3

    # --- Mastering dynamics ---
    auto_loudness: bool = False
    multiband_compress: bool = False
    mb_low_ratio: float = 1.5
    mb_mid_ratio: float = 1.5
    mb_high_ratio: float = 1.5

    # --- Tonal ---
    # Content-side corrections (e.g. tape rolloff restoration) on top of
    # the device-side EQ.
    bass_correction_db: float = 0.0
    treble_correction_db: float = 0.0

    # --- Saturation ---
    saturation: bool = False
    saturation_drive_db: float = 3.0
    saturation_mix: float = 0.2

    # --- Output dynamics target ---
    final_lufs_override: float | None = None  # None = use DeviceProfile.final_lufs
    headroom_db_override: float | None = None

    # --- Explicit device-override vetoes ---
    # Most content profiles leave multiband_compress/saturation at their
    # dataclass default (False) simply because they never took a position
    # on it — for those, the device profile's own hardware-protective
    # choice (e.g. bluetooth_speaker enabling gentle multiband compression
    # to keep small drivers from breaking up) should still win, same as
    # before. A few profiles (Carnatic/Hindustani, devotional/folk,
    # modern-mastered) DO take an explicit position — "preserve dynamics",
    # "don't compress hard", "don't compound what's already mastered" —
    # and that position should not be silently overridden just because a
    # loud soundbar or Bluetooth speaker profile wants compression/warmth
    # for its own hardware reasons. These two flags let a content profile
    # say "no, really, don't" instead of only being able to say "yes".
    veto_device_multiband: bool = False
    veto_device_saturation: bool = False


# ---------------------------------------------------------------------------
# Defined content profiles
# ---------------------------------------------------------------------------

CONTENT_PROFILES: dict[ContentKey, ContentProfile] = {

    "classic_analogue": ContentProfile(
        key="classic_analogue",
        display_name="Classic Analogue (pre-1980)",
        description=(
            "Analogue tape-era recordings, any language or region. Expect "
            "hiss, tape rolloff on highs, clicks from old media, and "
            "genuine dynamic range worth preserving. Restoration first, "
            "then gentle enhancement."
        ),
        declick=True,
        declick_threshold=2.0,
        denoise=True,
        denoise_amount=10.0,   # afftdn measurably strips real high-frequency
                                # musical content along with hiss at higher
                                # settings (dose-dependent; nr=14 costs
                                # roughly -8.6dB in the 6-12kHz band vs
                                # nr=10's ~-7dB in testing), so this trades
                                # slightly less hiss removal for more
                                # preserved detail
        denoise_floor=-50.0,
        denoise_type="white",
        use_stems=False,            # Demucs works poorly on mono/near-mono old recordings
        vocal_presence_db=1.5,
        vocal_air_db=2.0,
        vocal_mud_cut_db=-1.5,
        deesser=True,
        deesser_threshold_db=-26.0,
        deesser_ratio=3.0,
        deesser_freq=5500,
        inst_bass_shelf_db=1.5,    # restore low-end lost to tape rolloff
        inst_air_shelf_db=2.5,     # restore highs lost to tape compression
        inst_harmonic_exciter=True,
        inst_exciter_freq=3500,
        inst_exciter_amount=0.25,
        auto_loudness=False,        # preserve the original dynamic character
        multiband_compress=False,
        bass_correction_db=1.5,
        treble_correction_db=4.5,  # compensates for both tape rolloff AND
                                    # afftdn's own collateral high-frequency
                                    # loss (see denoise_amount note above),
                                    # keeping net treble roughly neutral-to-
                                    # restored instead of a net loss vs source
        saturation=True,
        saturation_drive_db=4.0,   # restore warmth lost in digitisation
        saturation_mix=0.25,
        final_lufs_override=None,
    ),

    "cassette_era": ContentProfile(
        key="cassette_era",
        display_name="Cassette Era (1980s-1990s)",
        description=(
            "Cassette and early-digital-transfer era recordings, any "
            "language or region. Cassette hiss and wow/flutter character "
            "rather than vinyl/tape-reel artefacts. Vocals are usually "
            "cleaner than the classic analogue era but not yet as "
            "polished as 2000s+ digital masters. Light restoration, "
            "stem-based vocal/instrument chain."
        ),
        declick=False,
        denoise=True,
        denoise_amount=6.0,
        denoise_floor=-52.0,
        denoise_type="white",
        use_stems=True,
        use_demucs=True,
        demucs_model="htdemucs",
        vocal_lufs=-18.0,
        music_lufs=-18.0,
        vocal_presence_db=1.3,
        vocal_air_db=1.5,
        vocal_mud_cut_db=-1.2,
        deesser=True,
        deesser_threshold_db=-25.0,
        deesser_ratio=3.2,
        deesser_freq=5200,
        inst_bass_shelf_db=1.0,
        inst_air_shelf_db=1.8,
        inst_harmonic_exciter=True,
        inst_exciter_freq=3200,
        inst_exciter_amount=0.25,
        auto_loudness=False,
        multiband_compress=True,
        mb_low_ratio=1.8,
        mb_mid_ratio=1.6,
        mb_high_ratio=1.5,
        bass_correction_db=0.8,
        treble_correction_db=1.5,
        saturation=True,
        saturation_drive_db=3.5,
        saturation_mix=0.2,
    ),

    "early_digital": ContentProfile(
        key="early_digital",
        display_name="Early Digital (mid-1980s-2000s)",
        description=(
            "Early digital or analogue-to-digital transfer recordings, "
            "any language or region. More dynamic range than the tape "
            "eras, occasional hiss, mixed high-frequency response. "
            "Moderate restoration and enhancement."
        ),
        declick=False,
        denoise=True,
        denoise_amount=10.0,
        denoise_type="white",
        use_stems=True,
        use_demucs=True,
        demucs_model="htdemucs",
        vocal_lufs=-18.0,
        music_lufs=-18.0,
        vocal_presence_db=1.0,
        vocal_air_db=1.5,
        vocal_mud_cut_db=-1.0,
        deesser=True,
        deesser_threshold_db=-24.0,
        deesser_ratio=3.5,
        deesser_freq=5000,
        inst_bass_shelf_db=0.5,
        inst_air_shelf_db=1.0,
        inst_harmonic_exciter=True,
        inst_exciter_freq=3000,
        inst_exciter_amount=0.2,
        auto_loudness=False,
        multiband_compress=True,
        mb_low_ratio=2.0,
        mb_mid_ratio=1.8,
        mb_high_ratio=1.5,
        bass_correction_db=0.5,
        treble_correction_db=1.0,
        saturation=True,
        saturation_drive_db=4.0,
        saturation_mix=0.2,
    ),

    "modern_mastered": ContentProfile(
        key="modern_mastered",
        display_name="Modern Mastered (post-2000s)",
        description=(
            "Already well-mastered post-2000s recordings, any language, "
            "genre, or region. May already be loudness-maximised. Less "
            "aggressive processing to avoid compounding what the original "
            "mastering already did."
        ),
        declick=False,
        denoise=False,
        use_stems=True,
        use_demucs=True,
        demucs_model="htdemucs_ft",
        vocal_lufs=-18.0,
        music_lufs=-18.0,
        vocal_presence_db=1.0,
        vocal_air_db=1.0,
        vocal_mud_cut_db=-0.5,
        deesser=True,
        deesser_threshold_db=-22.0,
        deesser_ratio=4.0,
        deesser_freq=5000,
        inst_bass_shelf_db=0.0,
        inst_air_shelf_db=0.5,
        inst_harmonic_exciter=False,
        auto_loudness=True,
        multiband_compress=True,
        mb_low_ratio=2.0,
        mb_mid_ratio=1.8,
        mb_high_ratio=1.5,
        bass_correction_db=0.0,
        treble_correction_db=0.0,
        saturation=False,
        final_lufs_override=None,
        # "Less aggressive processing to avoid compounding what the
        # original mastering already did" (see class docstring) is an
        # explicit veto on saturation specifically — a device profile's
        # own warmth/character choice shouldn't be layered on top of a
        # recording that's already been mastered once.
        veto_device_saturation=True,
    ),

    "carnatic_hindustani": ContentProfile(
        key="carnatic_hindustani",
        display_name="Carnatic / Hindustani Classical",
        description=(
            "Indian classical music, Carnatic or Hindustani. Preserve "
            "dynamics and timbre above all. Minimal processing — the "
            "recording is the performance, and heavy enhancement would "
            "corrupt the natural acoustic of the instruments and voice."
        ),
        declick=True,
        declick_threshold=1.5,
        denoise=True,
        denoise_amount=8.0,
        denoise_type="white",
        use_stems=False,
        vocal_presence_db=0.5,
        vocal_air_db=0.5,
        vocal_mud_cut_db=0.0,
        deesser=False,
        inst_bass_shelf_db=0.0,
        inst_air_shelf_db=0.5,
        inst_harmonic_exciter=False,
        auto_loudness=False,
        multiband_compress=False,
        bass_correction_db=0.0,
        treble_correction_db=0.5,
        saturation=False,
        final_lufs_override=-18.0,  # wider dynamic range for classical
        headroom_db_override=1.0,
        # "Preserve dynamics and timbre above all" (see class docstring) is
        # an explicit veto, not just an unset default — don't let a device
        # profile's own compression/saturation defaults override it.
        veto_device_multiband=True,
        veto_device_saturation=True,
    ),

    "devotional_folk": ContentProfile(
        key="devotional_folk",
        display_name="Devotional / Folk",
        description=(
            "Bhajans, qawwali, kirtan, and folk recordings from any "
            "region — often live or group-recorded, with harmonium, "
            "tabla, or acoustic instruments and group/chorus vocals. "
            "Light restoration, moderate stem separation to help the "
            "lead vocal stand out from a chorus, dynamics kept natural "
            "rather than compressed hard."
        ),
        declick=False,
        denoise=True,
        denoise_amount=6.0,
        denoise_floor=-52.0,
        denoise_type="white",
        use_stems=True,
        use_demucs=True,
        demucs_model="htdemucs",
        vocal_lufs=-17.0,
        music_lufs=-18.0,
        vocal_presence_db=1.0,
        vocal_air_db=1.0,
        vocal_mud_cut_db=-1.0,
        deesser=True,
        deesser_threshold_db=-25.0,
        deesser_ratio=3.0,
        deesser_freq=5200,
        inst_bass_shelf_db=0.5,
        inst_air_shelf_db=1.0,
        inst_harmonic_exciter=False,   # keep acoustic instruments natural
        auto_loudness=False,
        multiband_compress=False,       # preserve live-performance dynamics
        bass_correction_db=0.0,
        treble_correction_db=0.5,
        saturation=False,
        final_lufs_override=-16.0,      # a bit wider than pop mastering
        headroom_db_override=0.5,
        # "Dynamics kept natural rather than compressed hard" is an
        # explicit veto on multiband compression, not just an unset
        # default — see class docstring.
        veto_device_multiband=True,
    ),

    "bgm": ContentProfile(
        key="bgm",
        display_name="BGM / Instrumental",
        description=(
            "Film background scores and instrumental tracks, any region. "
            "Orchestral warmth, wide stereo image, no vocal-specific "
            "processing. Emphasis on spatial depth and harmonic richness."
        ),
        declick=False,
        denoise=False,
        use_stems=False,
        deesser=False,
        inst_bass_shelf_db=1.0,
        inst_air_shelf_db=2.0,
        inst_harmonic_exciter=True,
        inst_exciter_freq=2500,
        inst_exciter_amount=0.35,
        auto_loudness=False,
        multiband_compress=True,
        mb_low_ratio=2.0,
        mb_mid_ratio=1.5,
        mb_high_ratio=1.5,
        bass_correction_db=0.5,
        treble_correction_db=1.5,
        saturation=True,
        saturation_drive_db=5.0,
        saturation_mix=0.3,
    ),

}


# ---------------------------------------------------------------------------
# Content auto-detection helpers (used by detection logic in the engine)
# ---------------------------------------------------------------------------

# Year-to-era mapping: (start_year_inclusive, end_year_inclusive) -> content_key
# These boundaries are approximate and were originally calibrated against
# Tamil-cinema recording history; mastering conventions shift by region and
# label, so treat this as a reasonable default, not a hard rule. If you have
# calibration data from another region/era, see CONTRIBUTING.md.
ERA_YEAR_RANGES: list[tuple[int, int, ContentKey]] = [
    (0,    1979, "classic_analogue"),
    (1980, 1999, "cassette_era"),
    (2000, 2005, "early_digital"),
    (2006, 9999, "modern_mastered"),
]


def era_from_year(year: int) -> ContentKey | None:
    """Map a recording year to the most likely content key based on era alone.
    Returns None if year is outside all defined ranges (e.g. 0 or unknown).
    """
    if year <= 0:
        return None
    for start, end, key in ERA_YEAR_RANGES:
        if start <= year <= end:
            return key
    return None


# Keywords in filename/tags → content key hint. Deliberately genre/era
# based rather than language based — a filename containing "bhajan" or
# "carnatic" is a strong content signal regardless of language, whereas
# guessing content type from a language name (e.g. assuming any Hindi- or
# Tamil-tagged file is film music) produces exactly the kind of bias this
# project is trying to avoid.
FILENAME_KEYWORD_MAP: list[tuple[list[str], ContentKey]] = [
    (["bgm", "background", "theme", "instrumental", "score", "ost"], "bgm"),
    (["carnatic", "hindustani", "classical", "kriti", "varnam",
      "thillana", "raga", "alapana", "pallavi", "khayal", "dhrupad"],
     "carnatic_hindustani"),
    (["bhajan", "qawwali", "kirtan", "devotional", "aarti", "sufi",
      "folk"], "devotional_folk"),
    (["remix", "club", "edm", "dance"], "modern_mastered"),
]


def content_hint_from_filename(filename: str) -> ContentKey | None:
    """Scan a filename (lowercased) for known content-type keywords.
    Returns the first matching content key, or None.
    """
    lower = filename.lower()
    for keywords, key in FILENAME_KEYWORD_MAP:
        if any(kw in lower for kw in keywords):
            return key
    return None