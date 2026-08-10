"""device_profiles.py — Playback hardware profiles for OpenRemaster.

Pure data module: no executable top-level code, no imports from other
project files.

A DeviceProfile describes a CATEGORY of playback hardware, not a specific
product. The categories are deliberately generic (a "soundbar_stereo"
profile should work reasonably on any 2.0 soundbar, not just one brand)
because the goal is to cover the range of gear people actually own —
Bluetooth speakers, soundbars, home theatre systems, headphones — without
requiring a hand-tuned entry per product.

The reasoning behind each profile's values is about the CLASS of hardware,
not a specific unit:
  - Small/cheap drivers (Bluetooth speakers, basic soundbars) tend to be
    thin in the bass and can get harsh or muddy pushed loud — profiles
    for these lean on gentle multiband compression and modest shelving
    rather than aggressive boosts that would clip or distort further.
  - Systems with no dedicated centre channel (most 2.0/2.1/2.2 soundbars,
    stereo home theatre) tend to bury vocals in a busy mix — profiles for
    these add a presence-band boost to compensate.
  - Systems with their own subwoofer (2.1, 5.1) don't need as much bass
    shelving from us — the sub handles the low end after bass management,
    so piling on more bass here just causes muddiness or clipping at the
    sub. Systems with no sub (Bluetooth speaker, 2.0 soundbar) get more
    bass shelf, within the limits of what a small driver can handle
    without breaking up.
  - Full-size AV receivers driving 5.1 speakers often apply their own
    room correction / auto-EQ (e.g. Audyssey, YPAO, DTS:X post-processing
    on many mid-range receivers). Processing here stays conservative for
    5.1 so it doesn't fight whatever correction the receiver applies.
  - Headphones sit right against the ear with no room interaction, so
    hard-panned mixes can feel fatiguing — crossfeed and slightly reduced
    stereo width for extreme pans help there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DeviceKey = Literal[
    "bluetooth_speaker",
    "soundbar_stereo",
    "soundbar_2_1",
    "home_theatre_2_1",
    "home_theatre_5_1",
    "headphones",
    "general",
]
Layout    = Literal["stereo", "5.1"]
Codec     = Literal["ac3", "pcm"]
LFEMode   = Literal["silent", "gentle", "full"]


# ---------------------------------------------------------------------------
# DeviceProfile
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    """Describes a category of target playback hardware.

    These values compensate for the general characteristics of a hardware
    CLASS, not any single product's quirks. If you own a specific device
    and find a profile needs adjustment for it, use the DSP Review step
    to tweak sliders and (optionally) save your own named profile rather
    than editing these defaults — see CONTRIBUTING.md.
    """
    key: DeviceKey
    display_name: str
    description: str

    # --- Output format ---
    layout: Layout = "stereo"
    output_format: Literal["mp3", "flac", "wav"] = "mp3"
    bitrate: str = "320k"
    also_produce_stereo_mp3: bool = True  # always True for any 5.1 profile

    # --- Surround (5.1 only) ---
    surround_codec: Codec = "ac3"
    lfe_mode: LFEMode = "gentle"
    lfe_cutoff_hz: int = 80
    rear_cutoff_hz: int = 300
    rear_attenuation_db: float = -6.0
    rear_delay_ms: float = 15.0        # one-side delay for decorrelation
    centre_attenuation_db: float = -3.0

    # --- Loudness ---
    final_lufs: float = -14.0
    headroom_db: float = 0.5
    auto_loudness: bool = False

    # --- Tonal EQ (shelves) ---
    bass_shelf_hz: float = 100.0
    bass_shelf_db: float = 0.0
    treble_shelf_hz: float = 8000.0
    treble_shelf_db: float = 0.0

    # --- Presence (peaking) ---
    presence_hz: float = 2800.0
    presence_db: float = 0.0
    presence_q: float = 1.2

    # --- Notch (narrow cut — off by default; only for a hardware class
    # with a known, common resonance. Most generic categories leave this
    # at 0/disabled since a notch tuned for one product can do nothing,
    # or actively hurt, on another.) ---
    notch_hz: float = 0.0       # 0 = disabled
    notch_db: float = 0.0
    notch_q: float = 2.0

    # --- Dynamics ---
    multiband_compress: bool = False
    mb_low_ratio: float = 1.5
    mb_mid_ratio: float = 1.5
    mb_high_ratio: float = 1.5
    mb_low_crossover_hz: int = 200
    mb_high_crossover_hz: int = 4000

    # --- Saturation ---
    saturation: bool = False
    saturation_drive_db: float = 4.0
    saturation_mix: float = 0.2

    # --- Stereo width ---
    # For stereo output: single global width applied to the mastered mix.
    # For 5.1: width is applied per-band to FL/FR before the upmix step.
    width_bands: bool = False
    width_bass: float = 1.0
    width_mid: float = 1.0
    width_treble: float = 1.0

    # --- Crystalizer ---
    crystalizer: bool = False
    crystalizer_intensity: float = 2.0

    # --- Headphone crossfeed ---
    crossfeed: bool = False
    crossfeed_strength: float = 0.3
    crossfeed_range: float = 0.5


# ---------------------------------------------------------------------------
# Defined device profiles
# ---------------------------------------------------------------------------

DEVICE_PROFILES: dict[DeviceKey, DeviceProfile] = {

    "bluetooth_speaker": DeviceProfile(
        key="bluetooth_speaker",
        display_name="Bluetooth Speaker",
        description=(
            "Compact portable or desktop Bluetooth speaker — one or two "
            "small full-range drivers, no dedicated sub, limited headroom "
            "before distortion. Processing favours a light bass lift, "
            "gentle compression to avoid pushing small drivers into "
            "breakup, and a presence boost so vocals cut through."
        ),
        layout="stereo",
        output_format="mp3",
        bitrate="256k",
        also_produce_stereo_mp3=False,
        final_lufs=-14.0,
        headroom_db=0.7,           # a little extra safety margin for cheap DACs/amps
        auto_loudness=False,
        bass_shelf_hz=110.0,
        bass_shelf_db=1.5,         # modest — small drivers distort fast if pushed too hard
        treble_shelf_hz=9000.0,
        treble_shelf_db=1.0,
        presence_hz=2800.0,
        presence_db=1.5,           # helps vocals cut through on a single small driver
        presence_q=1.2,
        notch_hz=0.0,
        multiband_compress=True,
        mb_low_ratio=2.2,          # tighter bass control: keep small drivers from breaking up
        mb_mid_ratio=1.8,
        mb_high_ratio=1.6,
        mb_low_crossover_hz=200,
        mb_high_crossover_hz=4000,
        saturation=False,
        crystalizer=False,
        crossfeed=False,
        width_bands=False,
        width_bass=1.0,
        width_mid=1.0,
        width_treble=1.0,
    ),

    "soundbar_stereo": DeviceProfile(
        key="soundbar_stereo",
        display_name="Soundbar (2.0, no sub)",
        description=(
            "Stereo soundbar with no dedicated subwoofer and no centre "
            "channel. Needs bass shelving to compensate for the lack of a "
            "sub, and a presence boost since there's no centre speaker "
            "anchoring dialogue/vocals — everything shares the same pair "
            "of drivers as the music."
        ),
        layout="stereo",
        output_format="mp3",
        bitrate="320k",
        also_produce_stereo_mp3=False,
        final_lufs=-15.0,          # slightly more headroom than -14 for typical soundbar DSP
        headroom_db=0.5,
        auto_loudness=False,
        bass_shelf_hz=100.0,
        bass_shelf_db=2.0,         # compensate for no dedicated sub
        treble_shelf_hz=8000.0,
        treble_shelf_db=1.5,
        presence_hz=2800.0,
        presence_db=2.0,           # anchor vocals — no centre channel to do it for us
        presence_q=1.2,
        notch_hz=0.0,
        multiband_compress=True,
        mb_low_ratio=2.0,
        mb_mid_ratio=1.8,
        mb_high_ratio=1.6,
        mb_low_crossover_hz=200,
        mb_high_crossover_hz=4000,
        saturation=True,
        saturation_drive_db=4.0,
        saturation_mix=0.2,        # a little warmth — small/flat-sounding drivers benefit
        crystalizer=True,
        crystalizer_intensity=2.0,
        crossfeed=False,
        width_bands=True,
        width_bass=1.0,
        width_mid=1.2,
        width_treble=1.3,
    ),

    "soundbar_2_1": DeviceProfile(
        key="soundbar_2_1",
        display_name="Soundbar (2.1, with sub)",
        description=(
            "Soundbar with a dedicated (often wireless) subwoofer. The sub "
            "handles the low end, so bass shelving here stays flat to "
            "avoid doubling up and causing boom or clipping at the sub. "
            "Still no centre channel, so a presence boost for vocals stays."
        ),
        layout="stereo",
        output_format="mp3",
        bitrate="320k",
        also_produce_stereo_mp3=False,
        final_lufs=-15.0,
        headroom_db=0.5,
        auto_loudness=False,
        bass_shelf_hz=100.0,
        bass_shelf_db=0.0,         # the sub handles bass — don't double up
        treble_shelf_hz=8000.0,
        treble_shelf_db=1.0,
        presence_hz=2800.0,
        presence_db=1.5,
        presence_q=1.2,
        notch_hz=0.0,
        multiband_compress=True,
        mb_low_ratio=1.8,
        mb_mid_ratio=1.6,
        mb_high_ratio=1.5,
        mb_low_crossover_hz=200,
        mb_high_crossover_hz=4000,
        saturation=False,
        crystalizer=True,
        crystalizer_intensity=1.5,
        crossfeed=False,
        width_bands=True,
        width_bass=1.0,
        width_mid=1.2,
        width_treble=1.3,
    ),

    "home_theatre_2_1": DeviceProfile(
        key="home_theatre_2_1",
        display_name="Home Theatre (stereo + sub)",
        description=(
            "Stereo bookshelf/tower speaker pair with a separate powered "
            "subwoofer, in a proper listening room. More headroom and a "
            "larger listening distance than a soundbar, so processing is "
            "lighter — the goal is a clean, wide-range signal and let the "
            "system's own size do the work, similar to how a receiver's "
            "own room correction (if present) prefers a less-processed "
            "source to work from."
        ),
        layout="stereo",
        output_format="mp3",
        bitrate="320k",
        also_produce_stereo_mp3=False,
        final_lufs=-16.0,          # more dynamic range suits a proper room + real speakers
        headroom_db=0.5,
        auto_loudness=False,
        bass_shelf_hz=90.0,
        bass_shelf_db=0.0,         # the sub + bass management handles this
        treble_shelf_hz=9000.0,
        treble_shelf_db=0.5,
        presence_hz=0.0,           # full-range stereo speakers rarely need a presence crutch
        presence_db=0.0,
        notch_hz=0.0,
        multiband_compress=True,
        mb_low_ratio=1.6,
        mb_mid_ratio=1.4,
        mb_high_ratio=1.4,
        mb_low_crossover_hz=200,
        mb_high_crossover_hz=4000,
        saturation=False,
        crystalizer=False,
        crossfeed=False,
        width_bands=True,
        width_bass=1.0,
        width_mid=1.1,
        width_treble=1.2,
    ),

    "home_theatre_5_1": DeviceProfile(
        key="home_theatre_5_1",
        display_name="Home Theatre (5.1 Surround)",
        description=(
            "5.1 AV receiver + speaker system. Outputs an MKV with AC3 5.1 "
            "audio, plus a stereo MP3 alongside. Many receivers in this "
            "class apply their own room correction / auto-EQ (Audyssey, "
            "YPAO, and similar) — this profile keeps source-side "
            "processing minimal and dynamic range wide so it doesn't "
            "fight whatever the receiver does on its own."
        ),
        layout="5.1",
        output_format="mp3",
        bitrate="320k",
        also_produce_stereo_mp3=True,
        surround_codec="ac3",
        lfe_mode="gentle",
        lfe_cutoff_hz=80,
        rear_cutoff_hz=300,
        rear_attenuation_db=-6.0,
        rear_delay_ms=15.0,
        centre_attenuation_db=-3.0,
        final_lufs=-16.0,          # wider dynamic range suits a proper HT room
        headroom_db=0.5,
        auto_loudness=False,
        bass_shelf_hz=100.0,
        bass_shelf_db=0.5,         # modest — the sub handles the rest via bass management
        treble_shelf_hz=8000.0,
        treble_shelf_db=0.5,
        presence_hz=0.0,           # no vocal presence boost — the centre channel handles it
        presence_db=0.0,
        notch_hz=0.0,
        multiband_compress=True,
        mb_low_ratio=1.6,
        mb_mid_ratio=1.4,
        mb_high_ratio=1.4,
        mb_low_crossover_hz=200,
        mb_high_crossover_hz=4000,
        saturation=False,          # avoid stacking with whatever the receiver's own DSP does
        crystalizer=False,
        crossfeed=False,
        width_bands=True,
        width_bass=1.0,            # bass stays centred (bass management handles it)
        width_mid=1.15,
        width_treble=1.25,
    ),

    "headphones": DeviceProfile(
        key="headphones",
        display_name="Headphones",
        description=(
            "Optimised for headphone listening. Crossfeed reduces fatigue "
            "from hard-panned stereo mixes. Slight bass lift compensates "
            "for typical headphone bass thinness (especially open-back)."
        ),
        layout="stereo",
        output_format="mp3",
        bitrate="320k",
        also_produce_stereo_mp3=False,
        final_lufs=-14.0,
        headroom_db=0.5,
        auto_loudness=False,
        bass_shelf_hz=60.0,
        bass_shelf_db=1.5,
        treble_shelf_hz=10000.0,
        treble_shelf_db=0.5,
        presence_db=0.0,
        notch_hz=0.0,
        multiband_compress=False,
        saturation=False,
        crystalizer=False,
        crossfeed=True,
        crossfeed_strength=0.3,
        crossfeed_range=0.5,
        width_bands=True,
        width_bass=1.0,
        width_mid=1.6,
        width_treble=1.8,
    ),

    "general": DeviceProfile(
        key="general",
        display_name="General / Streaming",
        description=(
            "Neutral master for any playback system or streaming upload. "
            "Targets -14 LUFS (Spotify / Apple Music / YouTube Music "
            "standard). Minimal device-specific EQ coloration — use this "
            "when you're not sure what the audio will play on, or when "
            "uploading somewhere that will re-encode/normalise anyway."
        ),
        layout="stereo",
        output_format="mp3",
        bitrate="320k",
        also_produce_stereo_mp3=False,
        final_lufs=-14.0,
        headroom_db=0.5,
        auto_loudness=True,
        bass_shelf_db=0.0,
        treble_shelf_db=0.0,
        presence_db=0.0,
        notch_hz=0.0,
        multiband_compress=False,
        saturation=False,
        crystalizer=False,
        crossfeed=False,
        width_bands=False,
    ),
}