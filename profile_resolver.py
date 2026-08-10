"""profile_resolver.py — Merges a DeviceProfile and a ContentProfile into
a single flat parameter dict that the pipeline orchestrator consumes.

Kept as its own module (rather than folded into device_profiles.py or
content_profiles.py) so each profile module stays a pure, independent
data file with no cross-imports — device and content profiles can be
edited, tested, or swapped independently of each other.
"""

from __future__ import annotations

from content_profiles import CONTENT_PROFILES, ContentKey
from device_profiles import DEVICE_PROFILES, DeviceKey


def resolve_profile(
    device_key: DeviceKey,
    content_key: ContentKey,
) -> dict:
    """Merge device and content profiles into a single flat parameter dict
    for the pipeline orchestrator.

    Merge rules:
    - Device profile sets hardware delivery parameters (layout, codec,
      loudness target, hardware-compensating EQ).
    - Content profile sets source-material parameters (restoration,
      stem processing, content-correcting EQ).
    - Where both define the same parameter, content profile wins for
      dynamics/restoration (content knows the source); device profile
      wins for loudness targets and hardware EQ (device knows the
      destination).
    - Numeric EQ corrections are ADDITIVE: device bass_shelf_db +
      content bass_correction_db = final bass shelf applied.
    """
    dp = DEVICE_PROFILES[device_key]
    cp = CONTENT_PROFILES[content_key]

    p: dict = {}

    # --- Device: hardware delivery ---
    p["layout"]                  = dp.layout
    p["output_format"]           = dp.output_format
    p["bitrate"]                 = dp.bitrate
    p["also_produce_stereo_mp3"] = dp.also_produce_stereo_mp3
    p["surround_codec"]          = dp.surround_codec
    p["lfe_mode"]                = dp.lfe_mode
    p["lfe_cutoff_hz"]           = dp.lfe_cutoff_hz
    p["rear_cutoff_hz"]          = dp.rear_cutoff_hz
    p["rear_attenuation_db"]     = dp.rear_attenuation_db
    p["rear_delay_ms"]           = dp.rear_delay_ms
    p["centre_attenuation_db"]   = dp.centre_attenuation_db

    # --- Loudness: content override wins if set ---
    p["final_lufs"]    = cp.final_lufs_override  if cp.final_lufs_override  is not None else dp.final_lufs
    p["headroom_db"]   = cp.headroom_db_override if cp.headroom_db_override is not None else dp.headroom_db
    p["auto_loudness"] = cp.auto_loudness or dp.auto_loudness

    # --- EQ: additive combination ---
    p["bass_shelf_hz"]   = dp.bass_shelf_hz
    p["bass_shelf_db"]   = dp.bass_shelf_db + cp.bass_correction_db
    p["treble_shelf_hz"] = dp.treble_shelf_hz
    p["treble_shelf_db"] = dp.treble_shelf_db + cp.treble_correction_db
    p["presence_hz"]     = dp.presence_hz
    p["presence_db"]     = dp.presence_db + cp.vocal_presence_db
    p["presence_q"]      = dp.presence_q
    p["notch_hz"]        = dp.notch_hz
    p["notch_db"]        = dp.notch_db
    p["notch_q"]         = dp.notch_q

    # --- Dynamics: content-driven (it knows the source material) ---
    p["multiband_compress"] = cp.multiband_compress or dp.multiband_compress
    # If both define multiband, take the more aggressive ratios
    p["mb_low_ratio"]         = max(dp.mb_low_ratio,  cp.mb_low_ratio)
    p["mb_mid_ratio"]         = max(dp.mb_mid_ratio,  cp.mb_mid_ratio)
    p["mb_high_ratio"]        = max(dp.mb_high_ratio, cp.mb_high_ratio)
    p["mb_low_crossover_hz"]  = dp.mb_low_crossover_hz
    p["mb_high_crossover_hz"] = dp.mb_high_crossover_hz

    # --- Saturation: content wins (source material determines need) ---
    p["saturation"]          = cp.saturation or dp.saturation
    p["saturation_drive_db"] = cp.saturation_drive_db if cp.saturation else dp.saturation_drive_db
    p["saturation_mix"]      = cp.saturation_mix      if cp.saturation else dp.saturation_mix

    # --- Stereo / surround spatial ---
    p["width_bands"]            = dp.width_bands
    p["width_bass"]             = dp.width_bass
    p["width_mid"]               = dp.width_mid
    p["width_treble"]           = dp.width_treble
    p["crystalizer"]            = dp.crystalizer
    p["crystalizer_intensity"]  = dp.crystalizer_intensity
    p["crossfeed"]              = dp.crossfeed
    p["crossfeed_strength"]     = dp.crossfeed_strength
    p["crossfeed_range"]        = dp.crossfeed_range

    # --- Restoration (content only) ---
    p["declick"]           = cp.declick
    p["declick_threshold"] = cp.declick_threshold
    p["denoise"]           = cp.denoise
    p["denoise_amount"]    = cp.denoise_amount
    p["denoise_floor"]     = cp.denoise_floor
    p["denoise_type"]      = cp.denoise_type

    # --- Stem processing (content only) ---
    p["use_stems"]    = cp.use_stems
    p["use_demucs"]   = cp.use_demucs
    p["demucs_model"] = cp.demucs_model
    p["vocal_lufs"]   = cp.vocal_lufs
    p["music_lufs"]   = cp.music_lufs

    # --- Vocal chain ---
    p["vocal_presence_db"]    = cp.vocal_presence_db
    p["vocal_air_db"]         = cp.vocal_air_db
    p["vocal_mud_cut_db"]     = cp.vocal_mud_cut_db
    p["deesser"]              = cp.deesser
    p["deesser_threshold_db"] = cp.deesser_threshold_db
    p["deesser_ratio"]        = cp.deesser_ratio
    p["deesser_freq"]         = cp.deesser_freq

    # --- Instrument chain ---
    p["inst_bass_shelf_db"]    = cp.inst_bass_shelf_db
    p["inst_air_shelf_db"]     = cp.inst_air_shelf_db
    p["inst_harmonic_exciter"] = cp.inst_harmonic_exciter
    p["inst_exciter_freq"]     = cp.inst_exciter_freq
    p["inst_exciter_amount"]   = cp.inst_exciter_amount

    return p