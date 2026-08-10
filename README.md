# OpenRemaster

**A guided, profile-based tool for restoring and optimizing your personal audio
collection for the gear you actually own.**

Most old recordings — film songs, classical performances, cassette rips,
early digital transfers — were mastered decades ago for equipment nobody
uses anymore. OpenRemaster takes a recording and a description of where
you'll play it back, and produces a version tuned for that combination.

You don't need to know what a multiband compressor or a Linkwitz-Riley
crossover is. You pick two things — **what kind of speaker you're playing
on** and **what era/type of recording this is** — and the wizard does the
rest, with every setting visible and adjustable if you want to go deeper.

---

## Why this exists

There are already good open-source tools for pieces of this problem:

- **[Matchering](https://github.com/sergree/matchering)** — reference-based
  mastering (feed it two tracks, it matches EQ/loudness between them).
- **[Demucs](https://github.com/facebookresearch/demucs)** — state-of-the-art
  stem separation, which OpenRemaster uses directly rather than reinventing.
- **[pyaudiorestoration](https://github.com/HENDRIX-ZT2/pyaudiorestoration)**,
  **cathar**, **GWC** — declick/denoise/dehum restoration for tape, vinyl,
  and cassette sources.
- **iZotope RX / Acon Digital** — the professional (paid) quality bar for
  restoration, built for engineers, not casual listeners.

What none of these do is **combine "what does this playback system need"
with "what does this recording need" into one resolved, guided decision**,
aimed at someone who just wants to point at an old MP3 and a speaker and
get something good — not tune 40 DSP parameters by hand.

That combination is the actual point of this project:

- **Device profiles** — Bluetooth speaker, 2.0/2.1/2.2 soundbar, 5.1 home
  theatre, headphones, general streaming — each with sane, hardware-class
  defaults (not tuned to one specific product).
- **Content profiles** — classic analogue, cassette-era, early-digital,
  modern-mastered, classical, devotional/folk, background score — each
  addressing what the *source recording* needs, independent of language
  or genre.
- **A resolver**, not a preset list — device and content profiles merge
  additively (e.g. a soundbar's treble shelf + a tape recording's
  rolloff-restoration shelf combine, rather than one overriding the other),
  so every device × content combination gets sensible output without
  needing its own hand-tuned entry.
- **A guided wizard**, not a DAW plugin — source → target system → content
  type (auto-detected, always overridable) → enhancement options → live
  DSP review with an EQ curve and A/B preview → export. Built for someone
  who has never opened an audio editor.
- **Stereo-to-5.1 upmix** — for people with a home theatre system, deriving
  a synthetic centre/rear/LFE layout from a plain stereo source, which is
  uncommon in free tooling.

If you're comfortable in a DAW or already know Matchering/RX, this project
probably won't replace your workflow. It's aimed at the much larger group
of people with a folder of old family recordings and a speaker they bought
off Amazon, who just want it to sound right.

---

## How it works

1. **Source & Output** — pick a file or folder, set output format.
2. **Target System** — what will you play this on? (Bluetooth speaker,
   soundbar, home theatre, headphones, general/streaming.)
3. **Content Type** — auto-detected from the audio itself (spectral
   analysis) and filename/tag hints, shown with a confidence score;
   override anytime.
4. **Enhancement** — optional stem separation (vocals/instruments) and
   experimental AI tools, off by default.
5. **DSP Review** — every parameter the first four steps resolved to,
   shown as readable sliders with a live EQ curve and A/B preview against
   the original. Nothing is hidden.
6. **Output & Run** — batch processing, per-track or album-consistent
   loudness, progress log, JSON report.

Device and content profiles are plain data (`device_profiles.py` /
`content_profiles.py`) — adding a new speaker category or recording era
is a matter of adding a profile, not writing new code.

---

## Status

Early / actively generalizing. The DSP engine (EQ, multiband compression,
de-essing, stereo widening, 5.1 upmix, loudness matching) is stable; the
profile set is being broadened from a personal setup to general-purpose
categories. Contributions of calibration data (noise floor / crest factor
/ spectral characteristics from your own recordings) and new device
profiles are especially welcome — see `CONTRIBUTING.md`.

## Requirements

- Python 3.9+, ffmpeg/ffprobe on PATH
- Recommended: numpy, scipy, pyloudnorm, soundfile, sounddevice, mutagen
- Optional: demucs (stem separation), voicefixer / deepfilternet
  (experimental, off by default)

## License

TBD
