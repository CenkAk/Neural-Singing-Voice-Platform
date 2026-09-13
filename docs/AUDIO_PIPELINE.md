# Audio Pipeline

- A waveform is sampled air-pressure amplitude. NSVP stores it as `float32 [channels, samples]`.
- Sample rate controls time resolution. Dataset preparation and Seed-VC default to 44.1 kHz; SoulX uses its model-configured rate (the documented SVC profile is 24 kHz). Adapters return audio at the source rate. Original model output is retained separately.
- STFT splits audio into overlapping frames and FFT frequency bins. Demucs and neural vocoders use spectral information internally; the core mixer remains waveform-domain.
- F0 is perceived pitch, not every harmonic. Harmonics are integer-related spectral components; formants are vocal-tract resonances that help identify a singer.
- Clipping occurs when samples exceed the export range. `mix_audio()` applies equal headroom to the completed mix instead of independently crushing dynamics.
- LUFS/true-peak require optional audio measurement dependencies and are not fabricated when unavailable.

Preprocessing uses controlled mono downmix, DC removal, resampling, and silence-aware segmentation. It deliberately avoids default loudness normalization, denoising, aggressive compression, and breath removal because those operations can erase singing expression.

Source separation may leave bleed, reverb, doubles and backing vocals. Default processing converts the combined vocal stem. Optional preprocessing stages run in configured order, with no-op as the default. Vocal preparation exposes sources for explicit selection, but no real multi-singer separator ships in v0.2. Neutral post-processing removes DC and protects mix headroom.

The dataset audit records voiced duration, pitch percentiles, MIDI register-bin counts, clipping/silence diagnostics, source contributions and split durations. Configurable warnings flag narrow coverage without inventing a required pitch range. Audit metadata does not change existing source-level splits or dataset version semantics.
