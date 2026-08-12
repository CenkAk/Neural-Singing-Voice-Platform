# Audio Pipeline

- A waveform is sampled air-pressure amplitude. NSVP stores it as `float32 [channels, samples]`.
- Sample rate controls time resolution. Training/SVC uses 44.1 kHz; TorchCREPE analysis uses 16 kHz internally.
- STFT splits audio into overlapping frames and FFT frequency bins. Demucs and neural vocoders use spectral information internally; the core mixer remains waveform-domain.
- F0 is perceived pitch, not every harmonic. Harmonics are integer-related spectral components; formants are vocal-tract resonances that help identify a singer.
- Clipping occurs when samples exceed the export range. `mix_audio()` applies equal headroom to the completed mix instead of independently crushing dynamics.
- LUFS/true-peak require optional audio measurement dependencies and are not fabricated when unavailable.

Preprocessing uses controlled mono downmix, DC removal, resampling, and silence-aware segmentation. It deliberately avoids default loudness normalization, denoising, aggressive compression, and breath removal because those operations can erase singing expression.

Source separation may leave bleed, reverb, doubles, and backing vocals. V1 converts the combined vocal stem. Neutral post-processing currently removes DC and protects mix headroom; unsupported “studio” controls are not exposed.

