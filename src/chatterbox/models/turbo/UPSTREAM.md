# Turbo and Nano source

These modules and `../../tts_turbo.py` derive from Resemble AI's Chatterbox
commit `5de7a54aa4e5e2baadb0182dde554908b48b85c2`:
https://github.com/resemble-ai/chatterbox/commit/5de7a54aa4e5e2baadb0182dde554908b48b85c2

The repository's MIT license and the Apache-2.0 notices in individual files
apply. The source keeps the upstream inference algorithms. Relative imports
point to this namespace for the GPT2 configuration and mean-flow decoder,
and to existing modules for unchanged support code. The T3 training and
legacy Llama inference methods are omitted.

Keeping these modules separate preserves the `faster` branch's original and
multilingual CUDA graph paths and its S3Gen API. Future upstream updates must
check both model families before consolidating their implementations.

Turbo was released with Transformers 4.46.3. This fork keeps that version for
compatibility with its custom Llama implementation. Nano uses the same GPT2
API with a smaller configuration. `pyloudnorm` normalizes reference audio.
