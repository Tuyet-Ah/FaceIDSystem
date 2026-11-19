# AnomalyCLIP inference sources

Source: https://github.com/zqhang/AnomalyCLIP

Pinned commit: `3911738c0867544f545a076ad78f3f11d9ecbfdf`.

This directory contains the minimal inference library, CLIP tokenizer vocabulary,
and `prompt_ensemble.py` from the official AnomalyCLIP implementation. The
upstream MIT license is preserved as `LICENSE`. The underlying OpenAI CLIP MIT
notice is also included as `OPENAI_CLIP_LICENSE`.

Local compatibility changes:

- `pkg_resources.packaging` imports use the standalone `packaging` package.
- The prompt learner imports its tokenizer with a package-relative path.
- Learned deep text contexts follow the input tensor's device and dtype instead
  of unconditionally converting to FP16. CPU inference remains FP32.
- The unused `thop.profile` import is omitted from the inference model.
- The upstream author's hardcoded cache directory is replaced by the standard
  user cache fallback; the adapter passes a local checkpoint path and explicit
  model directory, so inference never downloads files.

The adapter follows `test_one_example.py`: 12 learned context tokens, 9 text
prompt levels, 4 tokens per deeper level, visual DPAM layer 20, intermediate
layers 6/12/18/24, softmax temperature 0.07, and Gaussian smoothing sigma 4.
Its returned map is the mean of the four layer anomaly maps, rather than their
sum, to retain a fixed 0–1 range. This division changes the scale, not ranking.
The image-level score is kept separate and is not a calibrated probability.

The official example uses a 518×518 image. The web additionally offers 336×336
for lower CPU cost. In both cases the complete input is resized to a square,
then the resulting anomaly map is mapped back to the original dimensions.
