# Stage-2 transition distillation

This research path trains one LTX-2.5 transformer evaluation to reproduce the
terminal video and audio latents of the existing deterministic three-step
stage-2 refiner. It does not change the default inference schedule.

## Capture teacher trajectories

Prepare a text file with one prompt per line, then run:

```bash
PYTHONPATH=packages/ltx-core-mlx/src:packages/ltx-pipelines-mlx/src:packages/ltx-trainer/src \
python scripts/capture_stage2_trajectories.py \
  --model /path/to/ltx-2.5-mlx-q8/snapshot \
  --prompts prompts.txt \
  --output /path/to/stage2-trajectories \
  --width 192 --height 192 --frames 97 --frame-rate 24 \
  --seed 42 --low-ram-streaming
```

The command runs the normal eight-step ancestral stage 1 and deterministic
three-step stage 2, but does not decode video. Each item stores the exact BF16
stage-2 start and terminal latents, projected text conditioning, sigma, seed,
and shape metadata in the existing `PrecomputedDataset` layout. New captures
also store the teacher state at sigma `0.421875`, immediately before the final
teacher refinement step.

Capture into scratch or a managed dataset volume. Projected 1024-token video
and audio conditioning is about 12 MiB per unique prompt before latent data,
so do not place a large trajectory set in a model cache.

### Frozen scale experiment

The checked-in `stage2_progressive_scale_manifest.jsonl` freezes 100 requests
at the prompt level: 80 train and 20 validation trajectories from 40/10
disjoint prompts, with two seeds per prompt. Its bucket distribution is 60 at
468 video tokens, 30 at 1536, and 10 at 3072. Rebuild and verify it with:

```bash
python scripts/build_stage2_scale_manifest.py \
  --output packages/ltx-trainer/configs/stage2_progressive_scale_manifest.jsonl \
  --overwrite
```

Capture buckets independently into the same scratch output. Global manifest
indices make every command safely resumable and prevent collisions:

```bash
for bucket in 468 1536 3072; do
  python scripts/capture_stage2_trajectories.py \
    --model /path/to/ltx-2.5-mlx-q8/snapshot \
    --manifest packages/ltx-trainer/configs/stage2_progressive_scale_manifest.jsonl \
    --bucket "$bucket" \
    --output /private/tmp/LTX-progressive-scale/trajectories \
    --low-ram-streaming
done
```

After all 700 component files exist, create hard-linked split views without
duplicating latent storage:

```bash
python scripts/split_stage2_manifest_dataset.py \
  --manifest packages/ltx-trainer/configs/stage2_progressive_scale_manifest.jsonl \
  --data /private/tmp/LTX-progressive-scale/trajectories \
  --output /private/tmp/LTX-progressive-scale/splits
```

Do not construct validation by trajectory index alone: both seeds for a prompt
must remain in the same split to avoid prompt leakage.

## Train the student

Copy `packages/ltx-trainer/configs/stage2_terminal_distill.yaml`, update the
model, dataset, and output paths, then run the normal trainer command:

```bash
ltx-2-mlx train --config /path/to/stage2_terminal_distill.yaml
```

The strategy fixes sigma at `0.909375` and constructs the velocity target

```text
velocity_target = (stage2_start - teacher_terminal) / sigma
```

so a single Euler step to sigma zero lands on the teacher terminal. Both video
and audio losses are enabled. Saved checkpoints include
`distillation=stage2_terminal`, `stage2_sigma=0.909375`, and
`stage2_steps=1` safetensors metadata. LoRA checkpoints also retain `lora_rank`
and `lora_alpha`, so fusion uses the same scale as training.

This is a terminal jump with the exact schedule `[0.909375, 0.0]`. The normal
`stage2_steps=1` pipeline option only truncates the original schedule to
`[0.909375, 0.725]`; it is not interchangeable with this checkpoint.

For the quality-preserving progressive experiment, copy
`stage2_progressive_distill.yaml`. It trains one evaluation to replace the
first two teacher evaluations:

```text
velocity_target = (stage2_start - teacher_intermediate) / (0.909375 - 0.421875)
product schedule = [0.909375, 0.421875, 0.0]
```

The final `0.421875 -> 0` evaluation remains the unchanged base transformer.
This reduces stage 2 from three evaluations to two while retaining a teacher
correction step. Checkpoint metadata records the target sigma and target data
directories so evaluation cannot silently compare against terminal latents.

## Evaluate the student

Evaluate on held-out teacher trajectories before integrating inference:

```bash
PYTHONPATH=packages/ltx-core-mlx/src:packages/ltx-pipelines-mlx/src:packages/ltx-trainer/src \
python scripts/evaluate_stage2_distillation.py \
  --model /path/to/ltx-2.5-mlx-q8/snapshot \
  --checkpoint /path/to/lora_weights_step_00500.safetensors \
  --data /path/to/held-out-trajectories \
  --output /path/to/metrics.json
```

The evaluator fuses the adapter only into a standalone stage-2 transformer,
runs exactly one model evaluation, and reports paired video/audio latent MSE,
MAE, relative RMSE, cosine similarity, and latency. This prevents the
stage-2-only adapter from changing stage 1. Use
`configs/stage2_terminal_pilot_prompts.txt` as the initial motion, speech,
impact, ambience, and synchronization coverage set.

Omit `--checkpoint` to measure the unadapted one-evaluation baseline against
the same teacher trajectories.

For the progressive baseline, additionally pass `--target-sigma 0.421875`,
`--video-target-dir stage2_video_intermediate_latents`, and
`--audio-target-dir stage2_audio_intermediate_latents`. The renderer detects
the progressive metadata, runs the distilled transition, reloads the clean
base transformer for the final teacher step, and decodes the true terminal
teacher/student pair.

Start with rank-8 Q/K/V adapters as a capacity probe. Do not expose a trained
adapter as a fast inference tier until it passes paired teacher/student video,
audio, synchronization, and latency evaluation. A lower training loss alone
is not a quality gate.

## MZR-3 envelope

On the 48 GiB M4 Pro test host, rank-8 Q/K/V backward with gradient
checkpointing measured 21.52 GiB at 468 video tokens, 31.63 GiB at 3072, and
50.10 GiB at 6144. Use a 468 -> 1536 -> 3072-token curriculum and reserve
6144-token full-resolution generation for validation.

## Related MLX runtime survey

On 2026-09-13, source inspection covered `mlx-vlm` commit
`45d6e125ab174cc279edea417f6be734870ff161` and `omlx` commit
`ea974149a2e6b94302cc313752402a14f73d447f`.

- Neither repository contains an LTX/Lightricks model implementation or an
  LTX-specific kernel path.
- `mlx-vlm` diffusion support is for image models. Its attention path uses
  MLX fast scaled-dot-product attention, which LTX already uses.
- `omlx` native paths are specialized for autoregressive LLM/MLLM prefill,
  query-length-one decode, Qwen shapes, MoE, or ANE offload. Those dispatch
  assumptions do not map directly to LTX's long bidirectional audio/video
  diffusion sequences.
- The reusable ideas are engineering patterns: fail-closed shape/model
  dispatch, ABI probes, an exact MLX fallback, and explicit stream/memory
  lifecycle. They may de-risk a future hotspot kernel, but do not provide the
  several-fold gain targeted here.

The highest-leverage route therefore remains fewer transformer evaluations
with a perceptual quality gate, not porting an LLM-serving kernel wholesale.
