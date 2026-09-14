# Stage-2 terminal distillation

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
and shape metadata in the existing `PrecomputedDataset` layout.

Capture into scratch or a managed dataset volume. Projected 1024-token video
and audio conditioning is about 12 MiB per unique prompt before latent data,
so do not place a large trajectory set in a model cache.

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

Start with rank-8 Q/K/V adapters as a capacity probe. Do not expose a trained
adapter as a fast inference tier until it passes paired teacher/student video,
audio, synchronization, and latency evaluation. A lower training loss alone
is not a quality gate.

## MZR-3 envelope

On the 48 GiB M4 Pro test host, rank-8 Q/K/V backward with gradient
checkpointing measured 21.52 GiB at 468 video tokens, 31.63 GiB at 3072, and
50.10 GiB at 6144. Use a 468 -> 1536 -> 3072-token curriculum and reserve
6144-token full-resolution generation for validation.
