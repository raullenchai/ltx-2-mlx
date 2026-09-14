# Stage-1 ancestral distillation

The LTX-2.5 distilled pipeline uses eight ancestral Stage-1 transformer
evaluations. Reaching the project `4 + 1` target requires compressing Stage 1
to four evaluations while retaining its stochastic motion and diversity.

## Capture

Capture all nine original boundaries without running Stage 2 or decoding:

```bash
python scripts/capture_stage1_trajectories.py \
  --model /path/to/ltx-2.5-mlx-q8/snapshot \
  --manifest /path/to/manifest.jsonl \
  --bucket 3072 \
  --output /private/tmp/LTX-stage1-trajectories \
  --low-ram-streaming
```

Each boundary carries the generation seed, ancestral noise seed, original step
index, sigma, video/audio latents and prompt conditioning. Capture is resumable
and refuses partial items.

## Two distinct experiments

`stage1_transition_distill.yaml` maps captured boundaries `6 -> 8` directly to
sigma zero. This deliberately difficult deterministic endpoint pilot is only a
capacity/failure probe. The teacher path includes an intermediate random draw
that a terminal student cannot inject. Better latent MSE does not establish
motion or diversity preservation.

`stage1_stochastic_transition_distill.yaml` instead maps non-terminal
boundaries `0 -> 2`. It reproduces the original seeded noise lane, removes that
known runtime noise from the supervised velocity target, and re-injects it in
the ancestral evaluator. A noise-coupled transition is forbidden from targeting
sigma zero.

Evaluate either checkpoint against prompt-disjoint trajectories:

```bash
python scripts/evaluate_stage1_distillation.py \
  --model /path/to/ltx-2.5-mlx-q8/snapshot \
  --transformer-file transformer-distilled.safetensors \
  --checkpoint /path/to/lora.safetensors \
  --data /path/to/held-out/stage1-trajectories \
  --output /path/to/result.json
```

The report compares the student with the unadapted base using the same runtime
transition. It emits `null`, rather than invalid JSON, if a baseline error is
exactly zero.

## Select the four-evaluation schedule

Rank boundary combinations from captured path geometry:

```bash
python scripts/analyze_stage1_schedule.py \
  --data /path/to/held-out/stage1-trajectories \
  --output /path/to/schedule-ranking.json
```

The analyzer preserves the original final `0.421875 -> 0` correction and ranks
all four-evaluation candidates by video/audio chord error. This is diagnostic:
the chosen schedule still requires end-to-end decoded blind tests across two
seeds per high-risk prompt.

## Acceptance

Do not productize a Stage-1 checkpoint on training loss, latent MSE or chord
error alone. Reject it for visible blur, reduced motion amplitude, missing
subjects, narrower seed diversity, altered audio content, or worse audio/video
synchronization. If noise-coupled supervised transitions still narrow the
distribution, escalate to stochastic consistency or distribution matching
rather than increasing LoRA rank against an unsuitable objective.
