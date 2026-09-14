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
boundaries `0 -> 3`. It reproduces the original seeded noise lane, removes that
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

On the initial eight training and two prompt-disjoint validation trajectories,
both splits independently selected boundaries `[0, 3, 5, 7, 8]`, or sigmas
`[1.0, 0.98125, 0.909375, 0.421875, 0]`. This `3 + 2 + 2 + 1` grouping is the
first four-evaluation candidate; the agreement is useful evidence but the
sample count remains too small to serve as a quality gate.

## Train and evaluate the complete candidate

After the single `0 -> 3` noise-coupled pilot passes, train one shared adapter
over all four selected transitions:

```bash
python scripts/train_stage1_compressed_curriculum.py \
  --model /path/to/ltx-2.5-mlx-q8/snapshot \
  --data /path/to/training/stage1-trajectories \
  --output-root /private/tmp/LTX-stage1-compressed
```

The runner uses a fresh process per phase, requires every handoff checkpoint,
and follows the primary pass with low-learning-rate replay over all four
transitions. Sequential replay is only a mitigation for forgetting, not proof
that it is absent. Evaluate every transition independently on held-out data:

```bash
python scripts/evaluate_stage1_compressed_curriculum.py \
  --model /path/to/ltx-2.5-mlx-q8/snapshot \
  --checkpoint /private/tmp/LTX-stage1-compressed/replay-7-8/checkpoints/lora_weights_step_00024.safetensors \
  --data /path/to/held-out/stage1-trajectories \
  --output /private/tmp/LTX-stage1-compressed/evaluation.json
```

Only a decoded-qualified checkpoint may be packaged. The package binds the
adapter to an immutable base revision, transformer/config fingerprint,
four-sigma-transition schedule, original eight-step noise lanes, artifact
digest, and qualification revision:

```bash
python scripts/package_fast_stage1.py \
  --model-dir /path/to/immutable/model/snapshot \
  --checkpoint /path/to/qualified/lora.safetensors \
  --output-dir /path/to/package \
  --base-model-id owner/model \
  --base-revision 0123456789abcdef0123456789abcdef01234567 \
  --qualification-revision qual-v1
```

The opt-in runtime is `--distilled --fast-stage1-manifest fast-stage1.json`.
It fails closed on a malformed or mismatched package and preserves the
original Stage-1 behavior when the flag is absent. A separately qualified
Stage-2 package can be composed with `--fast-stage2-manifest`; both packages
must bind to the same base transformer. Neither package is enabled by default.

## Acceptance

Do not productize a Stage-1 checkpoint on training loss, latent MSE or chord
error alone. Reject it for visible blur, reduced motion amplitude, missing
subjects, narrower seed diversity, altered audio content, or worse audio/video
synchronization. If noise-coupled supervised transitions still narrow the
distribution, escalate to stochastic consistency or distribution matching
rather than increasing LoRA rank against an unsuitable objective.
