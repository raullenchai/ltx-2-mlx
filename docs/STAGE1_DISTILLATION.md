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

That first coupling is retained as `lane` v1 for reproducibility, but it only
carries lane 0 across the teacher span `0 -> 1 -> 2 -> 3`; lanes 1 and 2 remain
hidden target randomness. The `span-v2` experiment instead propagates and
combines every seeded lane covered by a coarse transition. Its weights are the
exact accumulated stochastic coefficients of the fine ancestral Euler steps
when denoised predictions are held fixed, divided by the coarse step's noise
coefficient. Training and inference use the identical coupling. This preserves
the fine path's random forcing without adding model evaluations, but it does
not make the nonlinear teacher drift analytically equivalent.

This follows the motivation of [stochastic consistency
distillation](https://arxiv.org/abs/2403.01505): controlled SDE noise can repair
accumulated approximation error and preserve diversity, while excessive noise
can destabilize training. It remains experimental until decoded multi-seed
qualification passes.

A 30-iteration materialized microbenchmark on an M3 Ultra Studio with MLX
0.32.0 used video/audio noise shapes `(1, 3072, 128)` and `(1, 126, 128)`.
Legacy one-lane generation averaged 0.407 ms and the three-lane `0 -> 3` span
averaged 0.592 ms, a 0.185 ms delta. This isolates coupling overhead; it is not
an end-to-end speed result.

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

Run the complete-noise coupling in a separate output root so v1 and v2
checkpoints cannot be confused:

```bash
python scripts/train_stage1_compressed_curriculum.py \
  --model /path/to/ltx-2.5-mlx-q8/snapshot \
  --data /path/to/training/stage1-trajectories \
  --output-root /private/tmp/LTX-stage1-compressed-span-v2 \
  --noise-coupling span-v2
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

For a qualified span-v2 checkpoint, add `--noise-coupling span-v2`. The
packager requires a durable `span-v2` curriculum marker and writes the full
fine sigma schedule plus contiguous noise spans into the artifact. The loader
rejects a relabeled v1 checkpoint, missing or non-contiguous spans, and any
coarse/fine schedule mismatch.

The original `7 -> 8` transition already uses one base evaluation and has no
compression benefit. If a shared adapter perturbs that nearly exact terminal
step, package the replay `5 -> 7` checkpoint with
`--clean-final-transition`. This emits a separate
`ltx_stage1_compressed_span_v2_clean_final` capability. Runtime applies the
student only to `0 -> 3`, `3 -> 5`, and `5 -> 7`, materializes both modalities,
releases the adapter, and executes the unchanged `7 -> 8` transition on the
clean base. The schedule remains four evaluations; only model reload overhead
is added. Do not relabel an ordinary span-v2 package as clean-final.

The opt-in runtime is `--distilled --fast-stage1-manifest fast-stage1.json`.
It fails closed on a malformed or mismatched package and preserves the
original Stage-1 behavior when the flag is absent. A separately qualified
Stage-2 package can be composed with `--fast-stage2-manifest`; both packages
must bind to the same base transformer. Neither package is enabled by default.

### Segmented adapters for transition interference

A shared adapter can improve the first three transitions in aggregate while
still producing a visibly broken decoded sample. Per-transition evaluation is
the required diagnostic: if checkpoints specialize strongly to their own
sigma region and regress on later regions, treat that as weight interference,
not as evidence that the compressed schedule itself is invalid.

The segmented experiment binds a distinct span-v2 adapter to each learned
transition (`0 -> 3`, `3 -> 5`, and `5 -> 7`) and always executes `7 -> 8`
with the unchanged base. Distinct runtime files are insufficient: every span
must be trained from the same clean base, not resumed from the preceding
span. Train all three independent adapters with:

```bash
python scripts/train_stage1_segmented_curriculum.py \
  --model /path/to/ltx-2.5-mlx-q8/snapshot \
  --data /path/to/training/stage1-trajectories \
  --output-root /private/tmp/LTX-stage1-segmented
```

Use `--span 0-3` (or another bound span) for a targeted data/objective
control. This is useful on constrained hosts because it retains only the final
checkpoint for each selected span. Evaluate the complete set only on its bound
transitions:

```bash
python scripts/evaluate_stage1_segmented_curriculum.py \
  --model /path/to/ltx-2.5-mlx-q8/snapshot \
  --checkpoint /private/tmp/LTX-stage1-segmented/primary-0-3/checkpoints/lora_weights_step_00100.safetensors \
  --checkpoint /private/tmp/LTX-stage1-segmented/primary-3-5/checkpoints/lora_weights_step_00080.safetensors \
  --checkpoint /private/tmp/LTX-stage1-segmented/primary-5-7/checkpoints/lora_weights_step_00080.safetensors \
  --data /path/to/held-out/stage1-trajectories \
  --output /private/tmp/LTX-stage1-segmented/evaluation.json
```

Package the three source checkpoints in schedule order:

```bash
python scripts/package_segmented_fast_stage1.py \
  --model-dir /path/to/immutable/model/snapshot \
  --checkpoint /path/to/qualified-0-3.safetensors \
  --checkpoint /path/to/qualified-3-5.safetensors \
  --checkpoint /path/to/qualified-5-7.safetensors \
  --output-dir /path/to/package \
  --base-model-id owner/model \
  --base-revision 0123456789abcdef0123456789abcdef01234567 \
  --qualification-revision qual-segmented-v1
```

The packager rejects checkpoints without clean-base `independent` training
provenance, in addition to failing closed on adapter order, source span
metadata, rank/shape, artifact digest, immutable base identity, exact sigma
schedule, and runtime contract. This prevents cumulative shared-curriculum
checkpoints from being mislabeled as independent segment experts. The normal
package copies adapters. A
`--symlink-adapters` mode exists only for scratch diagnostics whose
qualification revision starts with `diagnostic-`; it must never be distributed.

After placing the manifest and three adapter files beside the bound model,
invoke it with:

```bash
python -m ltx_pipelines_mlx.cli generate \
  --model /path/to/model-with-packages \
  --distilled \
  --fast-stage1-segmented-manifest fast-stage1-segmented.json \
  --fast-stage2-manifest fast-stage2.json \
  ...
```

The shared and segmented Stage-1 flags are mutually exclusive. Segmentation
does not change arithmetic by Mac generation: it loads model-compatible
artifacts by content contract on any supported Apple Silicon machine. It does
add three transformer reload boundaries, so only an end-to-end benchmark can
establish whether four evaluations still deliver the target speedup.

After placing both qualified manifests and adapters in the immutable model
directory, measure the composed 4+1 path against the unchanged 8+3 path with
fresh CLI processes, identical prompts and seeds, and randomized run order:

```bash
python scripts/render_combined_fast_blind_suite.py \
  --model /path/to/model-with-qualified-packages \
  --fast-stage1-manifest fast-stage1.json \
  --fast-stage2-manifest fast-stage2.json \
  --runtime-revision "$(git rev-parse HEAD)" \
  --prompts packages/ltx-trainer/configs/stage2_terminal_pilot_prompts.txt \
  --output-dir /private/tmp/ltx-combined-fast-blind \
  --width 768 --height 512 --frames 241 --frame-rate 24
```

For the segmented candidate, replace `--fast-stage1-manifest` with
`--fast-stage1-segmented-manifest fast-stage1-segmented.json`.

The runner is resumable per completed case. It keeps role mappings and timing
results in dotfiles so reviewers do not learn which side is fast. Export only
the anonymous review surface with `export_blind_review_bundle.py`, record the
human judgments, and reveal `.benchmark-results.json` only afterward. The
reported speedup is full wall-clock time, including model loading, text
encoding, both denoising stages, decoding, and muxing.

## Acceptance

Do not productize a Stage-1 checkpoint on training loss, latent MSE or chord
error alone. Reject it for visible blur, reduced motion amplitude, missing
subjects, narrower seed diversity, altered audio content, or worse audio/video
synchronization. If noise-coupled supervised transitions still narrow the
distribution, escalate to stochastic consistency or distribution matching
rather than increasing LoRA rank against an unsuitable objective.

Training coverage is part of this gate. The first diagnostic cohort contained
only eight trajectories from four duplicated prompts and no human face or
speech. A three-adapter candidate trained on that cohort reached 2.079x at
768x512x241 but omitted the requested front-facing speaker in a chef case; it
is rejected. Before changing the loss, use a prompt-disjoint broad control
covering faces, speech, hands, fast motion, camera motion, texture, low light,
ambience, and silence. Keep the failing prompt and seed out of training so the
decoded check remains held out.
