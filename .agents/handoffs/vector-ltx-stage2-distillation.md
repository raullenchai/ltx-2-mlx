# Vector handoff: LTX-2.5 progressive stage-2 distillation

Receiving role: Atlas (architecture/inference integration), then Vector for
the scaled benchmark. Branch:
`vector/ltx25-distillation-feasibility` in `raullenchai/ltx-2-mlx`.

## Verified facts

- Direct stage-2 `3 -> 1` rank-8 distillation improved latent metrics but
  deleted a held-out moving bee. Rank-32 broad adapters regressed.
- Progressive `3 -> 2` learns `0.909375 -> 0.421875` and retains the unchanged
  base-model `0.421875 -> 0` correction.
- On the same 8-train / 4-held-out pilot, step 50 improves held-out video MSE
  from `0.029600` to `0.023035` (-22.2%) and audio MSE from `0.037557` to
  `0.014525` (-61.3%). Video/audio cosine are `0.981379` / `0.993464`.
- Paired decoded review preserves the central subject and coarse motion in
  fireworks, snow dog, locomotive, and bee/lavender samples. Visible detail
  and trajectory differences remain, so this is not release quality.
- Training took 4.6 minutes at 19.80 GiB peak on MZR-3. Full tests pass:
  `616 passed, 22 skipped` with `/opt/homebrew/bin` on `PATH`.
- A zero-shot 768x512 / 1536-token hummingbird probe also improved: video
  MSE `0.025395 -> 0.022698`, audio MSE `0.021937 -> 0.010367`, with the
  subject and motion preserved in the decode. One transformer evaluation was
  10.9-11.4 seconds at this shape.
- Source audits found no LTX-specific optimization in mlx-vlm or omlx. Their
  native LLM kernels do not match LTX sequence/model semantics.

## Artifacts

- Trajectories:
  `/Volumes/RTL-2T/datasets/ltx-stage2-distillation/progressive-pilot-2026-09-13/`
- Step-50 adapter:
  `/Volumes/RTL-2T/models-cold/ltx-stage2-distillation/progressive-pilot-2026-09-13/`
- 768x512 probe:
  `/Volumes/RTL-2T/datasets/ltx-stage2-distillation/progressive-hires-probe-2026-09-13/`
- Detailed metrics and decoded pairs are recorded outside this repository in
  `123/strategy/2026-09-13-ltx-stage2-distillation-pilot-results.md`.

## Risks and next action

The pilot is too small and low-resolution to establish perceptual equivalence
or full-resolution speed. Do not wire this checkpoint into default inference.
Vector should collect at least 100 balanced progressive trajectories across
468/1536/3072-token buckets, freeze a validation split, and repeat rank-8
training plus blinded decoded AV review. Atlas should review the eventual
checkpoint dispatch and ensure the adapter is active only for its distilled
transition, not stage 1 or the final base correction.

## Scale qualification (2026-09-14)

Commit `c475352` freezes a 100-trajectory prompt-disjoint manifest and adds
resumable bucket capture plus hard-linked split tooling. The distribution is
80 train / 20 validation and 60x468 / 30x1536 / 10x3072 video tokens.

MZR-3 completed capture into `/private/tmp/LTX-progressive-scale.zv79Uv`. The
fail-closed supervisor passed 420/630/700 file milestones. Split views contain
560 train and 140 validation components, and the 700-file source set is
archived at
`/Volumes/RTL-2T/datasets/ltx-stage2-distillation/progressive-scale-2026-09-14/`.

A shuffled mixed-shape process was OS-terminated before checkpoint without a
Python traceback, consistent with compiled graph/allocation accumulation. The
replacement curriculum uses separate processes: 100 steps at 468, 50 at 1536,
and 16 at 3072 tokens, with checkpoint handoff and descending learning rates.
The 468 stage is stable through step 5 at 12.63 seconds/step and loss `0.1746`.
`/private/tmp/LTX-training-curriculum-supervisor.sh` requires each final
checkpoint before advancing. After completion, evaluate all stage checkpoints
on the matching validation bucket and the final checkpoint across all 20
items. Never use validation prompts for optimization.

## Trainer and resume corrections (2026-09-14)

Two generic trainer defects were found while continuing at 3072 tokens:

- The loop materialized only `loss`, leaving lazy backward, clipping, and
  AdamW graphs connected. Materializing `(loss, grads)` at the phase boundary
  changed the real 3072-token run from two pre-step-1 OS kills to a complete
  16/16 run (18.6 minutes). The same fix completed a variable-shape mixed run.
- Saved adapters use diffusers keys and transposed tensors, but resume filtered
  only native lowercase keys. The path was logged while zero adapter tensors
  were actually loaded. Resume now converts `.lora_A.weight` and
  `.lora_B.weight` back to `.lora_a` and `.lora_b`, transposes them, and fails
  closed if no compatible tensors exist. A round-trip regression test covers
  the format. Python dataloader shuffling is now seeded alongside MLX.

After the resume fix, 24 low-learning-rate mixed-replay steps from the
1536-token step-50 checkpoint improved held-out video/audio MSE in every
bucket: 468 by 0.32%/0.54%, 1536 by 0.21%/1.14%, and 3072 by 0.16%/1.30%.
These are anti-regression results, not evidence of perceptual equivalence.
The valid candidate is
`output-rank8-mixed-replay-resume-fixed/checkpoints/lora_weights_step_00024.safetensors`.
Earlier `output-rank8-3072-graph-boundary` and
`output-rank8-mixed-replay-low-lr` checkpoints started from zero because of
the resume bug and must not be used.

A new blind 10-second decoded A/B reuses the previous prompt, seed, teacher
trajectory, and encode settings. Record human feedback before revealing the
mapping. If long-form detail or exposure still trails, add a long-temporal,
lower-spatial bucket whose total token count remains trainable; do not add
chip-model or installed-memory branches to product inference.

The human review subsequently passed: the reviewer reported that they could
not distinguish which side was better after the resume-fixed adjustment. The
reveal was A/left = mixed-replay step-24 student and B/right = three-evaluation
teacher. This reverses the prior result on the same prompt, where the teacher
was judged slightly brighter and more detailed. Treat it as a one-sample long
form perceptual pass, not a global default-on gate. Stage 2 remains measured at
about 206.7 seconds versus 301.5 seconds (31.4% lower latency, 1.46x); the
projected end-to-end gain remains about 1.20-1.25x because other stages are
unchanged.

## Compressed stage-1 noise compatibility (2026-09-14)

The four-transition stage-1 candidate uses teacher boundaries
`[0, 3, 5, 7, 8]`, corresponding to sigmas
`[1.0, 0.98125, 0.909375, 0.421875, 0.0]`. Both the eight-sample training set
and the two-sample prompt-disjoint validation set independently ranked this
schedule first by latent chord error. This ranking is diagnostic evidence,
not a perceptual quality gate.

`ancestral_denoise_loop` now accepts optional `noise_step_indices` and
`noise_total_steps` so a compressed schedule can reproduce the exact seeded
noise lanes used by the original eight-transition teacher. The candidate
mapping is `[0, 3, 5, 7]` with `noise_total_steps=8`. Omitting both arguments
preserves the existing sampler byte-for-byte; partial, wrong-length, and
out-of-range mappings fail closed. The runtime primitive is covered by the
full suite (`669 passed, 22 skipped`), but must remain unused in product
inference until a trained stage-1 artifact passes held-out latent evaluation,
blind decoded AV review, and cross-generation Apple Silicon qualification.

The reproducible shared-adapter prototype is
`scripts/train_stage1_compressed_curriculum.py`. It trains the four selected
transitions sequentially with their exact original noise lanes and then runs
a complete low-learning-rate replay pass over all four transitions. Each
phase runs in a fresh process, resumes from the prior adapter, writes an
atomic status, and requires the expected checkpoint before advancing. Existing
and newly written handoff checkpoints are metadata-validated against the exact
phase before use, so a stale schedule cannot silently resume. This
is intentionally not wired into a supervisor until the first `0 -> 3`
noise-coupled pilot passes held-out evaluation; every transition must then be
evaluated separately to detect catastrophic forgetting.
`scripts/evaluate_stage1_compressed_curriculum.py` loads the shared student
once, evaluates all four transitions against their captured held-out targets,
then repeats the same transitions with a clean base model and reports paired
video/audio MSE changes. Full-suite coverage after adding both tools is
`674 passed, 22 skipped`.

An opt-in Stage-1 product contract is now implemented for Atlas review. A
`fast-stage1.json` package binds the shared adapter to the immutable base
revision, transformer filename/config digest, exact five-sigma schedule,
original noise-lane indices, LoRA shape/scale, runtime major, artifact SHA-256,
and qualification revision. The distilled pipeline loads it only when
`--fast-stage1-manifest` is supplied, rejects additional LoRAs and schedule
overrides, and requires any composed Stage-2 package to target the same base.
The adapter is released before Stage 2 and cannot leak into a later request.
Absent the flag, the existing eight-step path is unchanged. This is a public
experimental API and default/marketing decisions remain Atlas-owned. It must
not be presented as available model functionality until a real shared adapter
passes all quality gates. Full-suite coverage for the contract and lifecycle
is `695 passed, 22 skipped`.

`scripts/analyze_blind_av_suite.py` computes per-case anonymous video
SSIM/PSNR and channel-wise audio APSNR without opening the hidden mapping. It
was checked against case 00 and reproduced SSIM `0.889730`, PSNR `28.984457`
dB, and audio APSNR `169.207`/`169.205` dB. The full suite now has `697`
passes and `22` skips.

The full progressive Stage-2 blind suite completed: 10/10 cases, each
768x512, 241 frames at 24 fps with 48 kHz stereo audio. Without opening the
mapping, aggregate diagnostics are video SSIM `0.924042`, video PSNR `32.504`
dB, and audio APSNR `167.519` dB. The lowest same-seed visual similarities are
case 03 (`0.840896`) and case 08 (`0.825009`), so human review should focus on
their camera/natural-motion behavior. Only A/B/side-by-side files, the review
index, and anonymous metrics were copied to
`123/strategy/ltx-stage2-qualification-2026-09-14/`; no named teacher/student
files or hidden mapping were copied. Human judgments are still required before
revealing the mapping or calling the ten-case suite a pass.

Post-render contact-sheet inspection found that the first review index was
wrong: `PrecomputedDataset` preserved filesystem glob order while the blind
runner labeled cases from a separately filename-sorted list. Teacher/student
pairs and their anonymous metrics were internally valid, but prompt/stress
labels were permuted. Dataset discovery is now sorted by relative filename,
and the blind runner derives both render positions and review prompts from the
same validated `PrecomputedDataset` ordering. The already-rendered progressive
suite did not need a costly rerender; its review index was corrected from the
exact pre-fix dataset order. The original incorrect index is retained as a
scratch backup. Do not use it. Full-suite coverage is `700 passed, 22 skipped`.

`scripts/run_stage1_compressed_qualification.py` now provides the next
fail-closed supervisor stage. It advances from the first noise-coupled pilot
only when both mean video and audio MSE improve by at least 10% and no held-out
sample regresses by more than 5%; passing this gate starts the full curriculum
and its four-transition evaluation. This gate authorizes compute only, not
product acceptance. The full suite is `699 passed, 22 skipped`.

## Terminal scale candidate (2026-09-14)

The terminal `3 -> 1` curriculum completed its 468/1536/3072-token phases and
a 24-step low-learning-rate replay without OOM. Replay took 10.7 minutes and
peaked at 19.79 GiB. On the prompt-disjoint 20-item held-out set, the final
checkpoint reduced mean video latent MSE by 23.1% and audio latent MSE by
74.7% versus the unadapted one-step baseline. All 20 paired items improved in
both modalities; the smallest video improvement was 5.2%. These are latent
diagnostics only, not decoded non-inferiority evidence.

A diagnostic package was built and successfully reopened by the real
fail-closed loader. It declares `ltx_stage2_terminal_v1`, schedule
`[0.909375, 0.0]`, runtime contract major 1, immutable base revision
`f1b56e7dc89f71a9af2cddac787b89ed22a8b7fc`, and qualification revision
`diagnostic-terminal-20260914`. Its adapter SHA-256 is
`6e8de0813b3731e1be5ccb66170e219bc1f6395052702a04f3a141cd98d749f1`.
The diagnostic qualification label deliberately prevents treating this as a
release-qualified artifact. A new 10-case 768x512 decoded blind suite is now
complete. Without opening its mapping, aggregate A/B similarity is video SSIM
`0.889767`, video PSNR `30.469` dB, and audio APSNR `167.780` dB. Cases 00-02
are the lowest visual-similarity group (`0.779-0.811`), followed by case 08
(`0.849`), and need focused human review. The terminal mean is below the
progressive suite's `0.924042` SSIM, so terminal remains a higher perceptual-
difference risk despite its stronger latent metrics.

The safe reviewer bundle is in
`123/strategy/ltx-stage2-terminal-qualification-2026-09-14/`: 10 cases, 32
files, about 128 MB. It contains no hidden mapping or teacher/student-named raw
media. The first remote analyzer supervisor failed because its non-interactive
`PATH` omitted `/opt/homebrew/bin` and could not resolve ffmpeg; media were
unaffected. Metrics were instead computed on Studio from an rsync allowlist of
anonymous files, without competing with the next MZR-3 GPU job. Future remote
analyzer supervisors must set `PATH=/opt/homebrew/bin:$PATH` explicitly.

The matched Stage-1 lane-v1 `0 -> 3` pilot is now running. Step 5 measured
17.76 seconds/step, projecting about 29 minutes for 100 steps before held-out
evaluation and the otherwise identical span-v2 pilot.

## Stage-1 coupling selection result (2026-09-14)

The matched 100-step pilots completed in 29.4 minutes each at 19.79 GiB peak.
The v1 single-lane control barely learned: loss moved from 5513 at step 5 to
5366 at step 100, and held-out video/audio MSE improved only 2.84%/1.69%.

Span-v2 changed only the noise coupling. Its loss moved from 334 at step 5 to
202 at step 100: 93.9% lower than v1 at step 5 and 96.2% lower at completion.
On the same prompt-disjoint held-out pair, video MSE changed
`0.06032 -> 0.04016` (-33.42%) and audio MSE changed
`0.05833 -> 0.03980` (-31.78%). Both samples improved in both modalities, so
the predeclared 10% mean / 5% worst-sample compute gate passed. This strongly
supports hidden fine noise as the v1 target pathology; it still does not prove
decoded perceptual non-inferiority.

The full span-v2 four-transition shared-adapter curriculum is now running.
Before it started, a fresh-phase handoff bug was fixed at `c30cb68`: the newly
trained path had omitted the selected coupling when validating a checkpoint
and would have rejected a valid span-v2 sampler under the default v1 rule.
Fresh and resumed phases now share one required-checkpoint validator. Full
tests pass: `736 passed, 22 skipped`.

## Clean-final and segmented Stage-1 results (2026-09-14)

The four-primary/four-replay curriculum completed. The shared adapter improved
the first three held-out transitions but regressed the already-uncompressed
`7 -> 8` step by 51834%/1525% video/audio MSE; its decoded 25-frame sample also
showed severe grayscale and structural collapse. The shared artifact is
rejected. A clean-final lifecycle retained the base `7 -> 8` evaluation and
measured 47.95 seconds versus standard 78.80 seconds at 768x512x25 (1.64x),
but did not repair decoded quality.

Per-transition evaluation then isolated strong sigma-region interference. The
primary `0 -> 3` checkpoint improved its own held-out video/audio MSE by
36.0%/31.3%, and the primary `3 -> 5` checkpoint improved its own region by
78.1%/78.8%; applying specialized weights outside their region produced
18-40% regressions.

Upstream `cc9091b` adds a segmented contract and lifecycle. Three exact
span-v2 adapters run only their bound `0 -> 3`, `3 -> 5`, and `5 -> 7`
transitions, preserving the same seeded span noise; the immutable clean base
runs `7 -> 8`. Each boundary materializes video/audio before the prior model
is released. The manifest binds adapter order/digests/shapes, exact schedules,
base content/config/revision, qualification revision, and runtime major.
Shared and segmented manifests are mutually exclusive. Non-diagnostic adapter
symlinks fail closed. No hardware identity selects the path. Full tests pass:
`767 passed, 22 skipped`.

The MZR-3 diagnostic package reuses the three existing primary checkpoints by
symlink because the host has only about 1 GiB free. A real 768x512 25-frame
combined `4 + 1` smoke completed in 46.47 seconds versus the prior standard
78.80 seconds (1.70x), with zero swap. Its contact sheet has none of the
shared-adapter grayscale/structure collapse, although its composition diverges
from the same-seed standard output and therefore is not a non-inferiority pass.

A 768x512 241-frame mountain-bike combined render completed in 259.65 seconds
with zero swap and 40.86 GB peak process footprint. Ten sampled frames retain
the rider, bicycle, forest, and motion across both standard and segmented
outputs. Human review must still judge motion blur, detail, audio content, and
sync. A fresh independent standard `8 + 3` CLI run for the identical prompt
and seed completed in 539.94 seconds. The segmented candidate is therefore
2.079x end to end (51.91% lower latency). Both runs reported zero swap. Fast
peak process footprint was 40.86 GB versus standard 39.65 GB, a 3.05%
increase. The fresh standard output SHA-256 exactly matches the earlier
teacher render, so the review pair has an identity-verified baseline. Atlas
must keep the public/default integration blocked pending human review, a
broader prompt/seed suite, and a second Apple GPU generation.
