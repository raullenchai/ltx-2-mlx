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

The broader screen then rejected this exact checkpoint set. Five additional
241-frame fast renders remained stable at 259.09-262.39 seconds with zero swap,
but the chef speech case omitted the requested front-facing speaker across 20
evenly spaced samples; standard retains the face throughout. The existing
terminal-Stage-2-only A/B retains the face on both sides, isolating the loss to
segmented Stage 1 or its downstream interaction rather than terminal Stage 2
alone. Stop spending qualification compute on these three adapters. Preserve
the portable runtime/package infrastructure, but retrain Stage 1 with broader
semantic/distribution coverage before restarting decoded gates.

Code inspection found a nearer-term confound before changing the objective:
the three packaged "segmented" checkpoints came from a sequential shared
curriculum. `3 -> 5` loaded the `0 -> 3` adapter and `5 -> 7` loaded both prior
updates, even though runtime later applied each file to only one span. Upstream
`d524c95` adds genuinely independent span-v2 training from the clean base and
marks that provenance in checkpoint metadata. The segmented package builder
now rejects cumulative/shared checkpoints. Full tests pass: `770 passed, 22
skipped`; scoped Ruff and diff checks pass.

MZR-3 is running the independent three-segment control against the same eight
training trajectories. Each phase retains only its final checkpoint to fit the
host's 1 GiB free-space constraint; no archived or source data was deleted.
This isolates cumulative weight interference before paying for broader capture
or a distribution-level objective. The current training cohort remains a major
coverage risk: it has only eight samples from four duplicated prompts (watch,
assembly robot, rain, candle), no human face or speech, plus two owl validation
samples. Even if independence repairs the chef case, broader prompt/seed
capture remains mandatory before product qualification.

Upstream `361c4c0` adds a three-checkpoint evaluator that verifies independent
provenance and evaluates each adapter only on its bound span. `4a3fc30` adds
targeted single-span training; the full suite passes `772 passed, 22 skipped`.
A second serialized control is queued after the independent chef render: it
captures 48 training trajectories from 24 prompts and 12 prompt-disjoint
validation trajectories from six prompts at the low-cost 468-token shape,
then retrains only `0 -> 3`. It compares the original and broad-data early
adapters on the same new validation set and combines the broad early adapter
with the independent middle/late adapters for the held-out chef decode. This
keeps the known failing chef prompt/seed out of training while distinguishing
data under-coverage from cumulative adapter interference.

The initial 60-trajectory broad capture would have duplicated about 755 MB of
prompt embeddings and violated the remote free-space margin. Upstream
`ce31075` adds `--reuse-conditions-from`: Stage-1 capture hard-links the exact
already-captured Stage-2 condition file for the same manifest index and writes
only fresh Stage-1 video/audio boundaries. It rejects a missing source or a
different existing destination. Full tests pass: `775 passed, 22 skipped`.
The queued broad control now uses the frozen Stage-2 split conditions on the
same filesystem; no cache redirection, model download, or source deletion is
required.

The clean-base independent control completed all three spans. Against the
same two held-out owl trajectories, `0 -> 3` improved video/audio MSE by
28.47%/31.54%, `3 -> 5` by 61.04%/70.06%, and `5 -> 7` by 4.81%/6.25%.
Those improvements are smaller than the cumulative checkpoints, as expected
after removing inherited updates. A production-shaped held-out chef decode
still omitted the speaker and retained almost only hands, bread, and oven,
while the correctly paired standard decode retained the chef's face and upper
body throughout. Independent segment initialization therefore does not repair
the semantic failure; data under-coverage remains the leading hypothesis.

The first comparison accidentally selected `case-06`, which the corrected
deterministic review index identifies as pottery, not speech. It was rejected
before judgment and retained only in a local invalid-baseline audit directory.
The valid chef standard is `case-05/sample-05-teacher.mp4`; all conclusions
above use that same-prompt, same-seed pair. The broad 24-prompt `0 -> 3`
control is now capturing trajectories. Its condition files are verified hard
links by inode; fresh Stage-1 boundaries are only about 0.5 MiB per sample.
The targeted independent trainer also supports a fail-closed `--steps N`
override only when exactly one `--span` is selected. If the first 100-step
broad run is underfit, use a fresh output root for an equal-exposure control;
do not overwrite or relabel the 100-step checkpoint.

The next orthogonal decode control can replace selected packaged adapter spans
with the immutable base while preserving the same compressed schedule and
span-v2 noise. `render_segmented_stage1_ablation.py` accepts this only for a
diagnostic qualification and the option is deliberately absent from the
product CLI. Queue `--base-span 0-3` and then all three base spans after the
broad early render to distinguish a harmful early correction from a schedule-
level failure. The control is configured only by the research script's private
state; the production pipeline constructor and CLI API remain unchanged. Full
tests pass: `781 passed, 22 skipped`.

Schedule attribution is now closed. Six clean-base/custom schedules spanning
1.956-2.123x all failed the same chef identity/frontal-framing gate. A targeted
broad-data `1 -> 3` student then improved 12 prompt-disjoint trajectories by
49.16% video MSE and 39.54% audio MSE with no greater-than-5% sample regression,
but both its combined 278.28-second decode and its 281.78-second early-only
decode omitted almost the entire chef. The early student itself therefore
enters the wrong semantic branch; downstream adapters are not required to
trigger the failure.

The next control uses progressive on-policy inputs. The new
`materialize_stage1_student_rollout.py` creates a fail-closed same-filesystem
hard-link view, replaces only the student's target video/audio boundary, and
records the checkpoint digest and lineage. It preserves all conditions, seeds,
noise lanes, and later teacher targets. MZR-3 materialized 48 train and 12
prompt-disjoint validation `1 -> 3` outputs in 97.32/24.39 student seconds.
The old independent `3 -> 5` adapter still improves video/audio MSE on that
shifted validation input by 51.40%/47.08%, so exposure bias alone is not a
sufficient diagnosis. A matched rank-4 on-policy `3 -> 5` control is running;
it must substantially exceed that reference and restore the held-out chef or
the next objective should weight semantic/perceptual error rather than train
another ordinary-MSE downstream span. Full tests pass: `791 passed, 22
skipped`. Public/default integration remains blocked and belongs to Atlas.

The first matched on-policy control failed its compute gate: the new rank-4
`3 -> 5` adapter improved shifted-input video/audio MSE by 47.43%/38.34%,
worse than the old teacher-forced adapter's 51.40%/47.08%. It was not decoded,
and `5 -> 7` training did not start. The sharper issue is target reachability:
that control asked a student boundary to jump to the original teacher boundary,
instead of asking the clean teacher where it would go from the student's actual
state.

`materialize_stage1_teacher_correction.py` now runs every original clean-teacher
fine step from an arbitrary student boundary with the matching original noise
lanes. A teacher-boundary `3 -> 5` smoke reproduced the captured target to
video/audio MSE `3.26e-10`/`7.38e-11`; only 5 of 27,904 stored bfloat16 values
differed, with maximum absolute difference 0.001953125. It then generated 48
train and 12 validation reachable targets in 193.35/47.58 seconds. The sole
rank-4 teacher-corrected control is running. Full tests pass: `793 passed, 22
skipped`. Do not decode unless it beats the prior latent references.

The teacher-corrected control also failed the compute gate. Against its
reachable validation target, the new rank-4 adapter improved video/audio MSE
by 50.71%/47.66%, while the existing teacher-forced rank-8 adapter improved
57.71%/55.89% on exactly the same inputs and targets. Neither on-policy
variant was decoded, and no `5 -> 7` training started. Ordinary endpoint MSE
retraining is now stopped.

The next decoded control keeps exact clean-base steps `0 -> 1 -> 2 -> 3`, then
uses the existing independent `3 -> 5` and `5 -> 7` adapters and exact
`7 -> 8`: boundaries `[0,1,2,3,5,7,8]`. This spends six Stage-1 calls to
protect the high-noise semantic branch and should establish whether a roughly
1.8x quality-safe schedule exists before pursuing the remaining portable
kernel/runtime gap to 2x.

That exact-high-noise control completed the full 768x512x241 chef workload in
301.40 seconds versus the identity-verified 539.94-second standard, a 1.791x
speedup and 44.18% latency reduction. It used zero swap and peaked at
40,379,845,680 bytes of process memory. Ten sampled frames preserve the chef,
face, upper body, bread, oven, composition, and motion; unlike every compressed
high-noise schedule, it does not omit the requested speaker. A synchronized
side-by-side and candidate-with-audio are recorded under
`123/strategy/ltx-exact-high-noise-chef-2026-09-15/`. This is a strong initial
screen, not yet human or suite-level non-inferiority.

The measured 4.49% 241-frame dequantized-matmul gain would project to about
287.87 seconds and 1.876x if it composes, leaving roughly 6.22% additional
latency reduction to cross 2x. The next experiment therefore merges only the
middle `3 -> 7` span while keeping exact `0 -> 1 -> 2 -> 3` and `7 -> 8`.
`train_stage1_segmented_curriculum.py --span 3-7` is diagnostic-only and does
not alter the default three-span training set. The resulting five-call route is
projected near 280 seconds before dispatch and 267 seconds after it, but it
must pass held-out latent and decoded gates before packaging.

The rank-4 merged-middle run completed 100 steps on the M3 Ultra Studio in
230.25 seconds at a trainer-reported 19.49 GB peak and zero swap. On 12
prompt-disjoint validation trajectories it improved video/audio MSE by
15.22%/16.45% versus the same clean-base `3 -> 7` jump. Every sample improved
in both modalities; the weakest audio sample still improved 2.51%.

The 768x512x241 MZR-3 chef decode then completed in 278.38 seconds, or 1.940x
versus the 539.94-second standard, with zero swap and a 40,381,926,520-byte
peak process footprint. Ten sampled frames keep the same male chef, face,
upper body, bread, oven, composition, and action sequence, passing the prior
categorical subject-preservation gate. Human detail/motion/audio review and a
broader suite remain open. The real merged-middle plus dequantized-matmul
dispatch run completed in 270.01 seconds: 1.9997x and 49.985% lower latency,
zero swap, and a 39,464,962,808-byte peak footprint. It misses the strict
269.97-second 2x boundary by 0.04 seconds and must not be rounded into a 2x
claim. All ten sampled frames still retain the chef and action sequence; SSIM
against the non-dispatch merged-middle render is 0.963892.

Product-contract implementation is now in progress without changing the CLI
surface or default. The segmented loader accepts a second exact capability,
`ltx_stage1_exact_prefix_middle_span_v1`, normalizes it to exact `0 -> 1`,
`1 -> 2`, `2 -> 3`, learned `3 -> 7`, and exact-final `7 -> 8`, and rejects
any other schedule or adapter span. A dedicated packager binds immutable base,
config/transformer/checkpoint digests, independent training provenance, noise
metadata, rank/shapes, and runtime major. The first real diagnostic manifest
loads successfully; focused tests pass 31/31 and the full suite passes 802
with 22 skips. Standard generation remains unchanged when the existing
segmented-manifest flag is absent. Atlas still owns release qualification and
default policy after human and broader-suite gates.

The real product entry point has now run end to end: `ltx_pipelines_mlx.cli
generate` loaded the exact-prefix manifest, composed the dequantized-matmul
dispatch and terminal-fast Stage 2, and completed in 269.21 seconds (2.0057x),
zero swap, with a 40,371,407,824-byte peak footprint. Its MP4 SHA-256 exactly
matches the prior 270.01-second research-path combined render. Across the two
combined runs, the range is 269.21-270.01 seconds and the median is 269.61
seconds (about 2.001x). Because the range straddles the strict 269.97-second 2x
boundary, describe the result as approximately 2x, not stably greater than 2x.

Adversarial self-review found that the packager enforced clean-base independent
training provenance but the loader did not repeat that check. A hand-crafted
manifest with a matching digest could therefore admit a cumulative/shared
checkpoint. The loader now requires
`stage1_curriculum_adapter_mode=independent` for every segmented capability,
with a direct bypass regression test. The full suite passes 803 with 22 skips.
The independent reviewer service on `spark2` could not start because its Codex
refresh token is revoked; it posted no PR comment and must be reauthenticated
before the automated review loop can grant LGTM.

The real product CLI has now also completed three broader 768x512x241 cases:
cafe faces/hands/dialogue in 269.76 seconds, mountain-bike fast motion in
269.25 seconds, and four-impact synchronization in 268.97 seconds. All used
zero swap; peak process footprint ranged from 39.51 to 40.42 GB. Sampled
frames preserve requested subjects and coarse actions. A simple transient
screen found the first four candidate impact attacks at
0.56/1.04/1.52/2.36 seconds versus 0.56/1.06/1.64/2.28 for standard, but the
candidate was about 5.3 dB quieter and human A/V review remains authoritative.
Artifacts are under `123/strategy/ltx-exact-prefix-suite-2026-09-15/`.

Next: owner reviews the three side-by-sides and audio files; Vector repeats the
same immutable artifact on a second Apple Silicon generation; Atlas decides
default/public integration only after those pass; reauthenticate `spark2` and
rerun the PR review loop. Do not upload or default the diagnostic artifact.
