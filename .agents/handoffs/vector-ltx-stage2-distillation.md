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
