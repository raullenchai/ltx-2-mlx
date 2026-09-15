"""Distilled two-stage video generation pipeline.

Mirrors upstream ``ltx_pipelines.distilled.DistilledPipeline`` 1:1:

  Stage 1: Distilled DiT at **half resolution** (8 steps, no CFG).
  Stage 2: Spatial 2x upscaler + distilled DiT refine at **full resolution**
           (3 steps, no CFG).

Same distilled checkpoint is used in both stages — no LoRA fusion between
stages (the model is already distilled). Use this pipeline when you want
the speed of the distilled model at higher target resolutions, where
running distilled directly at full res can produce out-of-distribution
artefacts.

For the simpler distilled-at-target one-stage path, see
:class:`BasePipeline`.

For dev model + CFG quality, see :class:`TI2VidTwoStagesPipeline` /
:class:`TI2VidTwoStagesHQPipeline`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.loader.fast_stage1 import FastStage1Package, read_fast_stage1_package
from ltx_core_mlx.loader.fast_stage1_segmented import (
    FastStage1SegmentedPackage,
    read_fast_stage1_segmented_package,
)
from ltx_core_mlx.loader.fast_stage2 import FastStage2Package, read_fast_stage2_package
from ltx_core_mlx.model.transformer.model import LTXModel, X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)

from .scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from .ti2vid_two_stages import TI2VidTwoStagesPipeline
from .utils._orchestration import detect_model_version
from .utils.helpers import create_noised_state
from .utils.progress import phase
from .utils.samplers import DenoiseOutput, ancestral_denoise_loop, denoise_loop

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern

# Generation from which stage 1 is sampled with the ancestral (SDE) Euler
# sampler instead of the deterministic one. Mirrors upstream
# ``ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)``; ``detect_model_version``
# returns "2.5.0" / "2.3", so the gate is a ``"2.5"`` prefix match.
ANCESTRAL_SAMPLER_SINCE_VERSION = "2.5"

# Fully ancestral noise injection: eta=0 is a plain Euler step, eta=1 injects
# the full variance-preserving amount at every step (upstream verbatim).
ANCESTRAL_ETA = 1.0
ANCESTRAL_S_NOISE = 1.0

# The loop's noise generator is seeded from the pipeline seed plus this offset
# (upstream verbatim). Without it the loop's first draw would be bit-identical
# to the initial latent noise: both draw the same-shaped randn from a freshly
# seeded generator.
ANCESTRAL_NOISE_SEED_OFFSET = 10000


def should_use_ancestral_sampler(model_dir: str | Path) -> bool:
    """Whether a checkpoint's generation calls for the ancestral stage-1 sampler.

    LTX-2.5 (``detect_model_version(model_dir)`` starting with ``"2.5"``) →
    True; the 2.3 port keeps the deterministic Euler stage 1 exactly as
    before. Mirrors upstream ``should_use_ancestral_sampler(transformer_path)``.
    """
    return detect_model_version(model_dir).startswith(ANCESTRAL_SAMPLER_SINCE_VERSION)


class DistilledPipeline(TI2VidTwoStagesPipeline):
    """Distilled two-stage T2V/I2V pipeline (half-res → upscale → full-res refine).

    Reuses :class:`TI2VidTwoStagesPipeline`'s upsampler loading and helpers but
    overrides ``generate_two_stage`` to:

    - Skip negative-prompt encoding (no CFG).
    - Load the distilled transformer directly (no dev model, no LoRA fusion).
    - Stage 1: ancestral (SDE) Euler sampler with ``DISTILLED_SIGMAS`` for
      LTX-2.5 checkpoints (``should_use_ancestral_sampler``); the 2.3 port
      keeps the deterministic ``denoise_loop`` exactly as before.
    - Run the same distilled transformer for stage 2 with ``STAGE_2_SIGMAS``
      (always deterministic).

    Args:
        model_dir: Path to model weights or HuggingFace repo ID. Must
            contain the distilled checkpoint (e.g. ``dgrauet/ltx-2.3-mlx-q8``
            ships ``transformer-distilled.safetensors``).
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        low_ram_streaming: Stream transformer blocks from disk.
        tile_count: Optional modality tiling configuration.
        fast_stage1_manifest: Optional schema-v1 compressed Stage-1 package
            manifest inside the model directory.
        fast_stage1_segmented_manifest: Optional schema-v1 segmented Stage-1
            package. Each learned transition uses its own adapter and the
            final transition runs on the exact base transformer.
        fast_stage2_manifest: Optional schema-v1 manifest filename inside the
            model directory. Enabling it validates and runs the portable
            learned first stage-2 transition before a clean-base correction.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        tile_count=None,
        fast_stage1_manifest: str | None = None,
        fast_stage1_segmented_manifest: str | None = None,
        fast_stage2_manifest: str | None = None,
    ):
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            tile_count=tile_count,
        )
        # Stage-1 sampler selection (mirrors upstream ``DistilledPipeline``):
        # LTX-2.5 checkpoints sample stage 1 with the ancestral (SDE) Euler
        # sampler; the 2.3 port keeps the deterministic Euler loop.
        self.use_ancestral_sampler = should_use_ancestral_sampler(self.model_dir)
        self._fast_stage1_package: FastStage1Package | None = None
        self._fast_stage1_segmented_package: FastStage1SegmentedPackage | None = None
        self._fast_stage2_package: FastStage2Package | None = None
        if fast_stage1_manifest is not None and fast_stage1_segmented_manifest is not None:
            raise ValueError("choose either a shared or segmented fast stage-1 package")
        if fast_stage1_manifest is not None:
            self._fast_stage1_package = read_fast_stage1_package(self.model_dir, fast_stage1_manifest)
            if not self.use_ancestral_sampler:
                raise ValueError("fast stage 1 requires an LTX-2.5 ancestral checkpoint")
        if fast_stage1_segmented_manifest is not None:
            self._fast_stage1_segmented_package = read_fast_stage1_segmented_package(
                self.model_dir,
                fast_stage1_segmented_manifest,
            )
            if not self.use_ancestral_sampler:
                raise ValueError("segmented fast stage 1 requires an LTX-2.5 ancestral checkpoint")
        if fast_stage2_manifest is not None:
            self._fast_stage2_package = read_fast_stage2_package(self.model_dir, fast_stage2_manifest)
        if self._fast_stage1_package is not None and self._fast_stage2_package is not None:
            stage1 = self._fast_stage1_package
            stage2 = self._fast_stage2_package
            if (
                stage1.transformer_path != stage2.transformer_path
                or stage1.contract.base_model_id != stage2.contract.base_model_id
                or stage1.contract.base_revision != stage2.contract.base_revision
                or stage1.contract.transformer_sha256 != stage2.contract.transformer_sha256
                or stage1.contract.transformer_config_sha256 != stage2.contract.transformer_config_sha256
            ):
                raise ValueError("fast stage-1 and stage-2 packages target different base transformers")
        if self._fast_stage1_segmented_package is not None and self._fast_stage2_package is not None:
            stage1 = self._fast_stage1_segmented_package
            stage2 = self._fast_stage2_package
            if (
                stage1.transformer_path != stage2.transformer_path
                or stage1.base_model_id != stage2.contract.base_model_id
                or stage1.base_revision != stage2.contract.base_revision
                or stage1.transformer_sha256 != stage2.contract.transformer_sha256
                or stage1.transformer_config_sha256 != stage2.contract.transformer_config_sha256
            ):
                raise ValueError("segmented fast stage-1 and stage-2 packages target different base transformers")

    def _distilled_transformer_path(self) -> Path:
        if self._fast_stage1_package is not None:
            return self._fast_stage1_package.transformer_path
        if self._fast_stage1_segmented_package is not None:
            return self._fast_stage1_segmented_package.transformer_path
        if self._fast_stage2_package is not None:
            return self._fast_stage2_package.transformer_path
        transformer_path = self.model_dir / "transformer.safetensors"
        if not transformer_path.exists():
            transformer_path = self._resolve_safetensors(self.model_dir, "transformer-distilled")
        return transformer_path

    def load(self) -> None:
        """Load distilled DiT + VAE encoder + upsampler (skip decoders).

        Skips reloading the text encoder: ``generate_two_stage`` encodes
        the prompt and frees Gemma BEFORE calling :meth:`load`. Loading
        Gemma again here would just thrash the Metal heap (7.5 GB
        load/mmap + free) right before DiT is loaded — a documented
        cause of macOS GPU watchdog crashes under sustained system
        contention.
        """
        if self._loaded:
            return

        if self.dit is None:
            if self._fast_stage1_package is None and self._fast_stage1_segmented_package is None:
                self.dit = self._load_transformer_with_optional_streaming(self._distilled_transformer_path())
            else:
                if self._fast_stage1_segmented_package is not None:
                    package = self._fast_stage1_segmented_package
                    first = package.segments[0]
                    adapter_path = first.adapter_path
                    strength = (
                        first.lora_alpha / first.lora_rank
                        if first.lora_alpha is not None and first.lora_rank is not None
                        else None
                    )
                else:
                    assert self._fast_stage1_package is not None
                    package = self._fast_stage1_package
                    adapter_path = package.adapter_path
                    strength = package.contract.lora_alpha / package.contract.lora_rank
                base_spans = getattr(self, "_diagnostic_base_stage1_spans", frozenset())
                if self._fast_stage1_segmented_package is not None and (
                    adapter_path is None or (first.start_index, first.end_index) in base_spans
                ):
                    self.dit = self._load_transformer_with_optional_streaming(package.transformer_path)
                else:
                    assert adapter_path is not None and strength is not None
                    self._pending_loras = [(str(adapter_path), strength)]
                    try:
                        self.dit = self._load_transformer_with_optional_streaming(package.transformer_path)
                    finally:
                        del self._pending_loras

        self._load_vae_encoder()

        if self.upsampler is None:
            self._load_upsampler()

        self._loaded = True

    def _stage2_model(self, dit: LTXModel, latent_shape: tuple[int, int, int]) -> X0Model:
        if self._tile_count is None:
            return X0Model(dit)

        from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

        return X0Model(TiledLTXModel(dit, VideoModalityTiler(self._tile_count, latent_shape=latent_shape)))

    def _run_clean_final_fast_stage1(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_embeds: mx.array,
        audio_embeds: mx.array,
        *,
        latent_shape: tuple[int, int, int],
        noise_seed: int,
    ) -> DenoiseOutput:
        """Run three learned coarse transitions and the exact clean-base final step."""
        package = self._fast_stage1_package
        assert package is not None and package.contract.clean_final_transition
        assert self.dit is not None
        contract = package.contract
        assert contract.noise_step_spans is not None
        assert contract.noise_reference_sigmas is not None
        schedule = list(contract.schedule)

        student_model = self._stage2_model(self.dit, latent_shape)
        learned = ancestral_denoise_loop(
            model=student_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=schedule[:-1],
            noise_seed=noise_seed,
            noise_step_spans=list(contract.noise_step_spans[:-1]),
            noise_reference_sigmas=list(contract.noise_reference_sigmas),
            noise_total_steps=contract.noise_total_steps,
            eta=ANCESTRAL_ETA,
            s_noise=ANCESTRAL_S_NOISE,
        )
        _materialize(learned.video_latent, learned.audio_latent)
        del student_model
        self.dit = None
        aggressive_cleanup()

        try:
            self.dit = self._load_transformer_with_optional_streaming(package.transformer_path)
            clean_model = self._stage2_model(self.dit, latent_shape)
            corrected = denoise_loop(
                model=clean_model,
                video_state=replace(video_state, latent=learned.video_latent),
                audio_state=replace(audio_state, latent=learned.audio_latent),
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=schedule[-2:],
            )
            del clean_model
            return corrected
        except Exception:
            self.dit = None
            self._loaded = False
            aggressive_cleanup()
            raise

    def _run_segmented_fast_stage1(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_embeds: mx.array,
        audio_embeds: mx.array,
        *,
        latent_shape: tuple[int, int, int],
        noise_seed: int,
    ) -> DenoiseOutput:
        """Run each validated adapter/base transition, then clean base."""
        package = self._fast_stage1_segmented_package
        assert package is not None
        assert self.dit is not None

        current_video = video_state
        current_audio = audio_state
        try:
            for index, segment in enumerate(package.segments):
                if index:
                    base_spans = getattr(self, "_diagnostic_base_stage1_spans", frozenset())
                    if segment.adapter_path is None or (segment.start_index, segment.end_index) in base_spans:
                        self.dit = self._load_transformer_with_optional_streaming(package.transformer_path)
                    else:
                        assert segment.lora_alpha is not None and segment.lora_rank is not None
                        self._pending_loras = [
                            (str(segment.adapter_path), segment.lora_alpha / segment.lora_rank)
                        ]
                        try:
                            self.dit = self._load_transformer_with_optional_streaming(package.transformer_path)
                        finally:
                            del self._pending_loras

                student_model = self._stage2_model(self.dit, latent_shape)
                learned = ancestral_denoise_loop(
                    model=student_model,
                    video_state=current_video,
                    audio_state=current_audio,
                    video_text_embeds=video_embeds,
                    audio_text_embeds=audio_embeds,
                    sigmas=[segment.sigma, segment.target_sigma],
                    noise_seed=noise_seed,
                    noise_step_spans=[(segment.start_index, segment.end_index)],
                    noise_reference_sigmas=list(package.noise_reference_sigmas),
                    noise_total_steps=package.noise_total_steps,
                    eta=ANCESTRAL_ETA,
                    s_noise=ANCESTRAL_S_NOISE,
                )
                _materialize(learned.video_latent, learned.audio_latent)
                current_video = replace(current_video, latent=learned.video_latent)
                current_audio = replace(current_audio, latent=learned.audio_latent)
                del student_model
                self.dit = None
                aggressive_cleanup()

            self.dit = self._load_transformer_with_optional_streaming(package.transformer_path)
            clean_model = self._stage2_model(self.dit, latent_shape)
            corrected = denoise_loop(
                model=clean_model,
                video_state=current_video,
                audio_state=current_audio,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=list(package.schedule[-2:]),
            )
            del clean_model
            return corrected
        except Exception:
            self.dit = None
            self._loaded = False
            aggressive_cleanup()
            raise

    def _run_fast_stage2(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_embeds: mx.array,
        audio_embeds: mx.array,
        *,
        latent_shape: tuple[int, int, int],
    ) -> DenoiseOutput:
        """Run a learned transition and its base correction when declared."""
        package = self._fast_stage2_package
        assert package is not None
        strength = package.contract.lora_alpha / package.contract.lora_rank
        schedule = list(package.contract.schedule)

        self.dit = None
        aggressive_cleanup()
        try:
            self._pending_loras = [(str(package.adapter_path), strength)]
            try:
                self.dit = self._load_transformer_with_optional_streaming(package.transformer_path)
            finally:
                del self._pending_loras

            student_model = self._stage2_model(self.dit, latent_shape)
            learned = denoise_loop(
                model=student_model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=schedule[:2],
            )
            # Finish the adapter-backed graph before releasing its weights.
            _materialize(learned.video_latent, learned.audio_latent)
            del student_model
            self.dit = None
            aggressive_cleanup()

            if len(schedule) == 2:
                # A qualified terminal student has already reached sigma zero.
                # Mark the pipeline unloaded so a later request reloads a clean
                # base model instead of retaining or reusing adapter state.
                self._loaded = False
                return learned

            self.dit = self._load_transformer_with_optional_streaming(package.transformer_path)
            correction_model = self._stage2_model(self.dit, latent_shape)
            corrected = denoise_loop(
                model=correction_model,
                video_state=replace(video_state, latent=learned.video_latent),
                audio_state=replace(audio_state, latent=learned.audio_latent),
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=schedule[1:],
            )
            del correction_model
            return corrected
        except Exception:
            self.dit = None
            self._loaded = False
            aggressive_cleanup()
            raise

    def generate_two_stage(  # type: ignore[override]
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        image: str | None = None,
        images=None,
        stage1_trajectory_callback: Callable[..., None] | None = None,
        stage2_trajectory_callback: Callable[..., None] | None = None,
        **_unused_kwargs,
    ) -> tuple[mx.array, mx.array]:
        """Generate video using the distilled two-stage pipeline.

        Args:
            prompt: Text prompt.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Stage 1 steps (default: full DISTILLED_SIGMAS = 8).
            stage2_steps: Stage 2 steps (default: full STAGE_2_SIGMAS = 3).
            image: Optional reference image for I2V conditioning.
            stage1_trajectory_callback: Optional research callback invoked for
                the initial stage-1 state and every completed transition.
            stage2_trajectory_callback: Optional research callback invoked with
                the exact stage-2 start/terminal video and audio tokens plus
                text conditioning. The default ``None`` has no runtime effect.
            **_unused_kwargs: Accepted (and ignored) for signature compatibility
                with :meth:`TI2VidTwoStagesPipeline.generate_two_stage`. CFG / STG /
                TeaCache flags don't apply to the distilled flow.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        if self._fast_stage1_package is not None or self._fast_stage1_segmented_package is not None:
            if stage1_steps is not None:
                raise ValueError("fast stage 1 uses its qualified schedule and cannot override stage1_steps")
            if stage1_trajectory_callback is not None:
                raise ValueError("fast stage 1 cannot capture a standard teacher trajectory")
        if self._fast_stage2_package is not None:
            if stage2_steps is not None:
                raise ValueError("fast stage 2 uses its qualified schedule and cannot override stage2_steps")
            if stage2_trajectory_callback is not None:
                raise ValueError("fast stage 2 cannot capture a standard teacher trajectory")
        if (
            self._fast_stage1_package is not None
            or self._fast_stage1_segmented_package is not None
            or self._fast_stage2_package is not None
        ) and getattr(self, "_pending_loras", None):
            raise ValueError("fast stage packages cannot be combined with additional LoRAs")

        # --- Text encoding (positive only — no CFG) ---
        self._load_text_encoder()
        with phase("Encoding prompt", verbose=self.verbose):
            video_embeds, audio_embeds = self._encode_text(prompt)
            _materialize(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        # --- Load distilled DiT + VAE encoder + upsampler ---
        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        # --- Stage 1: half resolution ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        # I2V conditioning at half resolution. ``images`` is the upstream-iso
        # multi-anchor list; ``image`` is the legacy single-image shorthand.
        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        enc_h_half = H_half * 32
        enc_w_half = W_half * 32
        resolved_images = list(images) if images else []
        if image is not None and not resolved_images:
            resolved_images = [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)]
        conditionings_1: list = []
        if resolved_images:
            conditionings_1 = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h_half,
                enc_w=enc_w_half,
                spatial_dims=(F, H_half, W_half),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
                model_dir=self.model_dir,
            )

        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings_1,
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H_half, W_half),  # unused
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )

        sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS
        noise_step_indices = None
        noise_step_spans = None
        noise_reference_sigmas = None
        noise_total_steps = None
        if self._fast_stage1_package is not None:
            contract = self._fast_stage1_package.contract
            sigmas_1 = list(contract.schedule)
            if contract.noise_step_indices is not None:
                noise_step_indices = list(contract.noise_step_indices)
            if contract.noise_step_spans is not None:
                noise_step_spans = list(contract.noise_step_spans)
            if contract.noise_reference_sigmas is not None:
                noise_reference_sigmas = list(contract.noise_reference_sigmas)
            noise_total_steps = contract.noise_total_steps
        elif self._fast_stage1_segmented_package is not None:
            sigmas_1 = list(self._fast_stage1_segmented_package.schedule)

        def capture_stage1(sigma: float, video: mx.array, audio: mx.array) -> None:
            if stage1_trajectory_callback is not None:
                stage1_trajectory_callback(
                    sigma=sigma,
                    video=video,
                    audio=audio,
                    video_text_embeds=video_embeds,
                    audio_text_embeds=audio_embeds,
                    spatial_dims=(F, H_half, W_half),
                    frame_rate=frame_rate,
                    noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
                    seed=seed,
                    prompt=prompt,
                )

        capture_stage1(sigmas_1[0], video_state.latent, audio_state.latent)

        self._pre_denoise_flush(video_state, audio_state)
        clean_final_stage1 = bool(
            self._fast_stage1_package is not None
            and getattr(self._fast_stage1_package.contract, "clean_final_transition", False)
        )
        segmented_stage1 = self._fast_stage1_segmented_package is not None
        stage1_dit = None
        x0_model = None
        if segmented_stage1:
            output_1 = self._run_segmented_fast_stage1(
                video_state,
                audio_state,
                video_embeds,
                audio_embeds,
                latent_shape=(F, H_half, W_half),
                noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
            )
        elif clean_final_stage1:
            output_1 = self._run_clean_final_fast_stage1(
                video_state,
                audio_state,
                video_embeds,
                audio_embeds,
                latent_shape=(F, H_half, W_half),
                noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
            )
        else:
            stage1_dit = self.dit
            if self._tile_count is not None:
                from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

                tiler_1 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_half, W_half))
                stage1_dit = TiledLTXModel(self.dit, tiler_1)
            x0_model = X0Model(stage1_dit)

        if not clean_final_stage1 and not segmented_stage1 and self.use_ancestral_sampler:
            # LTX-2.5 stage 1: ancestral (SDE) Euler — fresh seeded noise per
            # step (upstream ``ANCESTRAL_ETA`` / ``S_NOISE`` / seed offset).
            output_1 = ancestral_denoise_loop(
                model=x0_model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_1,
                noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
                noise_step_indices=noise_step_indices,
                noise_total_steps=noise_total_steps,
                noise_step_spans=noise_step_spans,
                noise_reference_sigmas=noise_reference_sigmas,
                eta=ANCESTRAL_ETA,
                s_noise=ANCESTRAL_S_NOISE,
                step_callback=capture_stage1 if stage1_trajectory_callback is not None else None,
            )
        elif not clean_final_stage1 and not segmented_stage1:
            # 2.3 stage 1: deterministic Euler (unchanged behaviour).
            output_1 = denoise_loop(
                model=x0_model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_1,
                step_callback=capture_stage1 if stage1_trajectory_callback is not None else None,
            )
        if self.low_memory:
            aggressive_cleanup()

        if self._fast_stage1_package is not None or self._fast_stage1_segmented_package is not None:
            # Finish adapter-backed graphs before releasing Stage 1. The clean
            # base or separately qualified Stage-2 adapter is loaded after
            # upscale, which also minimizes peak memory.
            _materialize(output_1.video_latent, output_1.audio_latent)
            del stage1_dit, x0_model
            if not clean_final_stage1 and not segmented_stage1:
                self.dit = None
            aggressive_cleanup()

        # --- Upscale (same denorm/upsample/renorm as TI2VidTwoStagesPipeline) ---
        # Strip appended keyframe tokens (multi-anchor with frame_idx>0).
        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        _materialize(video_upscaled)

        H_full = H_half * 2
        W_full = W_half * 2

        # I2V conditioning at full resolution (re-encode at upscaled dims)
        conditionings_2: list = []
        if resolved_images:
            enc_h_full = H_full * 32
            enc_w_full = W_full * 32
            conditionings_2 = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h_full,
                enc_w=enc_w_full,
                spatial_dims=(F, H_full, W_full),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
                model_dir=self.model_dir,
            )

        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: full resolution refine (no LoRA swap — already distilled) ---
        # Always deterministic — its 3-step refinement schedule is too short to
        # remove freshly injected noise (upstream-verbatim).
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        if self._fast_stage2_package is not None:
            sigmas_2 = list(self._fast_stage2_package.contract.schedule)
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)

        video_state_2 = create_noised_state(
            base_shape=video_tokens.shape,
            conditionings=conditionings_2,
            spatial_dims=(F, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
            legacy_scalar_blend=True,
        )

        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = create_noised_state(
            base_shape=audio_tokens_1.shape,
            conditionings=[],
            spatial_dims=(F, H_full, W_full),  # unused
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=audio_tokens_1,
        )

        self._pre_denoise_flush(video_state_2, audio_state_2)
        intermediate: dict[str, object] = {}

        def capture_intermediate(sigma_next: float, video: mx.array, audio: mx.array) -> None:
            if abs(sigma_next - STAGE_2_SIGMAS[-2]) < 1e-9:
                intermediate.update(sigma=sigma_next, video=video, audio=audio)

        if self._fast_stage2_package is not None:
            # Drop every Stage-1 model reference before loading the student.
            if self._fast_stage1_package is None and self._fast_stage1_segmented_package is None:
                del stage1_dit, x0_model
            output_2 = self._run_fast_stage2(
                video_state_2,
                audio_state_2,
                video_embeds,
                audio_embeds,
                latent_shape=(F, H_full, W_full),
            )
        else:
            if self.dit is None:
                self.dit = self._load_transformer_with_optional_streaming(self._distilled_transformer_path())
            stage2_x0_model = self._stage2_model(self.dit, (F, H_full, W_full))
            output_2 = denoise_loop(
                model=stage2_x0_model,
                video_state=video_state_2,
                audio_state=audio_state_2,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_2,
                step_callback=capture_intermediate if stage2_trajectory_callback is not None else None,
            )
        if stage2_trajectory_callback is not None:
            if len(sigmas_2) == len(STAGE_2_SIGMAS) and not intermediate:
                raise RuntimeError("full stage-2 trajectory did not capture the penultimate sigma")
            stage2_trajectory_callback(
                video_start=video_state_2.latent,
                video_terminal=output_2.video_latent,
                audio_start=audio_state_2.latent,
                audio_terminal=output_2.audio_latent,
                video_intermediate=intermediate.get("video"),
                audio_intermediate=intermediate.get("audio"),
                intermediate_sigma=intermediate.get("sigma"),
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                spatial_dims=(F, H_full, W_full),
                frame_rate=frame_rate,
                sigma=start_sigma,
                seed=seed,
                prompt=prompt,
            )
        if self.low_memory:
            aggressive_cleanup()

        gen_tokens_2 = output_2.video_latent[:, : F * H_full * W_full, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        if self._fast_stage1_package is not None or self._fast_stage1_segmented_package is not None:
            # A clean Stage-2 model may now be resident. Do not accidentally
            # reuse it as the compressed Stage-1 student on the next request.
            _materialize(video_latent, audio_latent)
            self.dit = None
            self._loaded = False
            aggressive_cleanup()

        return video_latent, audio_latent


__all__ = ["DistilledPipeline"]
