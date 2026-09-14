"""One-step stage-2 terminal-latent distillation strategy."""

from __future__ import annotations

from typing import Any, Literal

import mlx.core as mx

from ltx_trainer_mlx.distillation import terminal_velocity_target
from ltx_trainer_mlx.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModalityInputs,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class Stage2TerminalDistillConfig(TrainingStrategyConfigBase):
    """Configuration for deterministic stage-2 ``3 -> 1`` distillation."""

    name: Literal["stage2_terminal_distill"]

    def __init__(
        self,
        *,
        sigma: float = 0.909375,
        video_start_latents_dir: str = "stage2_video_start_latents",
        video_terminal_latents_dir: str = "stage2_video_terminal_latents",
        audio_start_latents_dir: str = "stage2_audio_start_latents",
        audio_terminal_latents_dir: str = "stage2_audio_terminal_latents",
        video_loss_weight: float = 1.0,
        audio_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(name="stage2_terminal_distill")
        if not 0 < sigma <= 1:
            raise ValueError("sigma must be in (0, 1]")
        if video_loss_weight < 0 or audio_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if video_loss_weight == 0 and audio_loss_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        self.sigma = sigma
        self.video_start_latents_dir = video_start_latents_dir
        self.video_terminal_latents_dir = video_terminal_latents_dir
        self.audio_start_latents_dir = audio_start_latents_dir
        self.audio_terminal_latents_dir = audio_terminal_latents_dir
        self.video_loss_weight = video_loss_weight
        self.audio_loss_weight = audio_loss_weight


class Stage2TerminalDistillStrategy(TrainingStrategy):
    """Train one model evaluation to reproduce the three-step teacher terminal."""

    config: Stage2TerminalDistillConfig

    @property
    def requires_audio(self) -> bool:
        return True

    def get_data_sources(self) -> dict[str, str]:
        return {
            self.config.video_start_latents_dir: "video_start",
            self.config.video_terminal_latents_dir: "video_terminal",
            self.config.audio_start_latents_dir: "audio_start",
            self.config.audio_terminal_latents_dir: "audio_terminal",
            "conditions": "conditions",
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        sigma_sampler: Any,
    ) -> ModelInputs:
        del sigma_sampler  # Stage 2 has a fixed product sigma.

        video_start_data = batch["video_start"]
        video_terminal_data = batch["video_terminal"]
        video_start, spatial_dims = self._video_patchifier.patchify(video_start_data["latents"])
        video_terminal, terminal_dims = self._video_patchifier.patchify(video_terminal_data["latents"])
        if spatial_dims != terminal_dims or video_start.shape != video_terminal.shape:
            raise ValueError("stage-2 video start and terminal latent shapes must match")

        batch_size, video_tokens, _ = video_start.shape
        num_frames, height, width = spatial_dims
        fps_data = video_start_data.get("fps")
        fps = float(fps_data[0].item()) if fps_data is not None else DEFAULT_FPS
        stored_sigma = video_start_data.get("sigma")
        if stored_sigma is not None and abs(float(stored_sigma[0].item()) - self.config.sigma) > 1e-6:
            raise ValueError(
                f"trajectory sigma {float(stored_sigma[0].item())} does not match configured sigma {self.config.sigma}"
            )

        audio_start, _ = self._audio_patchifier.patchify(batch["audio_start"]["latents"])
        audio_terminal, _ = self._audio_patchifier.patchify(batch["audio_terminal"]["latents"])
        if audio_start.shape != audio_terminal.shape:
            raise ValueError("stage-2 audio start and terminal latent shapes must match")

        sigma = mx.full((batch_size,), self.config.sigma, dtype=video_start.dtype)
        video_targets = terminal_velocity_target(video_start, video_terminal, self.config.sigma)
        audio_targets = terminal_velocity_target(audio_start, audio_terminal, self.config.sigma)
        video_timesteps = mx.full((batch_size, video_tokens), self.config.sigma, dtype=video_start.dtype)
        audio_timesteps = mx.full(
            (batch_size, audio_start.shape[1]),
            self.config.sigma,
            dtype=audio_start.dtype,
        )

        conditions = batch["conditions"]
        context_mask = conditions["prompt_attention_mask"]
        video_modality = ModalityInputs(
            enabled=True,
            latent=video_start,
            sigma=sigma,
            timesteps=video_timesteps,
            positions=self._get_video_positions(num_frames, height, width, fps),
            context=conditions["video_prompt_embeds"],
            context_mask=context_mask,
        )
        audio_modality = ModalityInputs(
            enabled=True,
            latent=audio_start,
            sigma=sigma,
            timesteps=audio_timesteps,
            positions=self._get_audio_positions(audio_start.shape[1]),
            context=conditions["audio_prompt_embeds"],
            context_mask=context_mask,
        )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=mx.ones((batch_size, video_tokens), dtype=mx.bool_),
            audio_loss_mask=mx.ones((batch_size, audio_start.shape[1]), dtype=mx.bool_),
        )

    def compute_loss(
        self,
        video_pred: mx.array,
        audio_pred: mx.array | None,
        inputs: ModelInputs,
    ) -> mx.array:
        if audio_pred is None or inputs.audio_targets is None:
            raise ValueError("stage-2 terminal distillation requires audio predictions")
        video_loss = mx.mean(mx.square(video_pred - inputs.video_targets))
        audio_loss = mx.mean(mx.square(audio_pred - inputs.audio_targets))
        return self.config.video_loss_weight * video_loss + self.config.audio_loss_weight * audio_loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "distillation": "stage2_terminal",
            "stage2_sigma": self.config.sigma,
            "stage2_steps": 1,
        }
