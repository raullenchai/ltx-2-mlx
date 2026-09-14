"""One-evaluation student for a captured ancestral stage-1 transition."""

from __future__ import annotations

from typing import Any, Literal

import mlx.core as mx

from ltx_trainer_mlx.distillation import transition_velocity_target
from ltx_trainer_mlx.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModalityInputs,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class Stage1TransitionDistillConfig(TrainingStrategyConfigBase):
    """Select two captured boundaries that one student evaluation must join."""

    name: Literal["stage1_transition_distill"]

    def __init__(
        self,
        *,
        sigma: float,
        target_sigma: float,
        video_start_latents_dir: str,
        video_terminal_latents_dir: str,
        audio_start_latents_dir: str,
        audio_terminal_latents_dir: str,
        conditions_dir: str = "stage1_conditions",
        video_loss_weight: float = 1.0,
        audio_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(name="stage1_transition_distill")
        if not 0 < sigma <= 1 or not 0 <= target_sigma < sigma:
            raise ValueError("stage-1 transition requires 0 <= target_sigma < sigma <= 1")
        if video_loss_weight < 0 or audio_loss_weight < 0 or video_loss_weight + audio_loss_weight == 0:
            raise ValueError("at least one non-negative loss weight must be positive")
        self.sigma = sigma
        self.target_sigma = target_sigma
        self.video_start_latents_dir = video_start_latents_dir
        self.video_terminal_latents_dir = video_terminal_latents_dir
        self.audio_start_latents_dir = audio_start_latents_dir
        self.audio_terminal_latents_dir = audio_terminal_latents_dir
        self.conditions_dir = conditions_dir
        self.video_loss_weight = video_loss_weight
        self.audio_loss_weight = audio_loss_weight


class Stage1TransitionDistillStrategy(TrainingStrategy):
    """Learn the deterministic endpoint of one captured two-step teacher move."""

    config: Stage1TransitionDistillConfig

    @property
    def requires_audio(self) -> bool:
        return True

    def get_data_sources(self) -> dict[str, str]:
        return {
            self.config.video_start_latents_dir: "video_start",
            self.config.video_terminal_latents_dir: "video_terminal",
            self.config.audio_start_latents_dir: "audio_start",
            self.config.audio_terminal_latents_dir: "audio_terminal",
            self.config.conditions_dir: "conditions",
        }

    def prepare_training_inputs(self, batch: dict[str, Any], sigma_sampler: Any) -> ModelInputs:
        del sigma_sampler
        video_start_data = batch["video_start"]
        video_terminal_data = batch["video_terminal"]
        video_start = video_start_data["latents"]
        video_terminal = video_terminal_data["latents"]
        if video_start.ndim != 3 or video_start.shape != video_terminal.shape:
            raise ValueError("stage-1 video boundary token shapes must match")

        batch_size, video_tokens, _ = video_start.shape
        num_frames = int(video_start_data["num_frames"][0].item())
        height = int(video_start_data["height"][0].item())
        width = int(video_start_data["width"][0].item())
        if num_frames * height * width != video_tokens:
            raise ValueError("stage-1 video metadata does not match token count")
        fps_data = video_start_data.get("fps")
        fps = float(fps_data[0].item()) if fps_data is not None else DEFAULT_FPS

        for data, expected, label in (
            (video_start_data, self.config.sigma, "start"),
            (video_terminal_data, self.config.target_sigma, "target"),
        ):
            stored = data.get("sigma")
            if stored is not None and abs(float(stored[0].item()) - expected) > 1e-6:
                raise ValueError(f"stage-1 {label} sigma does not match configured transition")

        audio_start, _ = self._audio_patchifier.patchify(batch["audio_start"]["latents"])
        audio_terminal, _ = self._audio_patchifier.patchify(batch["audio_terminal"]["latents"])
        if audio_start.shape != audio_terminal.shape:
            raise ValueError("stage-1 audio boundary shapes must match")

        sigma = mx.full((batch_size,), self.config.sigma, dtype=video_start.dtype)
        video_targets = transition_velocity_target(
            video_start, video_terminal, self.config.sigma, self.config.target_sigma
        )
        audio_targets = transition_velocity_target(
            audio_start, audio_terminal, self.config.sigma, self.config.target_sigma
        )
        conditions = batch["conditions"]
        context_mask = conditions["prompt_attention_mask"]
        return ModelInputs(
            video=ModalityInputs(
                enabled=True,
                latent=video_start,
                sigma=sigma,
                timesteps=mx.full((batch_size, video_tokens), self.config.sigma, dtype=video_start.dtype),
                positions=self._get_video_positions(num_frames, height, width, fps),
                context=conditions["video_prompt_embeds"],
                context_mask=context_mask,
            ),
            audio=ModalityInputs(
                enabled=True,
                latent=audio_start,
                sigma=sigma,
                timesteps=mx.full((batch_size, audio_start.shape[1]), self.config.sigma, dtype=audio_start.dtype),
                positions=self._get_audio_positions(audio_start.shape[1]),
                context=conditions["audio_prompt_embeds"],
                context_mask=context_mask,
            ),
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=mx.ones((batch_size, video_tokens), dtype=mx.bool_),
            audio_loss_mask=mx.ones((batch_size, audio_start.shape[1]), dtype=mx.bool_),
        )

    def compute_loss(self, video_pred: mx.array, audio_pred: mx.array | None, inputs: ModelInputs) -> mx.array:
        if audio_pred is None or inputs.audio_targets is None:
            raise ValueError("stage-1 transition distillation requires audio predictions")
        video_loss = mx.mean(mx.square(video_pred - inputs.video_targets))
        audio_loss = mx.mean(mx.square(audio_pred - inputs.audio_targets))
        return self.config.video_loss_weight * video_loss + self.config.audio_loss_weight * audio_loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "distillation": "stage1_transition",
            "stage1_sigma": self.config.sigma,
            "stage1_target_sigma": self.config.target_sigma,
            "stage1_video_start_latents_dir": self.config.video_start_latents_dir,
            "stage1_video_target_latents_dir": self.config.video_terminal_latents_dir,
            "stage1_audio_start_latents_dir": self.config.audio_start_latents_dir,
            "stage1_audio_target_latents_dir": self.config.audio_terminal_latents_dir,
            "stage1_conditions_dir": self.config.conditions_dir,
        }
