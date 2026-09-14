"""Training strategies for different conditioning modes.

This package implements the Strategy Pattern to handle different training modes:
- Text-to-video training (standard generation, optionally with audio)
- Video-to-video training (IC-LoRA mode with reference videos)

Each strategy encapsulates the specific logic for preparing model inputs
and computing loss.
"""

from __future__ import annotations

import logging

from ltx_trainer_mlx.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModalityInputs,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_trainer_mlx.training_strategies.stage1_transition_distill import (
    Stage1TransitionDistillConfig,
    Stage1TransitionDistillStrategy,
)
from ltx_trainer_mlx.training_strategies.stage2_terminal_distill import (
    Stage2TerminalDistillConfig,
    Stage2TerminalDistillStrategy,
)
from ltx_trainer_mlx.training_strategies.text_to_video import (
    TextToVideoConfig,
    TextToVideoStrategy,
)
from ltx_trainer_mlx.training_strategies.video_to_video import (
    VideoToVideoConfig,
    VideoToVideoStrategy,
)

logger = logging.getLogger(__name__)

# Type alias for all strategy config types
TrainingStrategyConfig = (
    TextToVideoConfig | VideoToVideoConfig | Stage1TransitionDistillConfig | Stage2TerminalDistillConfig
)

__all__ = [
    "DEFAULT_FPS",
    "VIDEO_SCALE_FACTORS",
    "ModalityInputs",
    "ModelInputs",
    "Stage1TransitionDistillConfig",
    "Stage1TransitionDistillStrategy",
    "Stage2TerminalDistillConfig",
    "Stage2TerminalDistillStrategy",
    "TextToVideoConfig",
    "TextToVideoStrategy",
    "TrainingStrategy",
    "TrainingStrategyConfig",
    "TrainingStrategyConfigBase",
    "VideoToVideoConfig",
    "VideoToVideoStrategy",
    "get_training_strategy",
]


def get_training_strategy(config: TrainingStrategyConfig | object) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.

    The strategy is determined by the ``name`` field in the configuration.
    Accepts either native strategy configs (``TextToVideoConfig``,
    ``VideoToVideoConfig``) or the Pydantic ``TrainingStrategyConfig`` from
    ``ltx_trainer_mlx.config``, which is automatically converted.

    Args:
        config: Strategy-specific configuration with a ``name`` field.

    Returns:
        The appropriate training strategy instance.

    Raises:
        ValueError: If strategy name is not supported.
    """
    # Convert Pydantic TrainingStrategyConfig to native strategy config
    if not isinstance(
        config,
        TextToVideoConfig | VideoToVideoConfig | Stage1TransitionDistillConfig | Stage2TerminalDistillConfig,
    ):
        name = getattr(config, "name", None)
        generate_audio = getattr(config, "generate_audio", False)
        if name == "text_to_video":
            config = TextToVideoConfig(with_audio=generate_audio)
        elif name == "video_to_video":
            config = VideoToVideoConfig()
        elif name == "stage2_terminal_distill":
            config = Stage2TerminalDistillConfig(
                sigma=getattr(config, "sigma", 0.909375),
                target_sigma=getattr(config, "target_sigma", 0.0),
                video_start_latents_dir=getattr(
                    config,
                    "video_start_latents_dir",
                    "stage2_video_start_latents",
                ),
                video_terminal_latents_dir=getattr(
                    config,
                    "video_terminal_latents_dir",
                    "stage2_video_terminal_latents",
                ),
                audio_start_latents_dir=getattr(
                    config,
                    "audio_start_latents_dir",
                    "stage2_audio_start_latents",
                ),
                audio_terminal_latents_dir=getattr(
                    config,
                    "audio_terminal_latents_dir",
                    "stage2_audio_terminal_latents",
                ),
                video_loss_weight=getattr(config, "video_loss_weight", 1.0),
                audio_loss_weight=getattr(config, "audio_loss_weight", 1.0),
            )
        elif name == "stage1_transition_distill":
            config = Stage1TransitionDistillConfig(
                sigma=config.sigma,
                target_sigma=config.target_sigma,
                video_start_latents_dir=config.video_start_latents_dir,
                video_terminal_latents_dir=config.video_terminal_latents_dir,
                audio_start_latents_dir=config.audio_start_latents_dir,
                audio_terminal_latents_dir=config.audio_terminal_latents_dir,
                conditions_dir=getattr(config, "conditions_dir", "stage1_conditions"),
                ancestral_noise_step_index=getattr(config, "ancestral_noise_step_index", None),
                ancestral_noise_total_steps=getattr(config, "ancestral_noise_total_steps", 8),
                ancestral_eta=getattr(config, "ancestral_eta", 1.0),
                ancestral_s_noise=getattr(config, "ancestral_s_noise", 1.0),
                video_loss_weight=getattr(config, "video_loss_weight", 1.0),
                audio_loss_weight=getattr(config, "audio_loss_weight", 1.0),
            )
        else:
            raise ValueError(f"Unknown training strategy name: {name}")

    match config:
        case TextToVideoConfig():
            strategy = TextToVideoStrategy(config)
        case VideoToVideoConfig():
            strategy = VideoToVideoStrategy(config)
        case Stage1TransitionDistillConfig():
            strategy = Stage1TransitionDistillStrategy(config)
        case Stage2TerminalDistillConfig():
            strategy = Stage2TerminalDistillStrategy(config)
        case _:
            raise ValueError(f"Unknown training strategy config type: {type(config).__name__}")

    audio_mode = "(audio enabled)" if strategy.requires_audio else "(audio disabled)"
    logger.debug(
        "Using %s training strategy %s",
        strategy.__class__.__name__,
        audio_mode,
    )
    return strategy
