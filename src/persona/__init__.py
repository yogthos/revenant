"""Persona system for subjective, non-robotic style transfer."""

from .config import PersonaConfig, get_persona_config, PERSONA_CONFIGS
from .prompt_builder import build_persona_prompt

__all__ = [
    "PersonaConfig",
    "get_persona_config",
    "PERSONA_CONFIGS",
    "build_persona_prompt",
]
