"""Persona system for subjective, non-robotic style transfer."""

from .prompt_builder import build_persona_instruction, build_persona_prompt

__all__ = [
    "build_persona_instruction",
    "build_persona_prompt",
]
