"""Configuration management for the style transfer pipeline."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StyleBlendingWeights:
    """Fitness weights when style blending is enabled."""
    enabled_weight: float = 0.15
    content_with_blending: float = 0.35
    length_with_blending: float = 0.18
    transition_with_blending: float = 0.12
    vocabulary_with_blending: float = 0.12
    fluency_with_blending: float = 0.08


@dataclass
class FitnessWeightsConfig:
    """Configuration for fitness function weights."""
    content: float = 0.40
    length: float = 0.20
    transition: float = 0.15
    vocabulary: float = 0.15
    fluency: float = 0.10
    style_blending: StyleBlendingWeights = field(default_factory=StyleBlendingWeights)


@dataclass
class ThresholdsConfig:
    """Configuration for various strictness thresholds."""
    overuse_word_count: int = 3  # Word appearing more than this is "overused"
    severe_overuse_count: int = 5  # Severe overuse penalty threshold
    entailment_score: float = 0.5  # Min entailment for semantic preservation
    delta_score: float = 1.5  # Burrows' Delta threshold
    content_preservation_min: float = 0.5  # Min content preservation ratio
    novelty_min: float = 0.95  # Min novelty for anachronistic tests
    anachronistic_pass_rate: float = 0.9  # Min pass rate for style generalization


@dataclass
class BlendingConfig:
    """Configuration for SLERP-based author style blending."""
    enabled: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_dir: str = "centroid_cache/"
    authors: Dict[str, float] = field(default_factory=dict)  # author_name -> weight


@dataclass
class LLMProviderConfig:
    """Configuration for a specific LLM provider."""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120


@dataclass
class LLMProviderRoles:
    """Configuration for role-based LLM provider assignment.

    Allows using different providers for different tasks:
    - writer: Fast local model for style generation (e.g., MLX with LoRA)
    - critic: Smarter API model for validation and repair (e.g., DeepSeek)
    - rtt: Provider for Round-Trip Translation neutralization (e.g., DeepSeek)
    """
    writer: str = "mlx"  # Provider for generation
    critic: str = "deepseek"  # Provider for critique/repair
    rtt: str = "deepseek"  # Provider for RTT neutralization


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProviderRoles = field(default_factory=LLMProviderRoles)
    providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)
    max_retries: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0

    def get_provider_config(self, provider_name: str) -> LLMProviderConfig:
        """Get configuration for a specific provider."""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
        return self.providers[provider_name]

    def get_writer_provider(self) -> str:
        """Get the provider name for generation/writing tasks."""
        return self.provider.writer

    def get_critic_provider(self) -> str:
        """Get the provider name for critique/repair tasks."""
        return self.provider.critic


@dataclass
class CorpusConfig:
    """Configuration for corpus processing."""
    min_sentences_per_paragraph: int = 2
    style_audit_threshold: int = 4
    opener_percentage: float = 0.15
    closer_percentage: float = 0.15


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    # Validation settings
    entailment_threshold: float = 0.7  # Min NLI score for semantic preservation

    # Repair settings
    max_repair_attempts: int = 3  # Max critic repair attempts per paragraph
    repair_temperature: float = 0.3  # Low temperature for precise edits

    # Length control settings
    max_expansion_ratio: float = 2.5  # Max output/input word ratio before warning
    target_expansion_ratio: float = 1.5  # Target for LoRA generation (1.5 = 50% expansion for flourish)
    expand_for_texture: bool = False  # Add stronger expansion prompt to encourage elaboration/flourishes

    # LoRA adapter settings (path -> config mapping)
    # Each adapter can have: scale, temperature, top_p, min_p, repetition_penalty, max_tokens, worldview, checkpoint
    lora_adapters: Dict[str, "LoRAAdapterConfig"] = field(default_factory=dict)

    # Neutralization settings
    skip_neutralization: bool = False  # If True, skip RTT and use original text as input

    # Post-processing settings
    repetition_threshold: int = 3  # Words used N+ times get replaced
    reduce_repetition: bool = True  # Enable repetition reduction

    # Document handling
    use_document_context: bool = True  # Extract document-level context
    pass_headings_unchanged: bool = True  # Don't transform headings
    min_paragraph_words: int = 10  # Skip paragraphs shorter than this

    # RAG settings
    use_structural_rag: bool = True  # Enable structural RAG for rhythm/syntax guidance
    use_structural_grafting: bool = True  # Enable structural grafting for argument skeletons
    rag_sample_size: int = 200  # Number of corpus chunks to sample for rhythm pattern analysis

    # Persona settings
    use_persona: bool = True  # Enable persona-based prompting
    apply_input_perturbation: bool = True  # Apply 8% noise to match training distribution
    narrativize_input: bool = True  # Convert impersonal exposition to first-person narrative (match training format)

    # Grammar correction settings (final post-processing pass)
    correct_grammar: bool = True  # Enable style-safe grammar correction
    grammar_language: str = "en-US"  # Language variant: "en-US" or "en-GB"

    # Sentence restructuring settings
    restructure_sentences: bool = True  # Enable balanced→inverted restructuring
    split_sentences: bool = True  # Enable sentence splitting at conjunction points
    max_sentence_length: int = 60  # Words - split sentences longer than this
    sentence_length_variance: float = 0.3  # Variance factor (0.3 = 70%-130% of max)


@dataclass
class LoRAAdapterConfig:
    """Configuration for a specific LoRA adapter.

    These settings control the balance between style strength and coherence.
    Each adapter in lora_adapters can have its own settings.
    """
    enabled: bool = True  # Whether this adapter is active
    scale: float = 1.0  # Adapter influence (0.0=base only, 1.0=full, >1.0=amplified)
    temperature: float = 0.6  # Higher = more creative, lower = more coherent
    top_p: float = 0.92  # Nucleus sampling threshold
    min_p: float = 0.05  # Minimum probability filter
    repetition_penalty: float = 1.15  # Penalty for repeating tokens
    max_tokens: int = 512  # Maximum tokens to generate
    worldview: str = ""  # Author worldview prompt file
    checkpoint: Optional[str] = None  # Specific checkpoint to use


# Keep LoRAConfig as alias for backward compatibility
LoRAConfig = LoRAAdapterConfig


@dataclass
class SemanticValidationConfig:
    """Configuration for semantic validation."""
    min_proposition_coverage: float = 0.9
    max_hallucinated_entities: int = 0
    require_citation_preservation: bool = True


@dataclass
class StatisticalValidationConfig:
    """Configuration for statistical validation."""
    length_tolerance: float = 0.2
    burstiness_tolerance: float = 0.3
    min_vocab_match: float = 0.5


@dataclass
class ValidationConfig:
    """Configuration for validation."""
    # Top-level validation settings
    entailment_threshold: float = 0.7  # Min NLI score for semantic preservation
    max_hallucinations_before_reject: int = 2  # Trigger repair after this many hallucinations
    # Nested validation configs
    semantic: SemanticValidationConfig = field(default_factory=SemanticValidationConfig)
    statistical: StatisticalValidationConfig = field(default_factory=StatisticalValidationConfig)


@dataclass
class ContextBudgetConfig:
    """Token budget configuration for context management."""
    system_prompt_max: int = 1500
    user_prompt_max: int = 600
    keep_last_n_sentences: int = 5
    max_conversation_tokens: int = 8000


@dataclass
class VoiceInjectionConfig:
    """Configuration for voice profile injection into generation."""
    enabled: bool = True
    assertiveness_weight: float = 0.7  # How much to weight assertiveness patterns
    rhetorical_weight: float = 0.8  # How much to weight rhetorical patterns


@dataclass
class StyleConfig:
    """Configuration for style transfer settings."""
    perspective: str = "preserve"  # preserve, first_person_singular, first_person_plural, third_person, author_voice_third_person
    voice_injection: VoiceInjectionConfig = field(default_factory=VoiceInjectionConfig)
    blending: BlendingConfig = field(default_factory=BlendingConfig)

    def validate_perspective(self) -> bool:
        """Check if perspective setting is valid."""
        valid_perspectives = {
            "preserve",
            "first_person_singular",
            "first_person_plural",
            "third_person",
            "author_voice_third_person",  # Writes AS the author using third person (not about the author)
        }
        return self.perspective in valid_perspectives

    @staticmethod
    def get_perspective_instruction(perspective: str, author: str) -> str:
        """Get the instruction text for a given perspective."""
        instructions = {
            "preserve": "Maintain the same perspective (first/third person) as the source text.",
            "first_person_singular": "Write in first person singular (I, me, my).",
            "first_person_plural": "Write in first person plural (we, us, our).",
            "third_person": "Write in third person (he, she, they, it).",
            "author_voice_third_person": f"Write AS {author} would write, using third person perspective. Channel {author}'s voice and style while referring to subjects in third person.",
        }
        return instructions.get(perspective, "")


@dataclass
class Config:
    """Main configuration container."""
    fitness_weights: FitnessWeightsConfig = field(default_factory=FitnessWeightsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    context_budget: ContextBudgetConfig = field(default_factory=ContextBudgetConfig)
    log_level: str = "INFO"
    log_json: bool = False


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in string values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        resolved = os.environ.get(env_var, "")
        if not resolved:
            logger.warning(f"Environment variable {env_var} not set")
        return resolved
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def _parse_llm_provider_config(data: Dict) -> LLMProviderConfig:
    """Parse LLM provider configuration."""
    return LLMProviderConfig(
        api_key=_resolve_env_vars(data.get("api_key", "")),
        base_url=data.get("base_url", ""),
        model=data.get("model", ""),
        max_tokens=data.get("max_tokens", 4096),
        temperature=data.get("temperature", 0.7),
        timeout=data.get("timeout", 120),
    )


def _parse_lora_adapter_config(data: Dict) -> LoRAAdapterConfig:
    """Parse LoRA adapter configuration from dict."""
    return LoRAAdapterConfig(
        enabled=data.get("enabled", True),
        scale=data.get("scale", 1.0),
        temperature=data.get("temperature", 0.6),
        top_p=data.get("top_p", 0.92),
        min_p=data.get("min_p", 0.05),
        repetition_penalty=data.get("repetition_penalty", 1.15),
        max_tokens=data.get("max_tokens", 512),
        worldview=data.get("worldview", ""),
        checkpoint=data.get("checkpoint"),
    )


def _parse_lora_adapters(data: Dict) -> Dict[str, LoRAAdapterConfig]:
    """Parse lora_adapters section into typed configs.

    Handles both old format (path -> scale) and new format (path -> config dict).
    """
    result = {}
    for path, value in data.items():
        if isinstance(value, dict):
            # New format: full config dict
            result[path] = _parse_lora_adapter_config(value)
        else:
            # Old format: just a scale number
            result[path] = LoRAAdapterConfig(scale=float(value))
    return result


def _parse_llm_config(data: Dict) -> LLMConfig:
    """Parse LLM configuration section."""
    providers = {}
    for name, provider_data in data.get("providers", {}).items():
        providers[name] = _parse_llm_provider_config(provider_data)

    retry_config = data.get("retry", {})
    provider_data = data.get("provider", {})

    provider_roles = LLMProviderRoles(
        writer=provider_data.get("writer", "mlx"),
        critic=provider_data.get("critic", "deepseek"),
        rtt=provider_data.get("rtt", "deepseek"),
    )

    return LLMConfig(
        provider=provider_roles,
        providers=providers,
        max_retries=retry_config.get("max_attempts", 5),
        base_delay=retry_config.get("base_delay", 2.0),
        max_delay=retry_config.get("max_delay", 60.0),
    )


# Module-level config cache
_config_cache: dict = {}


def load_config(config_path: str = "config.json") -> Config:
    """Load configuration from a JSON file.

    Uses a cache to avoid reloading the same config file multiple times.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Parsed configuration object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    # Return cached config if available
    if config_path in _config_cache:
        return _config_cache[config_path]

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please copy config.json.sample to config.json and configure it."
        )

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")

    # Parse each section
    config = Config()

    # Parse fitness weights
    if "fitness_weights" in data:
        fw_data = data["fitness_weights"]
        blending_data = fw_data.get("style_blending", {})
        config.fitness_weights = FitnessWeightsConfig(
            content=fw_data.get("content", 0.40),
            length=fw_data.get("length", 0.20),
            transition=fw_data.get("transition", 0.15),
            vocabulary=fw_data.get("vocabulary", 0.15),
            fluency=fw_data.get("fluency", 0.10),
            style_blending=StyleBlendingWeights(
                enabled_weight=blending_data.get("enabled_weight", 0.15),
                content_with_blending=blending_data.get("content_with_blending", 0.35),
                length_with_blending=blending_data.get("length_with_blending", 0.18),
                transition_with_blending=blending_data.get("transition_with_blending", 0.12),
                vocabulary_with_blending=blending_data.get("vocabulary_with_blending", 0.12),
                fluency_with_blending=blending_data.get("fluency_with_blending", 0.08),
            ),
        )

    # Parse thresholds
    if "thresholds" in data:
        th_data = data["thresholds"]
        config.thresholds = ThresholdsConfig(
            overuse_word_count=th_data.get("overuse_word_count", 3),
            severe_overuse_count=th_data.get("severe_overuse_count", 5),
            entailment_score=th_data.get("entailment_score", 0.5),
            delta_score=th_data.get("delta_score", 1.5),
            content_preservation_min=th_data.get("content_preservation_min", 0.5),
            novelty_min=th_data.get("novelty_min", 0.95),
            anachronistic_pass_rate=th_data.get("anachronistic_pass_rate", 0.9),
        )

    if "llm" in data:
        config.llm = _parse_llm_config(data["llm"])

    if "corpus" in data:
        config.corpus = CorpusConfig(
            min_sentences_per_paragraph=data["corpus"].get("min_sentences_per_paragraph", 2),
            style_audit_threshold=data["corpus"].get("style_audit_threshold", 4),
            opener_percentage=data["corpus"].get("opener_percentage", 0.15),
            closer_percentage=data["corpus"].get("closer_percentage", 0.15),
        )

    if "generation" in data:
        gen = data["generation"]
        # Get entailment_threshold from validation section or generation section
        val_data = data.get("validation", {})
        entailment_thresh = gen.get("entailment_threshold", val_data.get("entailment_threshold", 0.7))
        config.generation = GenerationConfig(
            # Validation settings
            entailment_threshold=entailment_thresh,
            # Repair settings
            max_repair_attempts=gen.get("max_repair_attempts", 3),
            repair_temperature=gen.get("repair_temperature", 0.3),
            # Length control
            max_expansion_ratio=gen.get("max_expansion_ratio", 2.5),
            target_expansion_ratio=gen.get("target_expansion_ratio", 1.5),
            expand_for_texture=gen.get("expand_for_texture", False),
            # LoRA adapters (path -> config mapping)
            lora_adapters=_parse_lora_adapters(gen.get("lora_adapters", {})),
            # Neutralization
            skip_neutralization=gen.get("skip_neutralization", False),
            # Post-processing
            repetition_threshold=gen.get("repetition_threshold", 3),
            reduce_repetition=gen.get("reduce_repetition", True),
            # Document handling
            use_document_context=gen.get("use_document_context", True),
            pass_headings_unchanged=gen.get("pass_headings_unchanged", True),
            min_paragraph_words=gen.get("min_paragraph_words", 10),
            # RAG settings
            use_structural_rag=gen.get("use_structural_rag", True),
            use_structural_grafting=gen.get("use_structural_grafting", True),
            rag_sample_size=gen.get("rag_sample_size", 200),
            # Persona settings
            use_persona=gen.get("use_persona", True),
            apply_input_perturbation=gen.get("apply_input_perturbation", True),
            narrativize_input=gen.get("narrativize_input", True),
            # Grammar correction settings
            correct_grammar=gen.get("correct_grammar", True),
            grammar_language=gen.get("grammar_language", "en-US"),
            # Sentence restructuring settings
            restructure_sentences=gen.get("restructure_sentences", True),
            split_sentences=gen.get("split_sentences", True),
            max_sentence_length=gen.get("max_sentence_length", 60),
            sentence_length_variance=gen.get("sentence_length_variance", 0.3),
        )

    if "style" in data:
        style_data = data["style"]
        voice_data = style_data.get("voice_injection", {})
        blending_data = style_data.get("blending", {})
        config.style = StyleConfig(
            perspective=style_data.get("perspective", "preserve"),
            voice_injection=VoiceInjectionConfig(
                enabled=voice_data.get("enabled", True),
                assertiveness_weight=voice_data.get("assertiveness_weight", 0.7),
                rhetorical_weight=voice_data.get("rhetorical_weight", 0.8),
            ),
            blending=BlendingConfig(
                enabled=blending_data.get("enabled", False),
                embedding_model=blending_data.get("embedding_model", "all-MiniLM-L6-v2"),
                cache_dir=blending_data.get("cache_dir", "centroid_cache/"),
                authors=blending_data.get("authors", {}),
            ),
        )
        if not config.style.validate_perspective():
            logger.warning(
                f"Invalid perspective '{config.style.perspective}', using 'preserve'"
            )
            config.style.perspective = "preserve"

    if "validation" in data:
        val_data = data["validation"]
        # Load top-level validation settings
        config.validation.entailment_threshold = val_data.get("entailment_threshold", 0.7)
        config.validation.max_hallucinations_before_reject = val_data.get("max_hallucinations_before_reject", 2)
        # Load nested configs
        if "semantic" in val_data:
            config.validation.semantic = SemanticValidationConfig(
                min_proposition_coverage=val_data["semantic"].get("min_proposition_coverage", 0.9),
                max_hallucinated_entities=val_data["semantic"].get("max_hallucinated_entities", 0),
                require_citation_preservation=val_data["semantic"].get("require_citation_preservation", True),
            )
        if "statistical" in val_data:
            config.validation.statistical = StatisticalValidationConfig(
                length_tolerance=val_data["statistical"].get("length_tolerance", 0.2),
                burstiness_tolerance=val_data["statistical"].get("burstiness_tolerance", 0.3),
                min_vocab_match=val_data["statistical"].get("min_vocab_match", 0.5),
            )

    if "context_budget" in data:
        config.context_budget = ContextBudgetConfig(
            system_prompt_max=data["context_budget"].get("system_prompt_max", 1500),
            user_prompt_max=data["context_budget"].get("user_prompt_max", 600),
            keep_last_n_sentences=data["context_budget"].get("keep_last_n_sentences", 5),
            max_conversation_tokens=data["context_budget"].get("max_conversation_tokens", 8000),
        )

    config.log_level = data.get("log_level", "INFO")
    config.log_json = data.get("log_json", False)

    # Cache the config
    _config_cache[config_path] = config

    logger.info(f"Loaded configuration from {config_path}")
    return config


def create_default_config() -> Dict:
    """Create a default configuration dictionary."""
    return {
        "llm": {
            "provider": {
                "writer": "mlx",
                "critic": "deepseek"
            },
            "providers": {
                "deepseek": {
                    "api_key": "${DEEPSEEK_API_KEY}",
                    "base_url": "https://api.deepseek.com",
                    "model": "deepseek-chat",
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "timeout": 120
                },
                "mlx": {
                    "model": "mlx-community/Qwen3-8B-4bit",
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "llama3",
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
            },
            "retry": {
                "max_attempts": 5,
                "base_delay": 2,
                "max_delay": 60
            }
        },
        "generation": {
            "max_repair_attempts": 3,
            "repair_temperature": 0.3,
            "repetition_threshold": 3
        },
        "validation": {
            "entailment_threshold": 0.7,
            "max_hallucinations_before_reject": 2
        },
        "log_level": "INFO"
    }


def get_adapter_config(adapter_path: Optional[str] = None) -> LoRAAdapterConfig:
    """Get LoRA adapter config for a specific adapter path.

    Args:
        adapter_path: Path to the adapter directory. If None, returns defaults.

    Returns:
        LoRAAdapterConfig for the adapter, or defaults if not found.
    """
    if not adapter_path:
        return LoRAAdapterConfig()

    try:
        config = load_config()
        adapters = config.generation.lora_adapters

        # Try exact match first
        if adapter_path in adapters:
            return adapters[adapter_path]

        # Try matching by adapter directory name
        from pathlib import Path
        adapter_name = Path(adapter_path).name
        for path, adapter_config in adapters.items():
            if Path(path).name == adapter_name:
                return adapter_config

    except Exception as e:
        logger.debug(f"Could not load adapter config: {e}")

    return LoRAAdapterConfig()
