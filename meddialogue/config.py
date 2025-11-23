"""
Configuration Classes for MedDialogue
====================================

Centralized configuration for all aspects of the framework including
safety, training, models, and task definitions.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.1.0 - Added field ordering, logical styles, and optional validation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats for model responses."""
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"


class PIISensitivity(Enum):
    """PII detection sensitivity levels (stub for backward compatibility)."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class SafetyConfig:
    """
    Configuration for safety guardrails (stub for backward compatibility).

    Safety checks have been removed. This stub maintains API compatibility.
    """
    enable_pii_detection: bool = False
    pii_sensitivity: PIISensitivity = PIISensitivity.LOW
    enable_bias_monitoring: bool = False
    enable_clinical_validation: bool = False
    max_pii_threshold: float = 1.0
    bias_demographic_fields: List[str] = field(default_factory=list)
    require_icd_validation: bool = False
    block_on_safety_failure: bool = False
    log_safety_events: bool = False
    pii_patterns: Dict[str, str] = field(default_factory=dict)
    custom_medical_terms: List[str] = field(default_factory=list)
    custom_pii_patterns: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """
    Configuration for a specific medical task.
    
    This is the core configuration that makes MedDialogue task-agnostic.
    Users define their task, output fields, question templates, and medical context.
    
    Attributes:
        task_name: Name of the task (e.g., "malnutrition_assessment", "diabetes_screening")
        task_description: Description for the system prompt
        input_field: Column name for input clinical text in your data
        output_fields: List of output fields to generate (e.g., ["diagnosis", "severity"])
        question_templates: Question templates by field (see example below)
        output_formats: Supported output formats for this task
        output_format_ratios: Weighted ratios for format selection
        field_ordering: Field priority for logical question ordering (NEW in v1.1.0)
        medical_terminology: Task-specific medical terms for validation
        diagnostic_codes: Valid diagnostic codes (ICD-10, ICD-9, SNOMED, etc.)
        severity_levels: Valid severity classifications
        default_system_prompt: System prompt template for the model
    
    Example Question Templates Structure:
        question_templates = {
            'diagnosis': [
                "What is the diagnosis?",
                "Identify the primary diagnosis",
                "What condition does this patient have?"
            ],
            'severity': [
                "What is the severity level?",
                "How severe is this condition?"
            ]
        }
    
    Example Output Format Ratios:
        output_format_ratios = {
            "text": 0.45,      # 45% plain text responses
            "json": 0.30,      # 30% JSON structured responses  
            "xml": 0.10,       # 10% XML responses
            "markdown": 0.15   # 15% Markdown formatted responses
        }
        
        Rules:
        - Keys must match OutputFormat enum values: "text", "json", "xml", "markdown"
        - Values must be floats between 0.0 and 1.0
        - Must sum to 1.0 (will be normalized if close)
        - If empty dict or None, formats selected uniformly (equal probability)
        - Format must be in output_formats list to be used
    
    Example Field Ordering (NEW in v1.1.0):
        field_ordering = {
            "growth_and_anthropometrics": 1,  # Assessment data (priority 1)
            "physical_exam": 1,                # Assessment data (priority 1)
            "labs_and_screening": 1,           # Assessment data (priority 1)
            "diagnosis_and_reasoning": 2,      # Diagnosis (priority 2)
            "care_plan": 3,                    # Treatment/Action (priority 3)
        }
        
        - Lower numbers = higher priority (asked first in logical styles)
        - Used for reasoning-oriented question combinations (styles 8-16)
        - Follows clinical logic: assess → diagnose → treat
        - If None, fields ordered as they appear in output_fields
    """
    task_name: str
    task_description: str
    input_field: str = "clinical_note"
    output_fields: List[str] = field(default_factory=lambda: ["diagnosis", "severity", "recommendations"])
    question_templates: Dict[str, List[str]] = field(default_factory=dict)
    output_formats: List[OutputFormat] = field(default_factory=lambda: [
        OutputFormat.TEXT, 
        OutputFormat.JSON, 
        OutputFormat.XML, 
        OutputFormat.MARKDOWN
    ])
    
    # Weighted format selection
    output_format_ratios: Dict[str, float] = field(default_factory=dict)
    
    # NEW in v1.1.0: Field ordering for logical question combinations
    field_ordering: Optional[Dict[str, int]] = None
    
    medical_terminology: Dict[str, List[str]] = field(default_factory=dict)
    diagnostic_codes: List[str] = field(default_factory=list)
    severity_levels: List[str] = field(default_factory=lambda: ["mild", "moderate", "severe"])
    default_system_prompt: str = "You are an expert medical AI assistant specializing in {task_name}. Provide clear, evidence-based clinical analyses."
    
    def __post_init__(self):
        """Generate default question templates, validate format ratios, and setup field ordering."""
        # Auto-generate question templates if none provided
        if not self.question_templates:
            self.question_templates = {}
            for field in self.output_fields:
                field_readable = field.replace('_', ' ').title()
                self.question_templates[field] = [
                    f"What is the {field.replace('_', ' ')}?",
                    f"Tell me the {field.replace('_', ' ')}",
                    f"Provide the {field.replace('_', ' ')}",
                    f"Identify the {field.replace('_', ' ')}",
                    f"What is the patient's {field.replace('_', ' ')}?"
                ]
            
            logger.info(f"Auto-generated question templates for {len(self.output_fields)} fields")
        
        # Setup default field ordering if not provided
        if self.field_ordering is None:
            # Default: fields ordered as they appear in output_fields
            self.field_ordering = {field: idx + 1 for idx, field in enumerate(self.output_fields)}
            logger.info("Using default field ordering (sequential)")
        else:
            # Validate field ordering - ensure all output_fields are covered
            missing_fields = set(self.output_fields) - set(self.field_ordering.keys())
            if missing_fields:
                logger.warning(f"Fields missing from field_ordering: {missing_fields}. Adding with default priority.")
                max_priority = max(self.field_ordering.values()) if self.field_ordering else 0
                for field in missing_fields:
                    self.field_ordering[field] = max_priority + 1
            
            logger.info(f"Using custom field ordering with {len(self.field_ordering)} fields")
        
        # Validate output_format_ratios if provided
        if self.output_format_ratios:
            self._validate_format_ratios()
        else:
            logger.info("No output_format_ratios specified - using uniform format selection")
    
    def _validate_format_ratios(self):
        """
        Validate that format ratios are valid and sum to 1.0.
        
        Checks:
        1. All format keys are valid (text, json, xml, markdown)
        2. All format keys exist in output_formats list
        3. All values are between 0.0 and 1.0
        4. Sum equals 1.0 (with 0.01 tolerance)
        5. No negative ratios
        
        Raises:
            ValueError: If validation fails
        """
        # Check that all format keys are valid OutputFormat values
        valid_format_names = {f.value for f in OutputFormat}
        invalid_formats = set(self.output_format_ratios.keys()) - valid_format_names
        
        if invalid_formats:
            raise ValueError(
                f"Invalid format names in output_format_ratios: {invalid_formats}. "
                f"Must be one of: {valid_format_names}"
            )
        
        # Check that all formats in ratios are in output_formats list
        current_format_names = {f.value for f in self.output_formats}
        missing_formats = set(self.output_format_ratios.keys()) - current_format_names
        
        if missing_formats:
            raise ValueError(
                f"Formats in output_format_ratios must be in output_formats list. "
                f"Missing from output_formats: {missing_formats}. "
                f"Current output_formats: {current_format_names}"
            )
        
        # Check for negative ratios
        negative_ratios = {k: v for k, v in self.output_format_ratios.items() if v < 0}
        if negative_ratios:
            raise ValueError(f"Format ratios cannot be negative: {negative_ratios}")
        
        # Check for values > 1.0
        invalid_values = {k: v for k, v in self.output_format_ratios.items() if v > 1.0}
        if invalid_values:
            raise ValueError(f"Format ratios cannot exceed 1.0: {invalid_values}")
        
        # Check that ratios sum to 1.0 (with tolerance for floating point errors)
        total = sum(self.output_format_ratios.values())
        
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"output_format_ratios must sum to 1.0, got {total:.4f}. "
                f"Current ratios: {self.output_format_ratios}"
            )
        
        # Normalize to exactly 1.0 if close but not exact
        if total != 1.0:
            factor = 1.0 / total
            self.output_format_ratios = {
                k: v * factor for k, v in self.output_format_ratios.items()
            }
            logger.info(f"Normalized format ratios to sum to 1.0 (was {total:.4f})")
        
        logger.info(f"Validated output_format_ratios: {self.output_format_ratios}")
        for format_name, ratio in sorted(self.output_format_ratios.items()):
            logger.info(f"  {format_name}: {ratio:.1%}")
    
    def get_system_prompt(self) -> str:
        """Get formatted system prompt."""
        return self.default_system_prompt.format(task_name=self.task_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_name": self.task_name,
            "task_description": self.task_description,
            "input_field": self.input_field,
            "output_fields": self.output_fields,
            "question_templates": self.question_templates,
            "output_formats": [f.value for f in self.output_formats],
            "output_format_ratios": self.output_format_ratios,
            "field_ordering": self.field_ordering,
            "medical_terminology": self.medical_terminology,
            "diagnostic_codes": self.diagnostic_codes,
            "severity_levels": self.severity_levels,
            "system_prompt": self.get_system_prompt()
        }


@dataclass
class TrainingConfig:
    """
    Training hyperparameters and configuration.
    
    Attributes:
        num_epochs: Number of training epochs (default: 3)
        batch_size: Batch size for training (default: 2, reduce if OOM)
        learning_rate: Learning rate (default: 2e-4, good for LoRA)
        max_steps: Maximum training steps (overrides epochs if set)
        warmup_ratio: Warmup ratio for learning rate scheduler (default: 0.05)
        weight_decay: Weight decay for optimizer (default: 0.01)
        gradient_accumulation_steps: Gradient accumulation steps (default: 16)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        save_strategy: When to save checkpoints ("steps" or "epoch")
        save_steps: Steps between saves (if save_strategy="steps")
        save_total_limit: Maximum number of checkpoints to keep (default: 2)
        evaluation_strategy: When to evaluate ("steps", "epoch", or "no")
        eval_steps: Steps between evaluations (if evaluation_strategy="steps")
        logging_steps: Steps between logging (default: 10)
        fp16: Enable FP16 training (default: False)
        bf16: Enable BF16 training (default: True, preferred if supported)
        use_full_dataset: Use full dataset (True) or subsample for quick testing (False)
        subsample_size: If use_full_dataset=False, size of subsample (default: 1000)
    """
    num_epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    save_total_limit: int = 2
    evaluation_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    logging_steps: int = 10
    fp16: bool = False
    bf16: bool = True
    use_full_dataset: bool = True
    subsample_size: int = 1000
    
    def __post_init__(self):
        """Validate configuration."""
        if self.save_strategy not in ["steps", "epoch"]:
            raise ValueError("save_strategy must be 'steps' or 'epoch'")
        if self.evaluation_strategy not in ["steps", "epoch", "no"]:
            raise ValueError("evaluation_strategy must be 'steps', 'epoch', or 'no'")


@dataclass
class LoRAConfig:
    """
    LoRA adapter configuration for efficient fine-tuning.
    
    Attributes:
        r: LoRA rank (higher = more parameters, better but slower)
        lora_alpha: LoRA alpha parameter (typically 2*r)
        lora_dropout: Dropout probability for LoRA layers (default: 0.0)
        target_modules: List of module names to apply LoRA to
        bias: Bias training strategy ("none", "all", "lora_only")
        use_rslora: Use rank-stabilized LoRA (default: False)
        use_gradient_checkpointing: Enable gradient checkpointing (default: "unsloth")
        init_lora_weights: How to initialize LoRA weights (default: True)
    """
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    use_rslora: bool = False
    use_gradient_checkpointing: str = "unsloth"
    init_lora_weights: bool = True


@dataclass
class ModelConfig:
    """
    Configuration for supported models.
    
    Attributes:
        model_name: Hugging Face model name
        model_type: Model family (llama, mistral, phi, qwen)
        chat_template: Chat template name (default: "default")
        max_seq_length: Maximum sequence length (default: 2048)
        supports_system_role: Whether model supports system messages (default: True)
        lora_config: LoRA configuration for this model
    """
    model_name: str
    model_type: str
    chat_template: str = "default"
    max_seq_length: int = 2048
    supports_system_role: bool = True
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "chat_template": self.chat_template,
            "max_seq_length": self.max_seq_length,
            "supports_system_role": self.supports_system_role,
            "lora_config": self.lora_config.__dict__
        }


@dataclass
class ConversationConfig:
    """
    Configuration for conversation generation strategy (v1.1.0).

    Controls how training conversations are structured and generated, including:
    - Single-turn vs multi-turn conversation distribution
    - Question combination styles (grammatical vs logical reasoning)
    - Context window management and response allocation
    - Typo injection for robustness training
    - Validation data splitting

    This class does NOT multiply data (maintains 1:1 row-to-example mapping).
    Instead, it configures HOW each conversation example is structured.

    Version 1.1.0 Changes:
    - Added logical_style_ratio for reasoning-oriented question combinations
    - Changed validation_split default to 0.0 (no validation by default)
    - Context window now specified in tokens (max_seq_length)

    Attributes:
        single_turn_ratio: Ratio of single-turn conversations (default: 0.5)
        max_multi_turns: Maximum conversation turns in multi-turn examples (default: 10)
        include_typos: Include questions with typos for robustness (default: True)
        typo_ratio: Ratio of questions with typos (default: 0.15)
        validation_split: Proportion of data for validation (default: 0.0 = no validation)
        include_followup_questions: Generate follow-up questions in multi-turn (default: True)
        logical_style_ratio: Ratio of logical vs grammatical question styles (default: 0.4)
        context_window_size: Context window in characters (computed from max_seq_length)
        response_allocation_ratio: Proportion of remaining context for response (default: 0.25)
        buffer_ratio: Safety buffer for tokenization variance (default: 0.10)
    """
    single_turn_ratio: float = 0.5
    max_multi_turns: int = 10
    include_typos: bool = True
    typo_ratio: float = 0.15
    validation_split: float = 0.0  # Changed default to 0.0 (no validation)
    include_followup_questions: bool = True
    
    # NEW in v1.1.0: Logical style usage ratio
    logical_style_ratio: float = 0.4  # 40% logical, 60% grammatical
    
    context_window_size: int = 8192  # In characters (~2048 tokens)
    response_allocation_ratio: float = 0.25
    buffer_ratio: float = 0.10
    min_question_length: int = 1000
    max_question_length: int = 8000
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.single_turn_ratio <= 1.0:
            raise ValueError("single_turn_ratio must be between 0.0 and 1.0")
        
        if not 0.0 <= self.typo_ratio <= 1.0:
            raise ValueError("typo_ratio must be between 0.0 and 1.0")
        
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must be between 0.0 and 1.0 (exclusive)")
        
        if not 0.0 <= self.logical_style_ratio <= 1.0:
            raise ValueError("logical_style_ratio must be between 0.0 and 1.0")
        
        if self.max_multi_turns < 1:
            raise ValueError("max_multi_turns must be at least 1")
        
        if not 0.0 < self.response_allocation_ratio < 1.0:
            raise ValueError("response_allocation_ratio must be between 0.0 and 1.0")
        
        if not 0.0 < self.buffer_ratio < 1.0:
            raise ValueError("buffer_ratio must be between 0.0 and 1.0")
        
        if self.min_question_length >= self.max_question_length:
            raise ValueError("min_question_length must be less than max_question_length")
        
        if self.context_window_size < 1000:
            raise ValueError("context_window_size must be at least 1000 chars")
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context allocation statistics."""
        return {
            "total_context": self.context_window_size,
            "response_ratio": self.response_allocation_ratio,
            "buffer_ratio": self.buffer_ratio,
            "question_range": f"{self.min_question_length}-{self.max_question_length} chars",
            "effective_tokens": f"~{self.context_window_size // 4}K",
            "logical_style_ratio": self.logical_style_ratio,
            "validation_enabled": self.validation_split > 0
        }


def load_config_from_json(json_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_config_to_json(config: Any, json_path: str):
    """Save configuration to JSON file."""
    with open(json_path, 'w') as f:
        if hasattr(config, 'to_dict'):
            json.dump(config.to_dict(), f, indent=2)
        else:
            config_dict = {}
            for key, value in config.__dict__.items():
                if isinstance(value, Enum):
                    config_dict[key] = value.value
                elif hasattr(value, '__dict__'):
                    config_dict[key] = value.__dict__
                else:
                    config_dict[key] = value
            json.dump(config_dict, f, indent=2)


# Backward compatibility alias
DataMultiplicationConfig = ConversationConfig