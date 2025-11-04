"""
MedDialogue: Healthcare Conversational Fine-Tuning Framework
============================================================

A task-agnostic framework for fine-tuning large language models on medical
conversational tasks with intelligent question combination, format teaching,
and strategic variation to prevent catastrophic forgetting.

Key Features (v1.0.0):
- 1:1 row-to-example mapping (no artificial multiplication)
- 16 question combination styles (7 grammatical + 9 logical reasoning)
- Intelligent field ordering for reasoning questions
- Optional validation split (configurable)
- Semantic variation across questions
- Format teaching (TEXT, JSON, XML, Markdown)
- Multi-turn conversations with context preservation
- Comprehensive safety checks (PII detection, bias monitoring)
- LoRA efficient fine-tuning

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0

Changelog (v1.0.0):
- Added 9 logical reasoning question combination styles (8-16)
- Added intelligent field ordering for clinical logic (assess → diagnose → treat)
- Made validation split truly optional (default: 0.0)
- Added logical_style_ratio configuration (default: 0.4)
- Enhanced logging and statistics
- Improved context utilization reporting

Example Usage:
    >>> from meddialogue import MedDialogue, TaskConfig, DataMultiplicationConfig
    >>> 
    >>> # Define task with field ordering
    >>> task_config = TaskConfig(
    ...     task_name="malnutrition_assessment",
    ...     output_fields=["growth_and_anthropometrics", "diagnosis", "care_plan"],
    ...     field_ordering={
    ...         "growth_and_anthropometrics": 1,  # Assessment first
    ...         "diagnosis": 2,                    # Diagnosis second
    ...         "care_plan": 3                     # Action third
    ...     }
    ... )
    >>> 
    >>> # Configure data preparation
    >>> mult_config = DataMultiplicationConfig(
    ...     single_turn_ratio=0.7,
    ...     validation_split=0.2,
    ...     logical_style_ratio=0.4
    ... )
    >>> 
    >>> # Train model
    >>> trainer = MedDialogue(task_config=task_config, mult_config=mult_config)
    >>> results = trainer.train_from_csv("data.csv")
    >>> 
    >>> # Inference
    >>> response = trainer.infer(clinical_note="Patient presents with...", format="json")
"""

from .config import (
    TaskConfig,
    SafetyConfig,
    TrainingConfig,
    ModelConfig,
    LoRAConfig,
    DataMultiplicationConfig,
    OutputFormat,
    PIISensitivity
)
from .core import MedDialogue
from .data_prep import (
    DataPrep,
    QuestionCombiner,
    ResponseFormatter,
    ConversationExample,
    DataMultiplier  # Backward compatibility alias
)
from .train import Trainer
from .inference import (
    InferencePipeline,
    BatchInference
)
from .models import (
    ModelRegistry,
    get_supported_models,
    load_model,
    apply_lora,
    get_device
)
from .safety import (
    SafetyChecker,
    PIIDetector,
    BiasMonitor,
    ClinicalValidator
)

__version__ = "1.0.0"
__author__ = "Frederick Gyasi"
__email__ = "gyasi@musc.edu"
__institution__ = "Medical University of South Carolina, Biomedical Informatics Center"

__all__ = [
    # Main interface
    "MedDialogue",
    
    # Configuration
    "TaskConfig",
    "SafetyConfig",
    "TrainingConfig",
    "ModelConfig",
    "LoRAConfig",
    "DataMultiplicationConfig",
    "OutputFormat",
    "PIISensitivity",
    
    # Data preparation
    "DataPrep",
    "DataMultiplier",  # Backward compatibility
    "QuestionCombiner",
    "ResponseFormatter",
    "ConversationExample",
    
    # Training
    "Trainer",
    
    # Inference
    "InferencePipeline",
    "BatchInference",
    
    # Models
    "ModelRegistry",
    "get_supported_models",
    "load_model",
    "apply_lora",
    "get_device",
    
    # Safety
    "SafetyChecker",
    "PIIDetector",
    "BiasMonitor",
    "ClinicalValidator",
]


# Version information
def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "institution": __institution__,
        "features": [
            "1:1 row-to-example mapping",
            "16 question combination styles",
            "Intelligent field ordering",
            "Optional validation split",
            "LoRA efficient fine-tuning",
            "Multi-format support (TEXT, JSON, XML, Markdown)",
            "Safety checks (PII, bias, clinical validation)",
            "Multi-turn conversations"
        ],
        "changelog_v1.0.0": [
            "Added 9 logical reasoning styles",
            "Added field ordering for clinical logic",
            "Made validation truly optional",
            "Enhanced logging and statistics"
        ]
    }