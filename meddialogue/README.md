# MedDialogue Package API Reference

This is the core MedDialogue framework package containing all modules for conversational medical dialogue fine-tuning.

## Package Structure

```
meddialogue/
├── __init__.py          # Main exports and version info
├── core.py              # MedDialogue main interface
├── config.py            # Configuration classes
├── data_prep.py         # Data preparation pipeline
├── train.py             # Training pipeline
├── inference.py         # Inference pipeline
├── models.py            # Model registry and loading
├── safety.py            # Safety checks (PII, bias, validation)
├── utils.py             # Utility functions
├── cli.py               # Command-line interface
└── setup.py             # Package setup
```

## Core API

### MedDialogue

Main interface for training and inference.

```python
from meddialogue import MedDialogue

trainer = MedDialogue(
    task_config: TaskConfig,              # Task definition (required)
    model_type: str = "llama",            # Model type
    safety_config: Optional[SafetyConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    conversation_config: Optional[ConversationConfig] = None,
    output_dir: str = "./output",
    enable_safety: bool = True,
    cuda_device: int = 0,
    verbose: bool = True
)
```

**Methods**:
- `train(data, epochs, batch_size, learning_rate, max_steps)` - Train on DataFrame
- `train_from_csv(csv_path, ...)` - Train from CSV file
- `infer(clinical_note, question, format, return_full_response)` - Single inference
- `batch_infer(clinical_notes, questions, format, show_progress)` - Batch inference
- `load_trained_model(model_path)` - Load saved model for inference
- `get_config_summary()` - Get configuration summary

### TaskConfig

Define your medical task.

```python
from meddialogue import TaskConfig

config = TaskConfig(
    task_name: str,                                # Task identifier
    task_description: str,                         # Task description
    input_field: str = "clinical_note",            # Input column name
    output_fields: List[str],                      # Output field names
    question_templates: Dict[str, List[str]],      # Questions per field
    field_ordering: Dict[str, int] = {},           # Clinical logic ordering
    output_formats: List[OutputFormat] = [...],    # Supported formats
    output_format_ratios: Dict[str, float] = {},   # Format distribution
    medical_terminology: Dict[str, List[str]] = {}, # Domain terms
    diagnostic_codes: List[str] = [],              # ICD/CPT codes
    severity_levels: List[str] = [],               # Severity classifications
    default_system_prompt: str = ""                # System role definition
)
```

### ConversationConfig

Configure conversation generation (v1.0.0 - 1:1 mapping).

```python
from meddialogue import ConversationConfig

config = ConversationConfig(
    single_turn_ratio: float = 0.5,                # 50% single, 50% multi-turn
    max_multi_turns: int = 3,                      # Max conversation turns
    include_typos: bool = True,                    # Typo robustness
    typo_ratio: float = 0.15,                      # 15% with typos
    validation_split: float = 0.2,                 # Validation set size
    logical_style_ratio: float = 0.4,              # 40% logical, 60% grammatical
    context_window_size: int = 64000,              # ~16K tokens
    response_allocation_ratio: float = 0.25,       # Response space
    buffer_ratio: float = 0.10,                    # Safety buffer
    min_question_length: int = 1000,               # Min question chars
    max_question_length: int = 8000,               # Max question chars
    include_followup_questions: bool = True        # Multi-turn enabled
)
```

**Key Changes in v1.0.0**:
- **1:1 mapping**: Each row → ONE conversation (no multiplication)
- **16 question styles**: 7 grammatical + 9 logical reasoning
- **Field ordering**: Clinical logic (assess → diagnose → plan)
- **Optional validation**: Set `validation_split=0.0` to use all data for training

### SafetyConfig

Configure safety checks and compliance.

```python
from meddialogue import SafetyConfig, PIISensitivity

config = SafetyConfig(
    enable_pii_detection: bool = True,
    pii_sensitivity: PIISensitivity = PIISensitivity.HIGH,
    max_pii_threshold: float = 0.95,
    enable_bias_monitoring: bool = True,
    bias_demographic_fields: List[str] = ["age", "gender", "race"],
    enable_clinical_validation: bool = True,
    require_icd_validation: bool = False,
    block_on_safety_failure: bool = False,
    log_safety_events: bool = True,
    custom_pii_patterns: Dict[str, str] = {},
    custom_medical_terms: List[str] = []
)
```

### TrainingConfig

Configure training parameters.

```python
from meddialogue import TrainingConfig

config = TrainingConfig(
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    save_strategy: str = "epoch",
    save_steps: Optional[int] = None,
    save_total_limit: int = 2,
    evaluation_strategy: str = "epoch",
    eval_steps: Optional[int] = None,
    logging_steps: int = 10,
    fp16: bool = False,
    bf16: bool = True,
    max_steps: Optional[int] = None,
    use_full_dataset: bool = True,
    subsample_size: int = 10000
)
```

## Data Preparation

### DataPrep

Main data preparation pipeline.

```python
from meddialogue import DataPrep

prep = DataPrep(task_config, conversation_config)

# Prepare data (1:1 mapping)
train_examples, val_examples = prep.prepare_data(dataframe)

# Convert to tokenized dataset
dataset = prep.prepare_dataset(examples, tokenizer, num_proc=4)
```

### ConversationExample

Training example class.

```python
class ConversationExample:
    conversation: List[Dict[str, str]]  # Chat messages
    metadata: Dict[str, Any]            # Example metadata
```

## Inference

### InferencePipeline

Run inference on trained models.

```python
from meddialogue import InferencePipeline

pipeline = InferencePipeline(
    model,
    tokenizer,
    task_config,
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
    device = torch.device("cuda")
)

# Single inference
response = pipeline.infer(
    clinical_note: str,
    question: Optional[str] = None,
    output_format: OutputFormat = OutputFormat.TEXT,
    return_full_response: bool = False
)

# Batch inference
responses = pipeline.batch_infer(
    clinical_notes: List[str],
    questions: Optional[List[str]] = None,
    output_format: OutputFormat = OutputFormat.TEXT
)
```

## Safety Modules

### SafetyChecker

Comprehensive safety checking.

```python
from meddialogue import SafetyChecker

checker = SafetyChecker(safety_config, task_config)

# Check entire dataset
results = checker.check_dataset(
    data: pd.DataFrame,
    text_column: str = "clinical_note"
)

# Check single text
result = checker.check_text(text: str)
```

### PIIDetector

HIPAA-compliant PII detection.

```python
from meddialogue import PIIDetector

detector = PIIDetector(safety_config)

result = detector.detect(text: str)
# result.has_pii: bool
# result.pii_types: List[str]
# result.confidence: float
# result.anonymized_text: str
```

### BiasMonitor

Demographic bias monitoring.

```python
from meddialogue import BiasMonitor

monitor = BiasMonitor(safety_config)

metrics = monitor.analyze(
    data: pd.DataFrame,
    label_field: str = "diagnosis"
)
# metrics.is_balanced: bool
# metrics.imbalance_score: float
# metrics.warnings: List[str]
# metrics.demographic_distribution: Dict
```

### ClinicalValidator

Medical terminology and code validation.

```python
from meddialogue import ClinicalValidator

validator = ClinicalValidator(safety_config, task_config)

result = validator.validate(
    text: str,
    expected_fields: List[str]
)
# result.is_valid: bool
# result.validated_codes: List[str]
# result.warnings: List[str]
```

## Model Registry

### ModelRegistry

Manage supported models.

```python
from meddialogue import ModelRegistry

# List available models
models = ModelRegistry.list_available_models()

# Get model configuration
config = ModelRegistry.get_config("llama")

# Supported models
# - llama: Llama-3.1, Llama-3.2 (8B, 70B)
# - phi-4: Phi-4 (14B)
# - mistral: Mistral-7B, Mistral-Nemo
# - qwen: Qwen2.5 (7B, 14B, 32B)
```

### Model Loading

```python
from meddialogue import load_model, ModelConfig

model_config = ModelConfig(
    model_name: str,
    model_type: str,
    chat_template: str,
    max_seq_length: int = 16384
)

model, tokenizer = load_model(model_config, max_seq_length)
```

## Enumerations

### OutputFormat

```python
from meddialogue import OutputFormat

OutputFormat.TEXT       # Natural language
OutputFormat.JSON       # Structured JSON
OutputFormat.XML        # XML format
OutputFormat.MARKDOWN   # Markdown format
```

### PIISensitivity

```python
from meddialogue import PIISensitivity

PIISensitivity.LOW      # Minimal PII detection
PIISensitivity.MEDIUM   # Balanced detection
PIISensitivity.HIGH     # Maximum detection (recommended for healthcare)
```

## Usage Examples

### Complete Training Example

```python
from meddialogue import (
    MedDialogue, TaskConfig, ConversationConfig,
    SafetyConfig, TrainingConfig, OutputFormat
)

# 1. Define task
task_config = TaskConfig(
    task_name="malnutrition_assessment",
    input_field="clinical_note",
    output_fields=["diagnosis", "severity", "care_plan"],
    question_templates={
        'diagnosis': ["What is the diagnosis?", "Assess malnutrition"],
        'severity': ["What is the severity?", "Grade severity"],
        'care_plan': ["What do you recommend?", "Treatment plan?"]
    },
    field_ordering={
        'diagnosis': 1,
        'severity': 2,
        'care_plan': 3
    }
)

# 2. Configure conversation generation (v1.0.0 - 1:1 mapping)
conversation_config = ConversationConfig(
    single_turn_ratio=0.5,
    max_multi_turns=3,
    validation_split=0.2,
    logical_style_ratio=0.4
)

# 3. Configure safety
safety_config = SafetyConfig(
    enable_pii_detection=True,
    enable_bias_monitoring=True
)

# 4. Configure training
training_config = TrainingConfig(
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-4
)

# 5. Train
trainer = MedDialogue(
    task_config=task_config,
    conversation_config=conversation_config,
    safety_config=safety_config,
    training_config=training_config,
    model_type="llama"
)

results = trainer.train_from_csv("data.csv")

# 6. Infer
response = trainer.infer("Patient presents...", format="json")
```

### Loading Trained Model

```python
from meddialogue import MedDialogue, TaskConfig

# Define same task config used for training
task_config = TaskConfig(...)

# Initialize
trainer = MedDialogue(task_config=task_config, model_type="llama")

# Load model
trainer.load_trained_model("./output/merged_llama_20250115_120000")

# Inference
response = trainer.infer("Clinical note...", format="json")
```

## Version Information

```python
import meddialogue

print(meddialogue.__version__)        # "1.0.0"
print(meddialogue.__author__)         # "Frederick Gyasi"
print(meddialogue.__email__)          # "gyasi@musc.edu"

# Get detailed version info
info = meddialogue.get_version_info()
```

## Support

- **Main Documentation**: [../README.md](../README.md)
- **Technical Docs**: [../DOCUMENTATION.md](../DOCUMENTATION.md)
- **GitHub**: https://github.com/gyasifred/meddialogue
- **Email**: gyasi@musc.edu

---

**MedDialogue v1.0.0** - General-purpose medical dialogue fine-tuning framework
