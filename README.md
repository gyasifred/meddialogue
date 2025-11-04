# MedDialogue: General-Purpose Medical Dialogue Fine-Tuning Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Institution: MUSC](https://img.shields.io/badge/Institution-MUSC-green.svg)](https://medicine.musc.edu/departments/biomedical-informatics)

**A general-purpose framework for fine-tuning large language models on ANY medical dialogue task through conversational learning.**

---

## What is MedDialogue?

MedDialogue is a **general-purpose framework** for training Large Language Models (LLMs) on medical conversational tasks. It teaches models to engage in natural clinical dialogues rather than memorizing rigid prompt-response patterns.

### Why "General-Purpose"?

MedDialogue is NOT limited to any specific medical domain. It works for:
- **Diagnosis & Classification**: Malnutrition, diabetes, sepsis, cancer staging, etc.
- **Clinical Triage**: Emergency severity, surgical risk, ICU admission
- **Patient Education**: Treatment explanations, medication counseling
- **Documentation**: Progress notes, discharge summaries, referral letters
- **Research**: Literature review, cohort extraction, quality metrics

The included malnutrition example (`train_malnutrition.py`) is just a **demonstration** - the framework itself (`meddialogue/`) contains no domain-specific code.

---

## Key Features

### Conversational Learning

**Traditional Instruction Fine-Tuning (Brittle)**:
```
Input: "Diagnose this patient: [note]"
Output: "Malnutrition Present"
```
Fails on: "Is there malnutrition?", "What's the nutritional status?", etc.

**MedDialogue Conversational Fine-Tuning (Robust)**:
```
Turn 1: "What's the malnutrition status?" → [Complete assessment]
Turn 2: "Why do you think that?" → [Clinical reasoning]
Turn 3: "How severe is it?" → [Severity with evidence]
```
Handles natural variations, follow-ups, and context

### Core Capabilities

- **1:1 Data Mapping**: Each clinical note → ONE training conversation (no artificial multiplication)
- **16 Question Styles**: 7 grammatical variations + 9 logical reasoning patterns
- **Smart Field Ordering**: Clinical logic (assess → diagnose → plan)
- **Multi-Format Output**: TEXT, JSON, XML, Markdown
- **Safety First**: HIPAA-compliant PII detection, bias monitoring, clinical validation
- **Memory Efficient**: LoRA fine-tuning with automatic OOM recovery
- **Multi-Model Support**: Llama, Phi-4, Mistral, Qwen families

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/gyasifred/meddialogue.git
cd meddialogue

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers datasets unsloth peft trl pandas numpy scikit-learn tqdm

# Install package
pip install -e .
```

### Your First Training

```python
from meddialogue import MedDialogue, TaskConfig, ConversationConfig

# 1. Define your task
task_config = TaskConfig(
    task_name="your_medical_task",
    task_description="Describe your task",
    input_field="clinical_note",
    output_fields=["field1", "field2", "field3"],
    question_templates={
        'field1': [
            "What is field1?",
            "Determine field1",
            "Assess field1"
        ],
        'field2': [...],
        'field3': [...]
    }
)

# 2. Configure conversation generation
conversation_config = ConversationConfig(
    single_turn_ratio=0.5,      # 50% single-turn, 50% multi-turn
    max_multi_turns=3,           # Up to 4 conversation turns
    validation_split=0.2,        # 20% for validation
    logical_style_ratio=0.4      # 40% logical reasoning styles
)

# 3. Train
trainer = MedDialogue(
    task_config=task_config,
    conversation_config=conversation_config,
    model_type="llama"
)
trainer.train_from_csv("your_data.csv", epochs=3)

# 4. Inference
response = trainer.infer("Clinical note...", format="json")
print(response)
```

### Running the Malnutrition Example

```bash
# Train on malnutrition data
python train_malnutrition.py \
    --csv data.csv \
    --epochs 3 \
    --model llama \
    --output ./models

# Evaluate
python evaluate_malnutrition.py \
    --model ./models/llama/merged_llama_* \
    --csv test.csv \
    --output ./results
```

---

## Data Format

Your CSV should have an input field (e.g., `clinical_note`) and your defined output fields:

```csv
clinical_note,diagnosis,severity,recommendations
"Patient presents with BMI <5th percentile...","Present","moderate","Nutritional supplementation..."
"Normal growth parameters...","Absent","null","Continue monitoring"
```

**Required**:
- Input field (default: `clinical_note`)
- Output fields you defined in `TaskConfig`

**Optional but Recommended**:
- Additional context fields
- Reasoning fields for explainability

---

## How It Works

### 1. Question Generation

For each output field, the framework generates multiple question variations:

**Grammatical Variations (7 styles)**:
- Direct: "What is the diagnosis?"
- Interrogative: "Is malnutrition present?"
- Imperative: "Assess malnutrition status"
- Conversational: "Can you tell me about the diagnosis?"
- Formal: "Provide clinical assessment"
- With Jargon: "Evaluate for pediatric undernutrition"
- With Typos (15%): "What is the diagnsois?" (robustness)

**Logical Combinations (9 styles)**:
- Sequential: "First assess X, then Y"
- With Evidence: "Provide X with supporting evidence"
- With Reasoning: "Explain X and your reasoning"
- Comprehensive: "Complete evaluation of all aspects"
- And 5 more patterns...

### 2. Conversation Structure

**Single-Turn** (Direct):
```
System: You are a medical expert...
User: [Clinical note] + [Question]
Assistant: [Complete response with all fields]
```

**Multi-Turn** (Natural Dialogue):
```
System: You are a medical expert...

Turn 1:
User: [Clinical note] + "What's the diagnosis?"
Assistant: [Diagnosis with reasoning]

Turn 2:
User: "How severe is it?"
Assistant: [Severity classification]

Turn 3:
User: "What do you recommend?"
Assistant: [Treatment plan]
```

### 3. Response Formats

Models learn to output in multiple formats:

**TEXT**: Natural clinical narrative
**JSON**: `{"diagnosis": "Present", "severity": "moderate"}`
**XML**: `<assessment><diagnosis>Present</diagnosis>...</assessment>`
**Markdown**: Formatted with headers and lists

---

## Documentation

### Complete Documentation
- **Technical Documentation**: [DOCUMENTATION.md](DOCUMENTATION.md)
- **GitHub Pages**: https://gyasifred.github.io/meddialogue/
- **API Reference**: [meddialogue/README.md](meddialogue/README.md)

### Quick Links
- [Installation Guide](DOCUMENTATION.md#installation)
- [Configuration Reference](DOCUMENTATION.md#configuration-reference)
- [Safety & Compliance](DOCUMENTATION.md#safety--compliance)
- [Advanced Usage](DOCUMENTATION.md#advanced-usage)
- [Troubleshooting](DOCUMENTATION.md#troubleshooting)

---

## Examples

### Malnutrition Assessment
The included example demonstrates:
- 11 comprehensive clinical fields
- 110 question variations (10 per field)
- Temporal reasoning and prediction
- Multi-step clinical logic
- Field ordering (assess → reason → diagnose → classify → plan)

See [`train_malnutrition.py`](train_malnutrition.py) and [`evaluate_malnutrition.py`](evaluate_malnutrition.py)

### Creating Your Own Task

1. **Define your output fields**:
   ```python
   output_fields = ["diagnosis", "risk_level", "recommendations"]
   ```

2. **Create question templates for each field**:
   ```python
   question_templates = {
       'diagnosis': [
           "What is the diagnosis?",
           "Diagnose this patient",
           "Assess the clinical condition"
       ],
       'risk_level': [...],
       'recommendations': [...]
   }
   ```

3. **Set field ordering for logical flow**:
   ```python
   field_ordering = {
       'diagnosis': 1,        # Diagnose first
       'risk_level': 2,       # Then assess risk
       'recommendations': 3   # Finally, recommend treatment
   }
   ```

4. **Train**:
   ```python
   trainer = MedDialogue(task_config=task_config)
   trainer.train_from_csv("your_data.csv")
   ```

---

## Requirements

**System**:
- Python 3.8+
- CUDA-capable GPU (recommended: 24GB+ VRAM)
- 32GB+ RAM
- Linux or macOS (Windows via WSL2)

**Key Dependencies**:
```
torch>=2.8.0
transformers>=4.55.0
unsloth>=2025.9.7
peft>=0.17.0
datasets>=3.6.0
pandas>=2.3.0
scikit-learn>=1.7.0
```

See full requirements in [DOCUMENTATION.md](DOCUMENTATION.md#requirements)

---

## Safety & Compliance

### HIPAA-Compliant PII Detection
- Automatic detection of PHI (names, MRNs, dates, addresses)
- Configurable sensitivity levels
- Anonymization with redaction
- Audit logging

### Bias Monitoring
- Demographic distribution tracking
- Label imbalance detection
- Fairness metrics
- Warning system

### Clinical Validation
- Medical terminology validation
- ICD code verification
- Custom medical term support
- Safety event logging

---

## Model Support

### Supported Model Families
- **Llama**: Llama-3.1, Llama-3.2 (8B, 70B)
- **Phi**: Phi-4 (14B)
- **Mistral**: Mistral-7B, Mistral-Nemo
- **Qwen**: Qwen2.5 (7B, 14B, 32B)

### Training Performance
| Model | VRAM | Batch Size | Time (1000 examples) |
|-------|------|------------|----------------------|
| Llama-3.1-8B | 24GB | 2 | ~45 min |
| Phi-4 (14B) | 24GB | 1 | ~60 min |
| Mistral-7B | 24GB | 2 | ~40 min |
| Qwen2.5-7B | 24GB | 2 | ~42 min |

---

## Project Structure

```
meddialogue/
├── meddialogue/              # Core framework package
│   ├── core.py               # Main MedDialogue interface
│   ├── config.py             # Configuration classes
│   ├── data_prep.py          # Data preparation pipeline
│   ├── train.py              # Training pipeline
│   ├── inference.py          # Inference pipeline
│   ├── models.py             # Model registry and loading
│   ├── safety.py             # Safety checks (PII, bias)
│   └── utils.py              # Utilities
├── train_malnutrition.py     # Example: Training script
├── evaluate_malnutrition.py  # Example: Evaluation script
├── docs/                     # GitHub Pages documentation
├── DOCUMENTATION.md          # Complete technical docs
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## Contributing

We welcome contributions! Areas of interest:
1. **New Medical Tasks**: Add examples for different domains
2. **Model Support**: Add new model families
3. **Safety Features**: GDPR compliance, FHIR validation
4. **Performance**: Optimization, memory efficiency
5. **Documentation**: Tutorials, translations

See [DOCUMENTATION.md](DOCUMENTATION.md#contributing) for guidelines

---

## Citation

If you use MedDialogue in your research, please cite:

```bibtex
@software{meddialogue2025,
  author = {Gyasi, Frederick},
  title = {MedDialogue: General-Purpose Medical Dialogue Fine-Tuning Framework},
  year = {2025},
  publisher = {Medical University of South Carolina},
  institution = {Biomedical Informatics Center},
  url = {https://github.com/gyasifred/meddialogue}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Author

**Frederick Gyasi**
Email: gyasi@musc.edu
Institution: Medical University of South Carolina
Department: Biomedical Informatics Center, ClinicalNLP Lab

---

## Acknowledgments

- **Institution**: Medical University of South Carolina, Biomedical Informatics Center
- **Framework**: Built on Unsloth, Transformers, and PyTorch
- **Inspiration**: Clinical need for robust, conversational healthcare AI

---

## Support

- **GitHub Issues**: [github.com/gyasifred/meddialogue/issues](https://github.com/gyasifred/meddialogue/issues)
- **Email**: gyasi@musc.edu
- **Documentation**: [DOCUMENTATION.md](DOCUMENTATION.md)
- **Institution**: [MUSC Biomedical Informatics](https://medicine.musc.edu/departments/biomedical-informatics)

---

**MedDialogue**: Transforming healthcare AI through conversational learning, one dialogue at a time.
