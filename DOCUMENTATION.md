# MedDialogue: Complete Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [How Conversations Are Generated](#how-conversations-are-generated)
5. [Configuration Guide](#configuration-guide)
6. [Malnutrition Example Explained](#malnutrition-example-explained)
7. [Creating Your Own Task](#creating-your-own-task)
8. [Safety & Compliance](#safety--compliance)
9. [Training & Inference](#training--inference)
10. [Troubleshooting](#troubleshooting)

---

## Overview

**MedDialogue** is a general-purpose framework for training Large Language Models on ANY medical dialogue task through conversational fine-tuning.

### Key Principles

1. **1:1 Data Mapping**: Each clinical note → ONE training conversation (no artificial multiplication)
2. **Natural Variation**: 16 question styles create diversity without data duplication
3. **Conversational Learning**: Multi-turn dialogues mirror real clinical interactions
4. **Format Flexibility**: Output in TEXT, JSON, XML, or Markdown
5. **Safety First**: HIPAA-compliant PII detection, bias monitoring
6. **General Purpose**: Works for ANY medical task

### Why Conversational Fine-Tuning?

**Traditional Instruction Fine-Tuning** ❌:
```
Input: "Diagnose: [note]"
Output: "Malnutrition Present"
```
- Fails on paraphrase
- Rigid prompt dependency
- No multi-turn context

**MedDialogue Conversational Fine-Tuning** ✅:
```
Turn 1: "What's the status?" → [Assessment]
Turn 2: "Why?" → [Reasoning]
Turn 3: "How severe?" → [Severity]
```
- Robust to variations
- Natural dialogue
- Context preservation

---

## Installation

### Requirements

**System**:
- Python 3.8+
- CUDA GPU (24GB+ VRAM recommended)
- 32GB+ RAM
- Linux or macOS (Windows via WSL2)

**Dependencies**:
```
torch>=2.8.0
transformers>=4.55.0
unsloth>=2025.9.7
peft>=0.17.0
datasets>=3.6.0
pandas>=2.3.0
scikit-learn>=1.7.0
tqdm
```

### Install

```bash
git clone https://github.com/gyasifred/meddialogue.git
cd meddialogue

python -m venv venv
source venv/bin/activate

pip install torch transformers datasets unsloth peft trl pandas numpy scikit-learn tqdm
pip install -e .
```

### Verify

```bash
python -c "import meddialogue; print(meddialogue.__version__)"
# Output: 1.0.0
```

---

## Quick Start

### Minimal Example

```python
from meddialogue import MedDialogue, TaskConfig, ConversationConfig

# 1. Define task
task_config = TaskConfig(
    task_name="your_task",
    input_field="clinical_note",
    output_fields=["diagnosis", "severity"],
    question_templates={
        'diagnosis': ["What is the diagnosis?", "Assess diagnosis"],
        'severity': ["What is severity?", "Grade severity"]
    }
)

# 2. Configure conversations
conversation_config = ConversationConfig(
    single_turn_ratio=0.5,
    max_multi_turns=3,
    validation_split=0.2
)

# 3. Train
trainer = MedDialogue(
    task_config=task_config,
    conversation_config=conversation_config,
    model_type="llama"
)
trainer.train_from_csv("data.csv", epochs=3)

# 4. Infer
response = trainer.infer("Patient presents...", format="json")
print(response)
```

### Data Format

CSV with input field + output fields:

```csv
clinical_note,diagnosis,severity
"Patient with BMI <5th percentile...","Present","moderate"
"Normal growth...","Absent","null"
```

---

## How Conversations Are Generated

### Critical Understanding: 1:1 Mapping

**Each CSV row generates exactly ONE training conversation.**

- 100 rows → 100 training examples
- No data multiplication or duplication
- Variation comes from 16 question styles, not copying data

### Process for Each Row

```python
# Input: 1 CSV row
row = {
    'clinical_note': "Patient presents with...",
    'diagnosis': "Present",
    'severity': "moderate",
    'care_plan': "Nutritional supplementation..."
}

# Output: 1 conversation (either single-turn OR multi-turn)
```

#### Path 1: Single-Turn (50% of rows with `single_turn_ratio=0.5`)

```
System: You are a medical expert...

User: [Clinical note]

[ONE combined question for ALL fields using style 1-16]