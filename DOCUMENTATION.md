# MedDialogue: Complete Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Preparation Pipeline](#data-preparation-pipeline)
4. [Question Generation Strategy](#question-generation-strategy)
5. [Conversation Generation](#conversation-generation)
6. [Training Pipeline](#training-pipeline)
7. [Inference & Multi-Step Reasoning](#inference--multi-step-reasoning)
8. [Safety & Compliance](#safety--compliance)
9. [Configuration Reference](#configuration-reference)
10. [Advanced Usage](#advanced-usage)

---

## Overview

**MedDialogue** is a general-purpose framework for training Large Language Models (LLMs) on ANY medical dialogue task using conversational fine-tuning. Unlike traditional instruction-based fine-tuning, MedDialogue trains models through multi-turn conversations that mirror real clinical interactions.

### Key Principles

1. **Conversational Learning**: Models learn through dialogue, not rigid instruction-response pairs
2. **Clinical Reasoning**: Multi-step reasoning flows (evidence → diagnosis → classification)
3. **Format Flexibility**: Output in text, JSON, XML, or Markdown based on context
4. **Safety First**: HIPAA-compliant PII detection, bias monitoring, clinical validation
5. **General Purpose**: Works for ANY medical task (diagnosis, triage, education, etc.)

### Why MedDialogue?

**Traditional Approach (Instruction Fine-Tuning)**:
```
Input: "Diagnose this patient: [clinical note]"
Output: "Malnutrition Present, Moderate severity"
```
❌ Rigid, brittle, fails on paraphrase

**MedDialogue Approach (Conversational Fine-Tuning)**:
```
Turn 1: "What's the malnutrition status?" → [Full assessment]
Turn 2: "Why do you think that?" → [Clinical reasoning]
Turn 3: "How severe?" → [Severity with evidence]
```
✅ Natural, robust, handles variations

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        MedDialogue                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   TaskConfig │  │ConversationConfig│ │SafetyConfig│    │
│  │              │  │              │  │              │    │
│  │ - Task def   │  │ - Conv style │  │ - PII check  │    │
│  │ - Questions  │  │ - Turns      │  │ - Bias mon   │    │
│  │ - Output fmt │  │ - Typos      │  │ - Validation │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │            │
│         └──────────┬───────┴──────────────────┘            │
│                    ▼                                       │
│         ┌─────────────────────┐                           │
│         │     DataPrep        │                           │
│         │                     │                           │
│         │ - QuestionCombiner  │                           │
│         │ - ResponseFormatter │                           │
│         │ - ConversationGen   │                           │
│         └──────────┬──────────┘                           │
│                    ▼                                       │
│         ┌─────────────────────┐                           │
│         │  Training Examples  │                           │
│         │  (Conversations)    │                           │
│         └──────────┬──────────┘                           │
│                    ▼                                       │
│         ┌─────────────────────┐                           │
│         │   Trainer (LoRA)    │                           │
│         └──────────┬──────────┘                           │
│                    ▼                                       │
│         ┌─────────────────────┐                           │
│         │   Fine-tuned Model  │                           │
│         └──────────┬──────────┘                           │
│                    ▼                                       │
│         ┌─────────────────────┐                           │
│         │ InferencePipeline   │                           │
│         └─────────────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Preparation Pipeline

### Overview

The data preparation pipeline transforms raw clinical data (CSV rows) into rich conversational training examples. **Important**: MedDialogue uses a **1:1 row-to-example mapping** — each CSV row creates ONE training conversation, not multiple copies.

### Pipeline Stages

#### 1. Data Loading & Validation
```python
from meddialogue import MedDialogue, TaskConfig

task_config = TaskConfig(
    task_name="malnutrition_assessment",
    input_field="clinical_note",
    output_fields=["diagnosis", "severity", "recommendations"]
)

trainer = MedDialogue(task_config=task_config)
trainer.train_from_csv("data.csv")
```

**What happens**:
- Loads CSV with pandas
- Validates required columns exist
- Checks for missing values
- Runs safety checks (PII, bias)

#### 2. Question Generation

For each row, the system:
1. Identifies output fields (e.g., diagnosis, severity)
2. Generates question variations for each field
3. Combines questions using grammatical OR logical styles

**Example**:
```python
# Input fields: diagnosis, severity, recommendations
# Output questions:
"What is the diagnosis?"
"Assess malnutrition status and provide severity"
"Explain reasoning and recommend treatment"
```

#### 3. Response Formatting

Responses are formatted in multiple styles:
- **Natural Language**: Clinical narrative with structure
- **JSON**: `{"diagnosis": "Present", "severity": "Moderate"}`
- **XML**: `<assessment><diagnosis>Present</diagnosis>...</assessment>`
- **Markdown**: Formatted with headers and lists

#### 4. Conversation Structure

Each example becomes either:

**Single-Turn Conversation** (50% by default):
```
System: [Role definition]
User: [Clinical note] + [Combined question]
Assistant: [Complete response]
```

**Multi-Turn Conversation** (50% by default):
```
System: [Role definition]

Turn 1:
User: [Clinical note] + [Question 1]
Assistant: [Response 1]

Turn 2:
User: [Follow-up question 2]
Assistant: [Response 2]

Turn 3:
User: [Follow-up question 3]
Assistant: [Response 3]
```

---

## Question Generation Strategy

### Question Types

MedDialogue generates 16 distinct question styles, divided into two categories:

#### Grammatical Variations (7 styles)
These rephrase the same question grammatically:

1. **Direct**: "What is the diagnosis?"
2. **Interrogative**: "Is malnutrition present?"
3. **Imperative**: "Assess malnutrition status"
4. **Conversational**: "Can you tell me about the malnutrition status?"
5. **Formal**: "Provide a clinical assessment of malnutrition"
6. **Clinical Jargon**: "Evaluate for pediatric undernutrition"
7. **With Typos** (15% of the time): "What is the diagnsois?" (robustness training)

#### Logical Combinations (9 styles)
These combine multiple fields in logical order:

1. **Sequential**: "First assess diagnosis, then severity"
2. **With Evidence**: "Provide diagnosis with clinical evidence"
3. **With Reasoning**: "Explain diagnosis and your clinical reasoning"
4. **With Summary**: "Give diagnosis and summarize key findings"
5. **With Recommendations**: "Diagnose and recommend treatment"
6. **Comparative**: "Compare current findings to diagnostic criteria"
7. **Hierarchical**: "Evaluate systematically: assessment → diagnosis → plan"
8. **Comprehensive**: "Provide complete evaluation with all details"
9. **Targeted**: "Focus on [specific field]"

### Logical Style Ratio

The `ConversationConfig.logical_style_ratio` controls the mix:
```python
ConversationConfig(
    logical_style_ratio=0.4  # 40% logical, 60% grammatical
)
```

**Why this matters**:
- **Grammatical variations** teach the model to understand paraphrases
- **Logical combinations** teach the model to reason across multiple fields
- Balance prevents overfitting to specific patterns

---

## Conversation Generation

### Conversation Types

#### Single-Turn Conversations

Best for:
- Simple classification tasks
- Direct question-answer pairs
- Rapid inference scenarios

Structure:
```
System: You are a pediatric gastroenterologist...

User: [Clinical note]

What is the malnutrition status and severity?