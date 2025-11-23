#!/usr/bin/env python
"""
Malnutrition Assessment Training Script - v1.0.0 with Reasoning & Temporal Focus
================================================================================

INPUT: ONE CSV FILE - Framework handles train/validation split automatically
  • Set --validation_split 0.0 (default) → Use ALL data for training
  • Set --validation_split 0.2 → Automatic 80/20 split (framework does this)
  • No need for separate train.csv and val.csv files!

ENHANCEMENTS v1.0.0:
1. **11 comprehensive fields** including new 'clinical_symptoms_and_signs'
2. **10 reasoning-oriented questions per field** (110 total variations)
3. **Temporal reasoning emphasis**: "How has it changed?", "What's the trajectory?"
4. **Predictive questions**: "What will happen?", "What should we expect?"
5. **Action-oriented**: "What should we order?", "What's the monitoring schedule?"
6. **Clinical reasoning**: "Why?", "How did you decide?", "What's the rationale?"
7. **Logical field ordering**: Assess → Reason → Diagnose → Classify → Act
8. **Field ordering for reasoning styles**: Reasoning comes BEFORE status

Focus: Train models to REASON and PREDICT, not just EXTRACT.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import os
import logging
import argparse
from typing import Dict, List, Optional
import pandas as pd

from meddialogue import (
    MedDialogue,
    TaskConfig,
    SafetyConfig,
    TrainingConfig,
    ConversationConfig,
    OutputFormat
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class MalnutritionTaskConfig:
    """
    Comprehensive malnutrition assessment with temporal reasoning and prediction.
    
    NEW in v1.0.0:
    - 11 fields including clinical_symptoms_and_signs
    - 10 questions per field (110 total variations)
    - Reasoning-oriented questions: "Why?", "How did you decide?"
    - Temporal questions: "How has it changed?", "What's the trajectory?"
    - Predictive questions: "What should we expect?", "What will happen?"
    - Action-oriented: "What should we order?", "What's the monitoring plan?"
    - Field ordering: Reasoning comes BEFORE status (clinical logic)
    
    Covers 11 domains:
    1. Case Presentation - Context and timeline
    2. Clinical Symptoms & Signs (NEW) - Temporal symptom capture
    3. Growth & Anthropometrics - Trends and trajectories
    4. Physical Exam - Progression over time
    5. Nutrition & Intake - Patterns and changes
    6. Labs & Screening - Trends and recommendations
    7. Diagnosis & Reasoning - Clinical synthesis (BEFORE status)
    8. Malnutrition Status - Final classification (AFTER reasoning)
    9. Care Plan - Actions, monitoring, escalation
    10. Social Context - Barriers and interventions
    11. Clinical Insights - Teaching and prognosis
    """
    
    # [Previous QUESTION_TEMPLATES code remains exactly the same - 110 questions]
    QUESTION_TEMPLATES = {
        'case_presentation': [
            "Summarize the case presentation with timeline and key events.",
            "What brings this patient in? Tell me about the clinical context and how things unfolded.",
            "Walk me through the case: setting, chief concern, and temporal progression.",
            "Give me the patient's story with dates and progression of concerns.",
            "Is this a single-point assessment or serial/longitudinal? What's the temporal context?",
            "How did the family describe the problem? What's their perspective and timeline?",
            "What's the assessment setting and what temporal information is available?",
            "Based on the presentation, what's your initial assessment and what additional history would help?",
            "What temporal patterns emerge from the presentation? What questions remain?",
            "Describe the case presentation and identify what information is missing for temporal reasoning."
        ],
        
        'clinical_symptoms_and_signs': [
            "What are ALL the clinical symptoms and signs? For EACH symptom, give onset dates and progression.",
            "Describe the symptom trajectory: When did symptoms start? How have they progressed? Include dates.",
            "Document GI symptoms (vomiting, diarrhea, pain) with frequency, severity, and temporal pattern.",
            "What systemic symptoms are present (fever, fatigue, irritability)? Describe onset and duration.",
            "How have symptoms changed over time? Describe the trajectory: worsening, stable, or improving?",
            "What's the temporal relationship between symptoms? Which came first? How did they evolve?",
            "Based on symptom patterns, what's your differential diagnosis and reasoning?",
            "If symptoms aren't fully documented, what specific symptoms should be assessed and why?",
            "What symptom progression would you expect? What should we monitor?",
            "Looking at the symptom pattern, what complications should we watch for and on what timeline?"
        ],
        
        'growth_and_anthropometrics': [
            "What are ALL anthropometric measurements with DATES? Calculate trends and trajectories.",
            "Give me weight data with dates and percentiles. Calculate absolute change, percentage loss, and rate.",
            "Describe the growth trajectory: What's the pattern? Progressive decline, stable, or improving?",
            "What's the weight-for-height z-score trajectory? How has it changed over serial measurements?",
            "Analyze the growth pattern: What's concerning? What's the velocity? Compare to expected growth.",
            "How do current measurements compare to previous visits? Calculate the rate of change.",
            "Based on growth data, what severity would you assign and why? Use ASPEN guidelines.",
            "What measurements are missing? What should be obtained and why would it help?",
            "What's the expected growth trajectory if we intervene now versus wait? Project forward.",
            "What monitoring schedule for anthropometrics would you recommend and what triggers concern?"
        ],
        
        'physical_exam': [
            "Describe nutrition-focused physical exam findings. If serial exams, show progression with dates.",
            "What signs of malnutrition are present? Muscle wasting? Fat loss? Edema? Describe severity.",
            "Document general appearance and any changes noted across encounters.",
            "What physical findings support your diagnosis? How do they correlate with other data?",
            "If you had serial exams, how have physical findings changed? Describe the trajectory.",
            "Which exam findings are most concerning? Why? What do they tell you about severity?",
            "How do exam findings align with anthropometric data? Any discrepancies to explain?",
            "If a detailed NFPE wasn't performed, what specific components should be examined and why?",
            "What physical exam changes would you expect with treatment? What's the timeline?",
            "What physical findings should trigger immediate escalation of care?"
        ],
        
        'nutrition_and_intake': [
            "Describe nutrition intake patterns over time with dates and percentages of needs met.",
            "How has intake changed? Give me temporal trends: 'Was X% in December, Y% in January, Z% in February'.",
            "What's preventing adequate intake? Describe barriers and their duration.",
            "Quantify intake: How much, what types, oral vs enteral? Show changes over encounters.",
            "Analyze the intake trajectory: Is it declining, stable, or improving? At what rate?",
            "How does intake correlate with weight trajectory? Does it explain the growth pattern?",
            "If intake is inadequate, what's your reasoning about underlying causes?",
            "If intake isn't quantified, how should we assess it? What data would you collect?",
            "What's your plan to improve intake? Predict response: What should we see and by when?",
            "If intake doesn't improve in 7 days, what's the escalation plan? When do we consider tube feeding?"
        ],
        
        'labs_and_screening': [
            "What labs are available with DATES? For each lab, describe the trend and rate of change.",
            "Describe laboratory trajectories: 'Albumin was X on date1, Y on date2, Z on date3 - declining 16% over 2 months'.",
            "What micronutrient labs are available? Show temporal patterns and clinical significance.",
            "Are there any screening scores (PYMS, STRONG-kids)? What do they suggest?",
            "Analyze lab trends: Which are most concerning? What do they tell you about severity and risk?",
            "How do lab findings correlate with clinical picture? Any surprises or discrepancies?",
            "What labs SHOULD be ordered and why? Provide specific tests with clinical rationale.",
            "If this is confirmed malnutrition, what's your lab monitoring schedule? Baseline, then serial frequency?",
            "What lab changes would indicate successful treatment? Predict expected trends.",
            "If you suspect refeeding syndrome, what labs should you monitor and how frequently?"
        ],
        
        'diagnosis_and_reasoning': [
            "What's your diagnosis with complete clinical reasoning? Synthesize ALL evidence temporally.",
            "Walk me through your diagnostic thought process: What's the diagnosis, severity, and etiology?",
            "How did you arrive at this diagnosis? What evidence supports it? Address temporal patterns.",
            "What diagnostic criteria are you using (ASPEN, WHO)? How does this case fit?",
            "If you have incomplete data, reason through it: What converging evidence do you have?",
            "What's your differential diagnosis? Why did you settle on this specific diagnosis?",
            "How confident are you in this diagnosis? What additional data would increase certainty?",
            "How did you determine severity? Walk through the criteria and supporting evidence.",
            "What's the etiology: illness-related, intake-related, or mixed? Explain your reasoning.",
            "Based on temporal patterns, is this acute, acute-on-chronic, or chronic? Justify."
        ],
        
        'malnutrition_status': [
            "Is malnutrition present or absent? State clearly.",
            "Does this patient have malnutrition? Yes or no, with brief justification.",
            "What's the malnutrition status classification?",
            "Based on your assessment, is malnutrition present?",
            "Would you diagnose this patient with malnutrition?",
            "Is there evidence of malnutrition in this case?",
            "Does this patient meet diagnostic criteria for malnutrition?",
            "What's your final classification: malnutrition present or absent?",
            "Classify malnutrition status using clinical judgment and criteria.",
            "State malnutrition status with confidence level in your classification."
        ],
        
        'care_plan': [
            "What's your comprehensive care plan? Include goals, interventions with doses, and monitoring schedule.",
            "Give me specific interventions: What should we do? Include formulations, doses, frequencies.",
            "What's the monitoring schedule? Week 1: when? Weeks 2-4: how often? Months 2-3: frequency?",
            "Describe your follow-up plan with specific dates and intervals. What do we check at each visit?",
            "What's the lab monitoring schedule? Baseline panel now, then serial labs when?",
            "What anthropometric monitoring? Weight how often? When do we remeasure height?",
            "What's the escalation plan? If <50g gain by Day 7, then what? Week 2 if no response?",
            "What happens if the plan doesn't work? Backup strategies? When do we escalate to specialist?",
            "Predict the recovery timeline: What should we see Week 1? Week 2? Month 1?",
            "What complications should we watch for? On what timeline? What are the red flags?"
        ],
        
        'social_context': [
            "What social factors are relevant? Describe any barriers to care.",
            "Tell me about the family situation, resources, and any access issues.",
            "What social determinants of health are affecting this case?",
            "Are there financial, housing, or food security concerns? How long have they existed?",
            "How have social circumstances changed over time? Any new barriers or improvements?",
            "What interventions have been tried to address social barriers? Results?",
            "If social history isn't documented, what should be assessed and why?",
            "What social support services should be offered? WIC? Food assistance? Case management?",
            "What social barriers might affect the care plan? How can we address them?",
            "What resources would most help this family? Connect them to what services?"
        ],
        
        'clinical_insights': [
            "Summarize key clinical insights and teaching points from this case.",
            "What are the most important takeaways? What would you teach about this case?",
            "Synthesize the temporal patterns: What's the big picture trajectory?",
            "What's the prognosis? What should we expect for this patient over the next 3-6 months?",
            "Predict the clinical course: Best case? Expected case? Worst case?",
            "What are the key decision points coming up? What will determine the path forward?",
            "What guidelines informed your approach (ASPEN, WHO, AND)? Why those choices?",
            "How does this case illustrate important clinical principles? What are the pearls?",
            "What complications should we anticipate? What's the risk-benefit of different approaches?",
            "What would make you worried? What temporal patterns would trigger concern?"
        ]
    }
    
    # [Rest of the class remains the same - medical terminology, codes, fields, etc.]
    MEDICAL_TERMINOLOGY = {
        'anthropometric': [
            'BMI', 'weight', 'height', 'percentile', 'z-score', 'WHZ', 'WAZ', 'HAZ', 'BAZ',
            'weight-for-height', 'weight-for-age', 'height-for-age', 'BMI-for-age',
            'MUAC', 'mid-upper arm circumference', 'head circumference', 'occipitofrontal circumference',
            'growth velocity', 'weight velocity', 'height velocity', 'growth rate',
            'weight loss percentage', 'weight gain velocity', 'weight trajectory',
            'SGA', 'AGA', 'LGA', 'small for gestational age', 'appropriate for gestational age',
            'faltering growth', 'growth deceleration', 'crossing percentiles'
        ],
        'nutritional_indicators': [
            'protein-energy malnutrition', 'PEM', 'PCM', 'protein-calorie malnutrition',
            'undernutrition', 'malnutrition', 'growth failure', 'failure to thrive', 'FTT',
            'wasting', 'stunting', 'underweight', 'SAM', 'MAM', 'severe acute malnutrition',
            'moderate acute malnutrition', 'chronic malnutrition', 'acute malnutrition',
            'marasmus', 'kwashiorkor', 'marasmic-kwashiorkor',
            'cachexia', 'sarcopenia'
        ],
        'clinical_assessment': [
            'muscle wasting', 'temporal wasting', 'fat loss', 'subcutaneous fat loss',
            'edema', 'pitting edema', 'bipedal edema', 'nutritional edema',
            'NFPE', 'nutrition-focused physical exam',
            'micronutrient deficiency', 'vitamin deficiency', 'mineral deficiency',
            'iron deficiency', 'vitamin A deficiency', 'vitamin D deficiency',
            'zinc deficiency', 'folate deficiency', 'B12 deficiency',
            'glossitis', 'cheilosis', 'angular stomatitis', 'pallor', 'dermatitis',
            'rickets', 'scurvy', 'beriberi', 'pellagra',
            'hair changes', 'flag sign', 'skin changes', 'nail changes'
        ],
        'symptoms': [
            'vomiting', 'emesis', 'diarrhea', 'constipation', 'reflux', 'GERD',
            'abdominal pain', 'nausea', 'early satiety', 'dysphagia',
            'poor appetite', 'anorexia', 'food refusal', 'feeding difficulty',
            'fatigue', 'lethargy', 'weakness', 'irritability',
            'fever', 'infection', 'recurrent infections'
        ],
        'feeding': [
            'enteral nutrition', 'parenteral nutrition', 'TPN', 'total parenteral nutrition',
            'NG tube', 'nasogastric tube', 'G-tube', 'gastrostomy', 'PEG',
            'oral intake', 'PO intake', 'NPO', 'nothing by mouth',
            'formula', 'infant formula', 'breast milk', 'breastfeeding',
            'fortification', 'calorie fortification', 'protein fortification',
            'feeding schedule', 'feeding frequency', 'feeding volume',
            'bolus feeding', 'continuous feeding', 'nocturnal feeding'
        ],
        'interventions': [
            'oral supplement', 'nutritional supplement', 'Pediasure', 'Boost', 'Ensure',
            'high-calorie diet', 'high-protein diet', 'high-energy diet',
            'refeeding syndrome', 'refeeding protocol', 'nutrition support',
            'dietitian consult', 'nutrition consult', 'feeding therapy',
            'growth monitoring', 'nutrition monitoring', 'weight checks',
            'caloric goal', 'protein goal', 'catch-up growth',
            'appetite stimulant', 'cyproheptadine', 'mirtazapine'
        ],
        'lab_tests': [
            'CBC', 'complete blood count', 'CMP', 'comprehensive metabolic panel',
            'BMP', 'basic metabolic panel', 'electrolytes',
            'prealbumin', 'albumin', 'total protein', 'transthyretin',
            'vitamin D', '25-OH vitamin D', 'zinc', 'zinc level',
            'iron studies', 'ferritin', 'transferrin', 'TIBC', 'iron saturation',
            'vitamin A', 'retinol', 'folate', 'folic acid', 'B12', 'cobalamin',
            'magnesium', 'phosphorus', 'calcium', 'PTH',
            'IGF-1', 'IGFBP-3', 'thyroid function', 'TSH',
            'inflammatory markers', 'CRP', 'ESR'
        ],
        'guidelines': [
            'ASPEN', 'American Society for Parenteral and Enteral Nutrition',
            'WHO', 'World Health Organization',
            'AND', 'Academy of Nutrition and Dietetics',
            'PYMS', 'Pediatric Yorkhill Malnutrition Score',
            'STRONG-kids', 'STAMP', 'Screening Tool for Assessment of Malnutrition in Paediatrics',
            'Growth chart', 'CDC growth chart', 'WHO growth standards'
        ]
    }
    
    DIAGNOSTIC_CODES = [
        'E40', 'E41', 'E42', 'E43', 'E44.0', 'E44.1', 'E45', 'E46',
        'R62.50', 'R62.51', 'R62.52', 'R62.7', 'R63.3', 'R63.4',
        'E64.0', 'E64.1', 'E64.2', 'E64.3', 'E64.9'
    ]
    
    SEVERITY_LEVELS = [
        'mild', 'moderate', 'severe',
        'at risk', 'not at risk',
        'null', 'not applicable', 'none', 'unknown', 'indeterminate'
    ]
    
    OUTPUT_FIELDS = [
        'case_presentation',
        'clinical_symptoms_and_signs',
        'growth_and_anthropometrics',
        'physical_exam',
        'nutrition_and_intake',
        'labs_and_screening',
        'diagnosis_and_reasoning',
        'malnutrition_status',
        'care_plan',
        'social_context',
        'clinical_insights'
    ]
    
    FIELD_ORDERING = {
        'case_presentation': 1,
        'clinical_symptoms_and_signs': 1,
        'growth_and_anthropometrics': 1,
        'physical_exam': 1,
        'nutrition_and_intake': 1,
        'labs_and_screening': 1,
        'diagnosis_and_reasoning': 2,
        'malnutrition_status': 3,
        'care_plan': 4,
        'social_context': 5,
        'clinical_insights': 5
    }
    
    @classmethod
    def create_task_config(cls) -> TaskConfig:
        """Create comprehensive malnutrition assessment task configuration."""
        return TaskConfig(
            task_name="pediatric_malnutrition_temporal_reasoning_v1_1",
            task_description=(
                "Pediatric malnutrition assessment with temporal reasoning and predictive analysis. "
                "Covers 11 comprehensive domains: case presentation, clinical symptoms and signs with temporal "
                "capture, anthropometric trends and trajectories, physical exam progression, nutritional intake "
                "patterns, laboratory trends and recommendations, diagnosis with clinical reasoning (BEFORE status), "
                "malnutrition status classification (AFTER reasoning), comprehensive care planning with monitoring "
                "schedules and escalation criteria, social context and barriers, and clinical insights with prognosis. "
                "Uses evidence-based guidelines (ASPEN, WHO, AND). "
                "Emphasizes TEMPORAL PATTERNS, PREDICTIVE REASONING, and ACTION-ORIENTED PLANNING. "
                "Teaches models to REASON and PREDICT, not just extract. "
                "Questions focus on 'Why?', 'How did you decide?', 'What should we expect?', 'What's the plan?'"
            ),
            input_field="txt",
            output_fields=cls.OUTPUT_FIELDS,
            question_templates=cls.QUESTION_TEMPLATES,
            field_ordering=cls.FIELD_ORDERING,
            output_formats=[
                OutputFormat.TEXT,
                OutputFormat.JSON,
                OutputFormat.XML,
                OutputFormat.MARKDOWN
            ],
            output_format_ratios={
                "text": 0.40,
                "json": 0.35,
                "xml": 0.10,
                "markdown": 0.15
            },
            medical_terminology=cls.MEDICAL_TERMINOLOGY,
            diagnostic_codes=cls.DIAGNOSTIC_CODES,
            severity_levels=cls.SEVERITY_LEVELS,
            default_system_prompt=(
                "You are an expert pediatric nutritionist and clinician specializing in malnutrition assessment. "
                "You excel at temporal reasoning, analyzing trends over time, and making predictions about clinical "
                "trajectories. You provide comprehensive assessments that integrate all available data temporally, "
                "reason through incomplete information, and develop actionable plans with specific monitoring schedules. "
                "You use evidence-based guidelines (ASPEN, WHO) and explain your clinical reasoning clearly. "
                "When data is incomplete, you recommend specific next steps with rationale. You think ahead about "
                "expected trajectories, monitoring schedules, and escalation criteria."
            )
        )


# [Helper functions]
def create_safety_config(
    enable_pii: bool = False,
    enable_bias: bool = False,
    enable_validation: bool = False,
    block_on_failure: bool = False
) -> SafetyConfig:
    """Create safety configuration (stub - safety features removed)."""
    return SafetyConfig()


def create_training_config(
    epochs: int = 2,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    max_steps: Optional[int] = None,
    quick_test: bool = False
) -> TrainingConfig:
    """Create training configuration."""
    return TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_ratio=0.05,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        save_strategy="epoch",
        save_steps=None,
        save_total_limit=2,
        evaluation_strategy="epoch",
        eval_steps=None,
        logging_steps=10,
        fp16=False,
        bf16=True,
        use_full_dataset=not quick_test,
        subsample_size=100 if quick_test else 10000
    )


def create_conversation_config(
    single_turn_ratio: float = 0.5,
    max_multi_turns: int = 12,
    validation_split: float = 0.0,  # Framework handles split internally
    logical_style_ratio: float = 0.5,
    include_typos: bool = True,
    typo_ratio: float = 0.15,
    max_seq_length: int = 16384
) -> ConversationConfig:
    """
    Create data multiplication configuration (v1.0.0).
    
    IMPORTANT: validation_split is handled internally by the framework.
    You only need ONE CSV file - the framework splits it automatically.
    """
    context_window_chars = max_seq_length * 4
    
    return ConversationConfig(
        single_turn_ratio=single_turn_ratio,
        max_multi_turns=max_multi_turns,
        include_typos=include_typos,
        typo_ratio=typo_ratio,
        validation_split=validation_split,  # Framework splits internally
        logical_style_ratio=logical_style_ratio,
        include_followup_questions=True,
        context_window_size=context_window_chars,
        response_allocation_ratio=0.25,
        buffer_ratio=0.10,
        min_question_length=1000,
        max_question_length=8000
    )


def validate_csv(csv_path: str) -> bool:
    """Validate CSV structure and required fields."""
    logger.info("=" * 80)
    logger.info(f"Validating CSV: {csv_path}")
    logger.info("=" * 80)
    
    df = pd.read_csv(csv_path)
    
    required_fields = [
        'txt',
        'input_label_value',
        'case_presentation',
        'clinical_symptoms_and_signs',
        'growth_and_anthropometrics',
        'physical_exam',
        'nutrition_and_intake',
        'diagnosis_and_reasoning',
        'labs_and_screening',
        'care_plan',
        'clinical_insights'
    ]
    
    optional_fields = ['social_context']
    
    missing_required = [col for col in required_fields if col not in df.columns]
    
    if missing_required:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Missing required columns: {missing_required}\n"
            f"Required: {required_fields}\n"
            f"Optional: {optional_fields}\n"
            f"Found: {available}"
        )
    
    missing_optional = [col for col in optional_fields if col not in df.columns]
    if missing_optional:
        logger.warning(f"Missing optional columns: {missing_optional}")
    
    unique_labels = df['input_label_value'].dropna().unique()
    invalid_labels = [l for l in unique_labels if l not in [0, 1, '0', '1']]
    if invalid_labels:
        raise ValueError(
            f"input_label_value must contain only 0 or 1. Found: {invalid_labels}"
        )
    
    label_counts = df['input_label_value'].value_counts()
    logger.info(f"Malnutrition status distribution:")
    logger.info(f"  Present (1): {label_counts.get(1, 0) + label_counts.get('1', 0)}")
    logger.info(f"  Absent (0): {label_counts.get(0, 0) + label_counts.get('0', 0)}")
    
    empty_notes = df['txt'].isna().sum()
    if empty_notes > 0:
        logger.warning(f"Found {empty_notes} empty clinical notes - will be filtered")
    
    short_notes = (df['txt'].str.len() < 100).sum()
    if short_notes > 0:
        logger.warning(f"Found {short_notes} very short notes (<100 chars)")
    
    logger.info(f"✓ CSV validation passed: {len(df)} rows")
    logger.info("=" * 80)
    
    return True


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess malnutrition assessment dataset."""
    logger.info("=" * 80)
    logger.info("Preprocessing data...")
    logger.info("=" * 80)
    
    original_count = len(df)
    
    df = df[df['txt'].notna()].copy()
    df = df[df['txt'].str.len() >= 100].reset_index(drop=True)
    
    logger.info(f"Filtered: {original_count} → {len(df)} rows")
    
    def convert_label_to_status(label):
        """Convert 0/1 label to varied natural language responses."""
        label = str(label).strip()
        
        if label in ['1', '1.0']:
            responses = [
                "Yes, malnutrition is present",
                "Malnutrition present",
                "Yes - this patient is malnourished",
                "Malnutrition is present based on assessment",
                "Yes, meets criteria for malnutrition",
                "Present",
                "Malnutrition confirmed"
            ]
        elif label in ['0', '0.0']:
            responses = [
                "No, malnutrition is absent",
                "Malnutrition absent",
                "No - this patient is not malnourished",
                "Malnutrition is not present",
                "No, does not meet criteria for malnutrition",
                "Absent",
                "No evidence of malnutrition"
            ]
        else:
            return "Unknown"
        
        import random
        return random.choice(responses)
    
    df['malnutrition_status'] = df['input_label_value'].apply(convert_label_to_status)
    
    logger.info(f"Created 'malnutrition_status' field from 'input_label_value'")
    
    if 'social_context' not in df.columns:
        df['social_context'] = "Social context not fully documented."
        logger.info("Added default 'social_context' field")
    else:
        df['social_context'] = df['social_context'].fillna(
            "Social context not fully documented."
        )
    
    text_fields = [
        'txt', 'malnutrition_status', 'case_presentation', 'clinical_symptoms_and_signs',
        'growth_and_anthropometrics', 'physical_exam', 'nutrition_and_intake',
        'diagnosis_and_reasoning', 'labs_and_screening', 'care_plan',
        'social_context', 'clinical_insights'
    ]
    
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip()
            df[field] = df[field].str.replace(r'\s+', ' ', regex=True)
    
    avg_note_length = df['txt'].str.len().mean()
    min_note_length = df['txt'].str.len().min()
    max_note_length = df['txt'].str.len().max()
    
    logger.info(f"Clinical note statistics:")
    logger.info(f"  Average: {avg_note_length:.0f} chars (~{avg_note_length/4:.0f} tokens)")
    logger.info(f"  Min: {min_note_length} chars (~{min_note_length/4:.0f} tokens)")
    logger.info(f"  Max: {max_note_length} chars (~{max_note_length/4:.0f} tokens)")
    logger.info("=" * 80)
    
    return df


def main():
    """Main training pipeline for malnutrition assessment."""
    parser = argparse.ArgumentParser(
        description="Train Pediatric Malnutrition Assessment - v1.0.0 with Temporal Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: Only ONE CSV file is required!
  The framework automatically splits it into train/validation based on --validation_split

Examples:
  # Use ALL data for training (no validation split)
  python train_malnutrition_v1_1.py --csv data.csv --validation_split 0.0
  
  # Automatic 80/20 split (framework does this internally)
  python train_malnutrition_v1_1.py --csv data.csv --validation_split 0.2
  
  # Automatic 70/30 split
  python train_malnutrition_v1_1.py --csv data.csv --validation_split 0.3
  
  # Train with 8K context and 20% validation
  python train_malnutrition_v1_1.py --csv data.csv --max_seq_length 8192 --validation_split 0.2
  
  # More reasoning styles (60% logical)
  python train_malnutrition_v1_1.py --csv data.csv --logical_style_ratio 0.6
  
  # Quick test (100 examples, no split)
  python train_malnutrition_v1_1.py --csv data.csv --quick_test

VALIDATION SPLIT EXPLAINED:
  --validation_split 0.0  → 100% train, 0% validation (use ALL data)
  --validation_split 0.2  → 80% train, 20% validation (framework splits automatically)
  --validation_split 0.3  → 70% train, 30% validation

  No need for separate train.csv and val.csv!
  The framework handles splitting internally based on the ratio you specify.

NEW in v1.0.0:
  ✓ 11 fields (NEW: clinical_symptoms_and_signs)
  ✓ 10 questions per field (110 total)
  ✓ Temporal reasoning: "How has it changed?"
  ✓ Predictive: "What should we expect?"
  ✓ Action: "What should we order?"
  ✓ Reasoning: "Why?", "How did you decide?"
  ✓ Field ordering: Reasoning BEFORE status
  ✓ 16 question styles (9 logical reasoning)
        """
    )
    
    # Required arguments
    parser.add_argument("--csv", required=True,
                       help="Path to CSV (ONE file - framework splits internally)")
    
    # Output and model
    parser.add_argument("--output", default="./malnutrition_models_v1_1",
                       help="Output directory (default: ./malnutrition_models_v1_1)")
    parser.add_argument("--model", default="llama",
                       choices=["llama", "phi-4", "mistral", "qwen"],
                       help="Model type (default: llama)")
    parser.add_argument("--models",
                       help="Train multiple models: llama,phi-4,mistral")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1,
                       help="Training epochs (default: 1)")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size (default: 2)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate (default: 2e-4)")
    parser.add_argument("--max_steps", type=int,
                       help="Max training steps (overrides epochs)")
    
    # Data configuration (v1.0.0)
    parser.add_argument("--max_seq_length", type=int, default=16384,
                       help="Max sequence length in TOKENS (default: 16384)")
    parser.add_argument("--validation_split", type=float, default=0.0,
                       help="Validation split ratio: 0.0=no split (default), 0.2=80/20, 0.3=70/30")
    parser.add_argument("--single_turn_ratio", type=float, default=0.5,
                       help="Single-turn conversation ratio (default: 0.5)")
    parser.add_argument("--logical_style_ratio", type=float, default=0.5,
                       help="Logical reasoning style ratio (default: 0.5)")
    parser.add_argument("--max_multi_turns", type=int, default=15,
                       help="Max conversation turns (default: 12)")
    
    # Testing options
    parser.add_argument("--quick_test", action="store_true",
                       help="Quick test (100 examples)")
    parser.add_argument("--disable_typos", action="store_true",
                       help="Disable typo injection")
    
    # System
    parser.add_argument("--cuda_device", type=int, default=0,
                       help="CUDA device (default: 0)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        return 1
    
    try:
        validate_csv(args.csv)
    except ValueError as e:
        logger.error(f"CSV validation failed: {e}")
        return 1
    
    # Determine models
    if args.models:
        model_types = [m.strip() for m in args.models.split(",")]
    else:
        model_types = [args.model]
    
    # Print header
    logger.info("")
    logger.info("=" * 80)
    logger.info("MEDDIALOGUE v1.0.0 - MALNUTRITION TEMPORAL REASONING")
    logger.info("=" * 80)
    logger.info(f"Input: {args.csv} (ONE file - framework splits internally)")
    logger.info(f"Output: {args.output}")
    logger.info(f"Models: {', '.join(model_types)}")
    logger.info("-" * 80)
    logger.info("Configuration:")
    logger.info(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}")
    logger.info(f"  Max sequence: {args.max_seq_length} tokens (~{args.max_seq_length * 4:,} chars)")
    logger.info("-" * 80)
    logger.info("Data Split (Framework handles this internally):")
    if args.validation_split == 0.0:
        logger.info(f"  Validation split: {args.validation_split * 100:.0f}% (NO SPLIT - ALL data for training)")
    else:
        logger.info(f"  Validation split: {args.validation_split * 100:.0f}% ({(1-args.validation_split)*100:.0f}% train, {args.validation_split*100:.0f}% val)")
    logger.info(f"  Single-turn: {args.single_turn_ratio * 100:.0f}%")
    logger.info(f"  Logical styles: {args.logical_style_ratio * 100:.0f}%")
    logger.info(f"  Max turns: {args.max_multi_turns + 1}")
    logger.info("-" * 80)
    logger.info("NEW FEATURES v1.0.0:")
    logger.info("  ✓ 11 fields (NEW: clinical_symptoms_and_signs)")
    logger.info("  ✓ 10 questions per field (110 total)")
    logger.info("  ✓ Temporal reasoning questions")
    logger.info("  ✓ Predictive questions")
    logger.info("  ✓ Action-oriented questions")
    logger.info("  ✓ Clinical reasoning focus")
    logger.info("  ✓ Field ordering: Reasoning BEFORE status")
    logger.info("  ✓ 16 question styles (9 logical)")
    logger.info("=" * 80)
    logger.info("")
    
    # Load data
    logger.info("Loading and preprocessing data...")
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} rows from ONE CSV file")
    df = preprocess_data(df)
    logger.info("")
    
    # Create configurations
    logger.info("Creating configurations...")
    
    task_config = MalnutritionTaskConfig.create_task_config()
    logger.info(f"✓ Task config: {len(task_config.output_fields)} fields")
    logger.info(f"  NEW field: clinical_symptoms_and_signs")
    logger.info(f"  Field ordering: Reasoning (priority 2) before Status (priority 3)")
    logger.info(f"  Question templates: 110 total (10 per field)")
    
    safety_config = create_safety_config()
    logger.info(f"✓ Safety config created (stub)")
    
    training_config = create_training_config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        quick_test=args.quick_test
    )
    logger.info(f"✓ Training config created")
    
    conversation_config = create_conversation_config(
        single_turn_ratio=args.single_turn_ratio,
        max_multi_turns=args.max_multi_turns,
        validation_split=args.validation_split,  # Framework handles split
        logical_style_ratio=args.logical_style_ratio,
        include_typos=not args.disable_typos,
        typo_ratio=0.15 if not args.disable_typos else 0.0,
        max_seq_length=args.max_seq_length
    )
    logger.info(f"✓ Data config created")
    logger.info(f"  Framework will split data internally: {(1-args.validation_split)*100:.0f}% train, {args.validation_split*100:.0f}% val")
    logger.info("")
    
    # Train models
    successful = []
    failed = []
    
    for i, model_type in enumerate(model_types):
        logger.info("=" * 80)
        logger.info(f"Training {model_type} ({i+1}/{len(model_types)})")
        logger.info("=" * 80)
        
        try:
            meddialogue = MedDialogue(
                task_config=task_config,
                model_type=model_type,
                safety_config=safety_config,
                training_config=training_config,
                conversation_config=conversation_config,
                output_dir=os.path.join(args.output, model_type),
                cuda_device=args.cuda_device,
                verbose=True
            )
            
            # Pass ONE dataframe - framework handles split
            results = meddialogue.train(
                data=df,  # ONE dataframe!
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_steps=args.max_steps
            )
            
            logger.info("=" * 80)
            logger.info(f"✓ {model_type} completed!")
            logger.info("=" * 80)
            logger.info(f"Train: {results['num_train_examples']} examples")
            logger.info(f"Has validation: {results['has_validation']}")
            logger.info(f"Time: {results['training_time_minutes']:.2f} min")
            logger.info(f"Loss: {results['training_results'].get('train_loss', 'N/A')}")
            
            if 'save_paths' in results:
                for save_type, path in results['save_paths'].items():
                    logger.info(f"  {save_type}: {path}")
            
            logger.info("=" * 80)
            successful.append(model_type)
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"✗ {model_type} FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}")
            
            import traceback
            logger.error(traceback.format_exc())
            
            failed.append((model_type, str(e)))
        
        logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Successful: {len(successful)}/{len(model_types)}")
    logger.info(f"Failed: {len(failed)}/{len(model_types)}")
    
    if successful:
        logger.info("\n✓ Successful:")
        for m in successful:
            logger.info(f"  - {m}: {os.path.join(args.output, m)}")
    
    if failed:
        logger.info("\n✗ Failed:")
        for m, error in failed:
            logger.info(f"  - {m}: {error}")
    
    logger.info("=" * 80)
    
    if successful:
        logger.info("\n✓ TRAINING COMPLETED - v1.0.0 TEMPORAL REASONING")
        logger.info(f"Output: {args.output}")
        logger.info(f"Input: ONE CSV file (framework handled split internally)")
        logger.info("\nYour model can now:")
        logger.info("  • Reason temporally about changes over time")
        logger.info("  • Predict expected outcomes and trajectories")
        logger.info("  • Recommend specific actions and labs")
        logger.info("  • Plan monitoring schedules with escalation")
        logger.info("  • Explain clinical reasoning and decisions")
        logger.info("  • Classify using clinical logic (Reason → Status)")
        return 0
    else:
        logger.error("\n✗ ALL MODELS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())