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
  QUESTION_TEMPLATES = {
        'case_presentation': [
            # 1. Assessment framing with temporal context
            "Document the clinical presentation including chief concern, temporal context, and assessment type (single-point vs serial/longitudinal). What additional history would strengthen the assessment?",
            
            # 2. Problem identification with family perspective
            "Summarize the presenting problem with timeline of events. What is the family's primary concern and how has it evolved over time?",
            
            # 3. Clinical context and setting
            "Describe the setting and circumstances of this assessment. What factors (acute illness, chronic condition, social determinants) are relevant to nutritional status?",
            
            # 4. Data completeness evaluation
            "What is documented in the presentation, and what critical information is missing? Prioritize what should be obtained and provide clinical rationale.",
            
            # 5. Pattern recognition and risk factors
            "Based on the presentation, what nutritional risk factors are evident? What is your initial clinical impression and differential diagnosis?",
            
            # 6. Temporal progression analysis
            "How has the clinical situation evolved? Describe the timeline from onset to current presentation with key events and changes.",
            
            # 7. Family understanding and barriers
            "What is the family's understanding of the problem? Are there communication, cultural, or health literacy factors that need to be addressed?",
            
            # 8. Previous interventions and response
            "What interventions have been attempted prior to this assessment? Document previous treatments, adherence, and response.",
            
            # 9. Acute vs chronic determination
            "Based on the presentation, is this an acute issue, chronic problem, or acute-on-chronic? Justify your assessment with temporal evidence.",
            
            # 10. Assessment type implications
            "Is this a single-point assessment or do serial measurements exist? How does the assessment type impact diagnostic confidence and available criteria?"
        ],
        
        'clinical_symptoms_and_signs': [
            # 1. Comprehensive symptom documentation with dates
            "Document all clinical symptoms with onset dates and progression: GI (vomiting, diarrhea, pain), systemic (fever, fatigue), feeding difficulties, and functional impairments. Describe the temporal trajectory.",
            
            # 2. Symptom-nutrition impact quantification
            "How do documented symptoms impact nutritional intake and status? Quantify the effect (e.g., 'vomiting limits intake to <50%') and specify duration.",
            
            # 3. Symptom trajectory characterization
            "What is the symptom trajectory: new onset, progressive, stable, improving, or resolved? For serial assessments, describe changes across encounters with dates.",
            
            # 4. GI symptom pattern analysis
            "Detail gastrointestinal symptoms (emesis frequency/volume, stool pattern, abdominal pain, reflux). How do these affect intake and absorption?",
            
            # 5. Systemic symptom documentation
            "Document systemic symptoms affecting nutrition: fever pattern, fatigue level, irritability, activity tolerance. How do these impact feeding behavior?",
            
            # 6. Clinical significance prioritization
            "Which symptoms are most clinically significant for nutritional assessment? How do they inform etiology (illness-related vs non-illness-related malnutrition)?",
            
            # 7. Symptom-diagnosis correlation
            "How do symptom patterns correlate with anthropometric findings and laboratory data? Does symptom onset explain observed nutritional decline?",
            
            # 8. Symptom monitoring recommendations
            "What symptom monitoring should be implemented? Specify frequency (daily diary, weekly assessment) and red flags requiring immediate evaluation.",
            
            # 9. Symptom management for intake optimization
            "What symptom management interventions would improve nutritional intake? Recommend specific medications (antiemetics, appetite stimulants) or feeding modifications.",
            
            # 10. Systematic symptom assessment gaps
            "If symptom documentation is incomplete, what specific symptoms should be systematically assessed? Provide clinical rationale using evidence-based screening tools."
        ],
        
        'growth_and_anthropometrics': [
            # 1. Complete measurement extraction with z-scores
            "Extract all anthropometric measurements with dates: weight, height/length, BMI, MUAC, head circumference. Calculate z-scores and percentiles using appropriate growth references (WHO 0-2y, CDC 2-20y).",
            
            # 2. Temporal trend analysis with quantification
            "For serial/longitudinal data: Calculate growth trends including absolute change, percentage change, velocity, and z-score trajectory. Quantify rate of change (e.g., 'BMI z-score declined -1.8 over 60 days, rate -0.9 per month'). For single-point: State limitation and recommend serial interval.",
            
            # 3. ASPEN anthropometric criteria application
            "Apply ASPEN anthropometric criteria: Specify severity (mild z-score -1 to -1.9, moderate -2 to -2.9, severe ≤-3) with exact measured values and dates. Document which specific criterion is met.",
            
            # 4. Growth velocity assessment (ASPEN indicator 2)
            "For serial/longitudinal assessments: Evaluate growth velocity criterion. Has z-score declined ≥1 (mild), ≥2 (moderate), or ≥3 (severe)? Specify measurements, dates, and rate. For single-point: State velocity cannot be assessed and recommend monitoring plan.",
            
            # 5. Growth pattern interpretation in context
            "Interpret growth pattern clinically: Is growth appropriate for age, sex, and underlying condition? Does pattern suggest acute vs chronic malnutrition? How do findings correlate with intake, symptoms, and illness severity?",
            
            # 6. Percentile trajectory analysis
            "Describe percentile tracking over time: Is the patient maintaining, crossing down, or crossing up percentile curves? Quantify percentile change (e.g., '50th to 10th percentile over 3 months').",
            
            # 7. WHO malnutrition classification
            "Apply WHO weight-for-height or BMI-for-age classification: Severe acute malnutrition (z<-3), moderate acute malnutrition (z -2 to -3), at risk (z -1 to -2), or normal. If height-for-age z<-2, document stunting.",
            
            # 8. Predictive trajectory modeling
            "Based on current growth trajectory, what is the projected anthropometric outcome if current pattern continues? What would expected z-score be in 1 month, 3 months?",
            
            # 9. Anthropometric monitoring schedule
            "Recommend anthropometric monitoring schedule based on severity: Specify frequency for weight (daily, 2-3x/week, weekly, biweekly), length/height (monthly, every 2-3 months), and when to remeasure MUAC, head circumference.",
            
            # 10. Missing measurement prioritization
            "What anthropometric data are missing? Prioritize which measurements should be obtained with clinical rationale (e.g., 'MUAC recommended for edematous malnutrition when BMI may be falsely elevated')."
        ],
        
        'physical_exam': [
            # 1. Nutrition-focused physical exam documentation
            "Document nutrition-focused physical exam findings systematically: muscle mass (temporal, clavicle, shoulder, ribs, extremities), subcutaneous fat (orbital, triceps, subscapular), edema (location, severity), micronutrient deficiency signs (skin, hair, nails, oral cavity). Grade severity.",
            
            # 2. ASPEN physical findings criterion application
            "Do physical findings meet ASPEN indicator 4 for malnutrition diagnosis? Document presence of muscle wasting AND/OR subcutaneous fat loss. Specify anatomical locations and severity grading.",
            
            # 3. Serial physical exam comparison
            "For serial assessments: Compare current exam to prior encounters with dates. Describe trajectory (worsening, stable, improving). Quantify change if possible (e.g., 'muscle wasting progressed from mild to moderate').",
            
            # 4. Single-point exam with data correlation
            "For single-point assessment: Correlate physical findings with anthropometric data and laboratory values to strengthen assessment. Do findings align with low BMI z-score and hypoalbuminemia?",
            
            # 5. Edema assessment and interpretation
            "Assess for edema: location (periorbital, pedal, sacral, anasarca), severity (1-4+), and etiology. Is this nutritional edema (kwashiorkor), fluid overload, or other cause? How does this affect weight interpretation?",
            
            # 6. Micronutrient deficiency signs
            "Document signs of specific micronutrient deficiencies: Vitamin A (xerosis, Bitot spots), Vitamin D (rachitic rosary, bowed legs), Iron (pallor, koilonychia), Zinc (perioral dermatitis, alopecia), B vitamins (glossitis, cheilosis). Recommend targeted repletion.",
            
            # 7. Functional assessment
            "Assess functional status: activity level, muscle strength (handgrip if age-appropriate), mobility, endurance. How do functional limitations correlate with anthropometric and biochemical findings?",
            
            # 8. Physical exam-diagnosis correlation
            "How do physical exam findings align with anthropometric measurements and clinical history? Explain discrepancies (e.g., 'low BMI with preserved fat suggests recent acute loss rather than chronic malnutrition').",
            
            # 9. Subjective Global Assessment integration
            "If SGA performed: Document overall rating (well-nourished, mild-moderately malnourished, severely malnourished) and key findings supporting the rating.",
            
            # 10. Physical exam completion recommendations
            "If nutrition-focused physical exam is incomplete, specify which components should be assessed: detailed muscle/fat assessment, functional testing, micronutrient deficiency screening. Provide clinical rationale for each."
        ],
        
        'nutrition_and_intake': [
            # 1. Quantitative intake documentation with temporal detail
            "Quantify nutritional intake with temporal detail: current intake as percentage of estimated needs, route (oral/enteral/parenteral), volume/calories/protein per day. Document duration of inadequate intake with specific dates and trends.",
            
            # 2. ASPEN inadequate intake criterion application
            "Apply ASPEN indicator 3: Is intake <50% of estimated needs for ≥1 week? Specify exact percentage, duration (e.g., '30-40% for 14 days'), and dates. Calculate daily energy and protein deficits.",
            
            # 3. Intake trajectory analysis over time
            "Describe intake pattern temporally: 'Intake declined from 80-90% estimated needs in December to 40-50% by February'. Identify inflection points and correlate with clinical events or interventions.",
            
            # 4. Estimated needs calculation and justification
            "Calculate estimated energy and protein needs using appropriate method (DRI, WHO, disease-specific equations). Justify chosen method based on patient age, condition, and activity. Show calculation.",
            
            # 5. Route and adequacy of nutrition support
            "Document current nutrition support: Formula type/concentration, feeding schedule (continuous/bolus), oral supplements used. Is current regimen meeting calculated needs? If not, quantify deficit.",
            
            # 6. Intake barriers identification
            "What factors are limiting adequate intake? Categories: mechanical (dysphagia, reflux), behavioral (food refusal, sensory aversion), medical (nausea, early satiety), environmental (food insecurity, feeding skills). Specify duration.",
            
            # 7. Intake-anthropometric correlation
            "How does intake pattern correlate with anthropometric trajectory? Does inadequate intake explain observed weight loss (g/day deficit × duration = cumulative deficit)? Is timing consistent?",
            
            # 8. Diet history and patterns
            "Document dietary history: typical meal pattern, food preferences/aversions, feeding development, bottle/breast history, introduction of solids. What cultural or religious dietary practices are relevant?",
            
            # 9. Intake optimization interventions
            "If intake is inadequate: Recommend specific interventions with targets (e.g., 'High-calorie formula fortification to 24 kcal/oz to achieve 120% DRI, protein 3g/kg/day'). Include feeding strategy modifications.",
            
            # 10. Intake monitoring and quantification plan
            "If intake assessment is incomplete, specify methods to quantify: 3-day food record, 24-hour recall, calorie count (how many days). Recommend intake monitoring frequency and documentation method."
        ],
        
        'labs_and_screening': [
            # 1. Comprehensive laboratory extraction with dates
            "Extract all nutrition-relevant laboratory values with dates: visceral proteins (albumin, prealbumin, total protein), CMP, CBC, micronutrients (25-OH vitamin D, iron studies, zinc, folate, B12), inflammatory markers (CRP, ESR). Document reference ranges and units.",
            
            # 2. Laboratory temporal trends with quantification
            "For serial data: Calculate trends for each parameter (e.g., 'Albumin declined from 3.8 g/dL on 1/15 to 3.2 g/dL on 3/15, 16% decrease over 59 days, rate -0.3 g/dL per month'). Graph significant trends mentally.",
            
            # 3. Inflammation-adjusted interpretation
            "Interpret laboratory values in context of inflammatory state. If CRP elevated (>5 mg/L): Recognize visceral proteins reflect inflammation not nutritional status. Prioritize functional measures (prealbumin, RBP) and anthropometrics.",
            
            # 4. Nutritional anemia assessment
            "Evaluate for nutritional anemia: Classify as iron deficiency (low ferritin, high TIBC), folate deficiency (macrocytic), B12 deficiency (macrocytic with neuro symptoms), or anemia of chronic disease (low iron, low TIBC, high ferritin).",
            
            # 5. Refeeding syndrome risk assessment
            "Assess refeeding risk using laboratory and clinical criteria: Electrolytes (K, Mg, Phos), thiamine status if available, severity/duration of malnutrition. Is patient at high risk requiring intensive electrolyte monitoring?",
            
            # 6. Micronutrient deficiency identification
            "Identify specific micronutrient deficiencies: Vitamin D <20 ng/mL (insufficiency) or <12 (deficiency), zinc <60 mcg/dL, iron deficiency (ferritin <15 ng/mL), others. Recommend repletion doses and duration.",
            
            # 7. Single-point lab-anthropometric correlation
            "For single-point assessment: Use laboratory data to corroborate anthropometric findings. Do low albumin, anemia, and vitamin deficiencies support diagnosis of malnutrition based on low BMI z-score?",
            
            # 8. Baseline laboratory panel recommendations
            "For confirmed or suspected malnutrition, recommend baseline laboratory panel: CMP with phosphorus and magnesium (refeeding monitoring), CBC with differential, albumin and prealbumin (trend monitoring), CRP (inflammation), vitamin D, iron studies, zinc. Justify each test.",
            
            # 9. Serial laboratory monitoring schedule
            "Establish laboratory monitoring schedule based on severity and intervention: Severe malnutrition with refeeding risk - daily electrolytes/phos/mag days 1-7, then biweekly. Moderate - weekly CMP for 2 weeks, then biweekly. Mild - baseline, then 2-4 weeks. Specify duration.",
            
            # 10. Screening tool documentation and interpretation
            "If malnutrition screening performed (PYMS, STRONG-kids, STAMP): Document score, risk category, and recommended action. Does screening result align with comprehensive assessment findings?"
        ],
        
        'diagnosis_and_reasoning': [
            # 1. Structured nutrition diagnosis (PES format)
            "Formulate nutrition diagnosis using PES format: Problem (malnutrition status, severity, chronicity), Etiology (illness-related/non-illness-related with specific cause), Signs/Symptoms (anthropometric, biochemical, clinical, dietary data supporting diagnosis).",
            
            # 2. ASPEN diagnostic criteria systematic application
            "Apply ASPEN criteria systematically: (1) State assessment type (single-point requires ≥1 indicator; serial/longitudinal requires ≥2). (2) Document 4 indicators: Indicator 1-Anthropometric deficit (z-score with date), Indicator 2-Growth velocity (if serial data), Indicator 3-Inadequate intake (<50% for ≥1 week), Indicator 4-Physical findings (muscle/fat loss). (3) Count indicators met (X/4) and confirm diagnostic threshold satisfied.",
            
            # 3. WHO severity classification application
            "If malnutrition present: Apply WHO classification using z-scores: Severe acute malnutrition (WHZ or BAZ <-3, <1st percentile), moderate acute malnutrition (z -2 to -3, 2nd-3rd percentile), or at risk (z -1 to -2). If stunting (HAZ <-2): document chronic malnutrition.",
            
            # 4. Temporal evidence synthesis
            "Synthesize all evidence temporally: How do anthropometric trends, intake patterns, physical exam progression, laboratory trajectories, and symptom evolution converge to support your diagnosis? Address timeline and causality.",
            
            # 5. Severity determination with specific criteria
            "Determine and justify severity (mild/moderate/severe): Specify which indicator(s) drive severity classification. Example: 'Moderate severity based on BMI z-score -2.3 (ASPEN moderate -2 to -2.9) and growth velocity decline of 1.8 z-scores (ASPEN moderate ≥2)'.",
            
            # 6. Etiology classification and implications
            "Classify etiology: Illness-related (secondary to disease with inflammatory response documented by CRP >5) or non-illness-related (primarily intake-related without inflammation). How does etiology inform prognosis and intervention intensity?",
            
            # 7. Acute vs chronic malnutrition determination
            "Classify chronicity: Acute malnutrition (recent weight loss, low WHZ, normal HAZ), chronic (stunting, low HAZ), or acute-on-chronic (low WHZ and HAZ). Justify using temporal data and growth pattern.",
            
            # 8. Convergent evidence analysis
            "Analyze convergent vs divergent evidence: What data points strongly support the diagnosis? Are there any conflicting findings that need explanation (e.g., 'Normal albumin despite low BMI explained by acute weight loss without time for visceral protein depletion')?",
            
            # 9. Diagnostic confidence and limitations
            "Assess diagnostic confidence: For single-point assessments, note limitations in evaluating growth velocity (ASPEN indicator 2). What is confidence level (definite/probable/possible) based on data completeness?",
            
            # 10. Recommended assessments to strengthen diagnosis
            "To strengthen diagnostic confidence, recommend: Serial anthropometric measurements (interval), quantified intake assessment (method), complete NFPE, baseline laboratory panel, malnutrition screening tool. Prioritize based on current data gaps."
        ],
        
        'malnutrition_status': [
            # 1. Clear binary classification
            "State the malnutrition classification clearly: 'Malnutrition present' or 'Malnutrition absent'.",
            
            # 2. Classification with severity if present
            "Is malnutrition present? If yes, specify: severity (mild/moderate/severe), chronicity (acute/chronic/acute-on-chronic), etiology (illness-related/non-illness-related).",
            
            # 3. ASPEN guideline-based determination
            "Does this patient meet ASPEN diagnostic criteria for pediatric malnutrition? State yes/no with supporting evidence (number of indicators met and threshold required).",
            
            # 4. WHO classification if applicable
            "Per WHO classification: Does patient have severe acute malnutrition (SAM), moderate acute malnutrition (MAM), at risk, or normal nutritional status? Specify z-score used.",
            
            # 5. Confidence-weighted classification
            "What is your final diagnostic impression with confidence level? Format: 'Malnutrition present - moderate severity, illness-related, acute presentation. Confidence: Definite (all 4 ASPEN indicators documented)'.",
            
            # 6. Clinical judgment synthesis
            "Based on synthesis of all available evidence (anthropometrics, physical exam, intake, labs, symptoms), what is your clinical judgment regarding malnutrition status?",
            
            # 7. Comparison to ground truth (if training)
            "Does your assessment align with the documented diagnosis? If discrepancy exists, explain what additional information would resolve it.",
            
            # 8. Status with diagnostic criteria citation
            "Malnutrition status with criteria: 'Present - meets ASPEN criteria with 3/4 indicators (anthropometric z-score -2.4, intake <50% for 10 days, muscle wasting on exam). Severity moderate per z-score -2 to -3 range'.",
            
            # 9. Prognostic classification
            "Classify malnutrition with prognostic implications: Is this likely to respond to nutrition support alone, or will it require treatment of underlying condition? Justify based on etiology.",
            
            # 10. Monitoring classification
            "What is the malnutrition status for documentation and monitoring purposes? Use ICD-10 terminology if applicable (e.g., 'Moderate protein-energy malnutrition, E44.0')."
        ],
        
        'care_plan': [
            # 1. Comprehensive intervention plan with goals
            "Develop comprehensive nutrition care plan: (1) Goals with timeline (anthropometric targets, intake goals, symptom resolution). (2) Interventions with specific doses/formulations (oral supplements, formula fortification, enteral/parenteral nutrition). (3) Etiology-directed treatments. (4) Education/counseling components.",
            
            # 2. Energy and protein prescription with rationale
            "Calculate and prescribe energy and protein targets: Specify kcal/kg/day and protein g/kg/day (e.g., '120% DRI = 110 kcal/kg/day, protein 3 g/kg/day for catch-up growth'). Justify prescription based on severity and goals. Specify route(s) to achieve target.",
            
            # 3. Specific formula/supplement recommendations
            "Recommend specific nutrition products: Formula type (standard, high-calorie, elemental, disease-specific), concentration (kcal/oz), volume schedule. If oral supplements: product name, flavor options, volume, frequency. Include second-line options if first-line fails.",
            
            # 4. Anthropometric monitoring schedule by severity
            "Establish anthropometric monitoring: Severe malnutrition - weight 2-3x/week, length/height monthly, calculate BMI weekly, MUAC biweekly. Moderate - weight weekly, length/height monthly. Mild - weight biweekly, length/height every 2-3 months. Specify measurement technique.",
            
            # 5. Laboratory monitoring protocol with refeeding consideration
            "Laboratory monitoring protocol: (1) Baseline panel if not done (CMP with phos/mag, CBC, albumin, CRP, vitamin D, iron studies). (2) Refeeding risk: daily lytes/phos/mag days 1-7, then biweekly weeks 2-4. (3) Standard monitoring: weekly CMP weeks 1-4, then biweekly with albumin/prealbumin. (4) Micronutrient reassessment month 3.",
            
            # 6. Clinical follow-up schedule with decision points
            "Specify follow-up schedule: Week 1 (day 5-7): Assess weight response, tolerance, intake, adjust if needed. Week 2: Progress check, modify plan if inadequate response. Weeks 3-4: Weekly visits. Weeks 4-8: Biweekly. Months 2-6: Monthly until goal reached. Define what constitutes adequate response.",
            
            # 7. Escalation criteria with specific timepoints
            "Define escalation plan with triggers: (1) Day 7: If weight gain <50g despite 100% intake → increase caloric density 10-20%. (2) Week 2: If no weight gain or continued loss → consider enteral nutrition, GI evaluation. (3) Week 4: If <50% expected progress → subspecialty referral (GI/endocrine/genetics). (4) Anytime: Worsening clinical status, intolerance, new symptoms → urgent reassessment.",
            
            # 8. Expected recovery trajectory with milestones
            "Predict recovery trajectory: (1) Week 1-2: Expected weight gain 20-30 g/day (formula-fed infant) or age-appropriate rate, improved energy/activity. (2) Month 1: Z-score improvement 0.5-1.0, intake consistently >80% needs. (3) Month 3: Approach baseline percentile, normalize labs, resolve physical findings. (4) Month 6: Sustained catch-up growth complete.",
            
            # 9. Complication monitoring and prevention
            "Specify complications to monitor: (1) Refeeding syndrome: electrolyte abnormalities days 1-7, cardiac symptoms. (2) Feeding intolerance: vomiting, diarrhea, abdominal distension. (3) Inadequate response: continued weight loss/faltering. (4) Overfeeding: excessive weight gain velocity, abnormal fat deposition. Define red flags.",
            
            # 10. Interdisciplinary coordination and referrals
            "Coordinate interdisciplinary care: (1) Dietitian: weekly initially for intake monitoring/plan adjustment. (2) GI: if underlying GI disease, feeding intolerance. (3) Social work: food security, formula assistance, care coordination. (4) Feeding therapy: if oral aversion/developmental feeding disorder. (5) Specify communication plan and team meetings."
        ],
        
        'social_context': [
            # 1. Social determinants systematic documentation
            "Document social determinants of health: Food security status (USDA screening), WIC/SNAP enrollment, housing stability, primary caregiver (single/dual parent, other), health literacy level, primary language, access to healthcare, financial barriers (insurance, copays, formula cost). Note temporal changes.",
            
            # 2. Food security and resource assessment
            "Assess food security: Administer validated screening (USDA 6-item or 2-item). If food insecure: Severity (low vs very low), duration, specific barriers (income, transportation, storage). Currently enrolled in assistance programs (WIC, SNAP, food pantry)?",
            
            # 3. Implementation barriers identification
            "Identify barriers to implementing nutrition care plan: (1) Financial: Can family afford specialized formula/supplements ($300-500/month)? Insurance coverage? (2) Access: Transportation to appointments, pharmacy access. (3) Practical: Refrigeration for formula, clean water supply. (4) Cultural/religious: Dietary restrictions affecting recommendations.",
            
            # 4. Caregiver factors affecting compliance
            "Assess caregiver factors: Health literacy (understanding of plan, demonstration of formula preparation), mental health (depression screening if indicated), physical health (ability to care for child), competing priorities (work schedule, other children), support system availability.",
            
            # 5. Cultural and linguistic considerations
            "Document cultural/linguistic factors: Primary language (interpreter needed?), cultural food practices/beliefs, traditional feeding practices, health belief model, family decision-making structure. How do these impact nutrition recommendations and should plan be adapted?",
            
            # 6. Social intervention referrals
            "Recommend specific social interventions: (1) WIC referral if income-eligible and <5 years (provides formula, foods, nutrition education). (2) SNAP application assistance if food insecure. (3) Formula assistance programs (manufacturer programs, foundations). (4) Transportation services (Medicaid transport). (5) Home health nursing if caregiver unable to clinic.",
            
            # 7. Care coordination and case management
            "If complex social needs: Recommend case management or care coordination services. Specify: frequency of contact, primary needs to address (housing, food, medical appointments), connection to community resources, advocacy needs (school, insurance).",
            
            # 8. Impact of social factors on prognosis
            "How do social determinants impact prognosis? If significant barriers exist: Predict challenges with plan adherence, may need modification (less frequent monitoring if transportation barrier, extended outpatient formula coverage if financial barrier). What adaptations would improve success?",
            
            # 9. Social history temporal changes
            "Document how social circumstances have changed over time: Recent job loss, housing instability, family crisis affecting care. How do temporal social changes correlate with nutritional decline timeline?",
            
            # 10. Social assessment completion
            "If social history incomplete, specify assessment needed: Validated food security screening, income documentation (for program eligibility), caregiver mental health screening (PHQ-2), home safety assessment. Explain how each domain affects nutrition outcomes and intervention planning."
        ],
        
        'clinical_insights': [
            # 1. Evidence-based guideline synthesis
            "Synthesize key clinical insights with guideline references: How does this case illustrate ASPEN pediatric malnutrition diagnostic criteria? Which WHO classification applies (SAM/MAM/stunting)? How do CDC/WHO growth references inform interpretation? Cite specific guideline sections applied (e.g., 'ASPEN 2014 Pediatric Malnutrition Consensus Statement').",
            
            # 2. Prognostic assessment with timeline
            "Provide prognostic assessment: (1) Expected clinical course over next 3-6 months (best/expected/worst case scenarios). (2) Factors influencing prognosis: severity, chronicity, etiology, underlying disease, initial intervention response, social support, caregiver capacity. (3) Long-term outlook for growth, development, and nutritional status normalization.",
            
            # 3. Critical decision points ahead
            "Identify upcoming critical decision points: (1) Week 1-2: Initial response assessment - continue current plan vs modify (triggers for modification). (2) Month 1: Reevaluate diagnosis if response inadequate - consider alternative etiology, unrecognized barrier. (3) Month 3: Transition from intensive to maintenance support - criteria for transition. What factors guide each decision?",
            
            # 4. Diagnostic teaching points
            "Key diagnostic pearls from this case: (1) Importance of serial measurements for velocity assessment (ASPEN indicator 2). (2) Correcting z-score signs (percentile <50th = negative z-score). (3) Recognizing all 4 ASPEN indicators (anthropometric, velocity, intake, physical). (4) Single-point vs serial implications for diagnostic threshold. (5) Interpreting labs in inflammatory context.",
            
            # 5. Management principles illustrated
            "Management principles demonstrated: (1) Calculating energy needs for catch-up growth (standard needs × 1.2-1.5). (2) Balancing aggressive refeeding with refeeding syndrome risk (electrolyte monitoring). (3) Route selection (oral → enteral → parenteral escalation). (4) Addressing underlying etiology not just symptoms. (5) Individualizing goals based on severity and social context.",
            
            # 6. Common pitfalls to avoid
            "What pitfalls should be avoided based on this case? (1) Diagnosing malnutrition based on single indicator without meeting threshold criteria. (2) Ignoring inflammatory state when interpreting albumin. (3) Using outdated growth references or incorrect reference for age. (4) Failing to assess food security/social barriers before prescribing expensive supplements. (5) Inadequate refeeding syndrome monitoring in severe malnutrition.",
            
            # 7. Temporal pattern significance
            "What temporal patterns are clinically significant? (1) Velocity of change: Rapid decline (1 z-score in 2 months) indicates acute process requiring urgent intervention vs slow decline (1 z-score in 12 months) suggests chronic inadequate intake. (2) Symptom-growth correlation: Did vomiting onset coincide with growth deceleration? (3) Intervention response: Is recovery trajectory meeting expected timeline?",
            
            # 8. Risk stratification and monitoring implications
            "Risk stratification from this case: (1) High-risk features requiring intensive monitoring: severe malnutrition (z<-3), refeeding risk, cardiac complications, rapid weight loss (>5% in 1 week). (2) Moderate-risk: moderate malnutrition (z -2 to -3), some intake, stable vital signs. (3) Lower-risk: mild malnutrition (z -1 to -2), adequate intake now. How does risk level determine monitoring frequency?",
            
            # 9. Interdisciplinary care coordination lessons
        "What interdisciplinary coordination is essential? (1) When to involve GI: feeding intolerance, suspected malabsorption, need for enteral access. (2) When to involve endocrine: poor growth velocity despite adequate intake, concern for growth hormone deficiency. (3) Social work role: complex social barriers, insurance issues, care coordination. (4) Feeding therapy: oral aversion, developmental feeding delays. Define triggers for each specialty referral.",
        
        # 10. Clinical reasoning about assessment limitations
        "What are the limitations of this assessment and how do they impact clinical reasoning? (1) Single-point limitations: Cannot assess ASPEN velocity criterion, limited diagnostic confidence, need for serial follow-up to confirm diagnosis. (2) Missing data impact: How does absence of intake quantification, complete NFPE, or laboratory data affect certainty? (3) What additional information would most strengthen the assessment?"
        ]
    }
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
