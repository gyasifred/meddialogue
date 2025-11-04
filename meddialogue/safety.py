"""
Safety Guardrails for MedDialogue
=================================

Comprehensive safety checks including PII detection, bias monitoring,
and clinical validation to ensure responsible healthcare AI.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter
import pandas as pd

from .config import SafetyConfig, PIISensitivity

logger = logging.getLogger(__name__)


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""
    has_pii: bool
    pii_types: List[str]
    confidence: float
    locations: List[Tuple[int, int]]
    anonymized_text: str


@dataclass
class BiasMetrics:
    """Bias monitoring metrics."""
    demographic_distribution: Dict[str, Counter]
    imbalance_score: float
    warnings: List[str]
    is_balanced: bool


@dataclass
class ValidationResult:
    """Clinical validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validated_codes: List[str]


class PIIDetector:
    """
    Detect and anonymize personally identifiable information (PII) in clinical text.
    
    Detects: SSN, phone numbers, email addresses, MRN, dates of birth, addresses.
    Configurable sensitivity levels for different use cases.
    """
    
    def __init__(self, config: SafetyConfig):
        """
        Initialize PII detector.
        
        Args:
            config: Safety configuration with PII patterns and sensitivity
        """
        self.config = config
        self.patterns = config.pii_patterns.copy()
        
        # Add custom patterns if provided
        if config.custom_pii_patterns:
            self.patterns.update(config.custom_pii_patterns)
        
        self.sensitivity = config.pii_sensitivity
        
        # Additional patterns based on sensitivity level
        self.extended_patterns = {
            "name_prefix": r"\b(Mr\.|Mrs\.|Ms\.|Dr\.|Miss)\s+[A-Z][a-z]+",
            "address": r"\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)",
            "zip": r"\b\d{5}(?:-\d{4})?\b",
            "age_dob": r"\b(age|DOB|born on|birthdate)[\s:]+[\d/-]+\b"
        }
    
    def detect(self, text: str) -> PIIDetectionResult:
        """
        Detect PII in text.
        
        Args:
            text: Clinical text to check
            
        Returns:
            PIIDetectionResult with detection details
        """
        if not self.config.enable_pii_detection:
            return PIIDetectionResult(False, [], 0.0, [], text)
        
        pii_types = []
        locations = []
        confidence_scores = []
        
        # Check base patterns
        for pii_type, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                pii_types.append(pii_type)
                locations.extend([(m.start(), m.end()) for m in matches])
                confidence_scores.append(0.9)  # High confidence for regex matches
        
        # Check extended patterns for higher sensitivity levels
        if self.sensitivity in [PIISensitivity.HIGH, PIISensitivity.MAXIMUM]:
            for pii_type, pattern in self.extended_patterns.items():
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    pii_types.append(pii_type)
                    locations.extend([(m.start(), m.end()) for m in matches])
                    confidence_scores.append(0.7)  # Medium confidence for extended patterns
        
        has_pii = len(pii_types) > 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Anonymize if PII found
        anonymized_text = self._anonymize(text, locations) if has_pii else text
        
        if has_pii:
            logger.warning(f"PII detected: {set(pii_types)} (confidence: {avg_confidence:.2f})")
            if self.config.log_safety_events:
                logger.info(f"PII types found: {', '.join(set(pii_types))}")
        
        # Check threshold
        if has_pii and avg_confidence > self.config.max_pii_threshold:
            logger.error(f"PII confidence {avg_confidence:.2f} exceeds threshold {self.config.max_pii_threshold}")
        
        return PIIDetectionResult(
            has_pii=has_pii,
            pii_types=list(set(pii_types)),
            confidence=avg_confidence,
            locations=locations,
            anonymized_text=anonymized_text
        )
    
    def _anonymize(self, text: str, locations: List[Tuple[int, int]]) -> str:
        """
        Anonymize detected PII in text.
        
        Args:
            text: Original text
            locations: List of (start, end) positions of PII
            
        Returns:
            Anonymized text
        """
        # Sort locations by start position (reverse order for replacement)
        sorted_locations = sorted(set(locations), key=lambda x: x[0], reverse=True)
        
        anonymized = text
        for start, end in sorted_locations:
            anonymized = anonymized[:start] + "[REDACTED]" + anonymized[end:]
        
        return anonymized
    
    def batch_detect(self, texts: List[str]) -> List[PIIDetectionResult]:
        """
        Detect PII in multiple texts.
        
        Args:
            texts: List of clinical texts
            
        Returns:
            List of PIIDetectionResults
        """
        return [self.detect(text) for text in texts]


class BiasMonitor:
    """
    Monitor training data for demographic and case distribution bias.
    
    Ensures balanced representation across age, gender, race, and case types.
    
    Note: Bias detection is a feature, not always a blocker. For example,
    a diabetes dataset might naturally skew to patients over 50. The monitor
    alerts you to imbalances so you can make informed decisions.
    """
    
    def __init__(self, config: SafetyConfig):
        """
        Initialize bias monitor.
        
        Args:
            config: Safety configuration with demographic fields to monitor
        """
        self.config = config
        self.demographic_fields = config.bias_demographic_fields
        self.imbalance_threshold = 0.3  # 30% deviation triggers warning
    
    def analyze(self, data: pd.DataFrame, label_field: str = "label") -> BiasMetrics:
        """
        Analyze dataset for bias.
        
        Args:
            data: Training data DataFrame
            label_field: Column name for labels (e.g., "diagnosis")
            
        Returns:
            BiasMetrics with analysis results
        """
        if not self.config.enable_bias_monitoring:
            return BiasMetrics({}, 0.0, [], True)
        
        demographic_dist = {}
        warnings = []
        
        # Check label distribution
        if label_field in data.columns:
            label_counts = Counter(data[label_field])
            total = sum(label_counts.values())
            label_dist = {k: v/total for k, v in label_counts.items()}
            demographic_dist['labels'] = label_counts
            
            # Check for severe imbalance
            if len(label_dist) > 1:
                max_ratio = max(label_dist.values()) / min(label_dist.values())
                if max_ratio > 2.0:
                    warning = f"Label imbalance detected: {dict(label_dist)}. Max/min ratio: {max_ratio:.2f}"
                    warnings.append(warning)
                    logger.warning(warning)
        
        # Check demographic fields if present
        for field in self.demographic_fields:
            if field in data.columns:
                field_counts = Counter(data[field].dropna())
                demographic_dist[field] = field_counts
                
                # Check distribution
                if len(field_counts) > 1:
                    total = sum(field_counts.values())
                    field_dist = {k: v/total for k, v in field_counts.items()}
                    max_val = max(field_dist.values())
                    min_val = min(field_dist.values())
                    
                    if (max_val - min_val) > self.imbalance_threshold:
                        warning = f"Imbalance in '{field}': {dict(field_dist)}"
                        warnings.append(warning)
                        logger.warning(warning)
        
        # Calculate overall imbalance score
        imbalance_score = self._calculate_imbalance_score(demographic_dist)
        is_balanced = imbalance_score < self.imbalance_threshold
        
        if warnings and self.config.log_safety_events:
            logger.info(f"Bias analysis completed with {len(warnings)} warnings")
        
        return BiasMetrics(
            demographic_distribution=demographic_dist,
            imbalance_score=imbalance_score,
            warnings=warnings,
            is_balanced=is_balanced
        )
    
    def _calculate_imbalance_score(self, distributions: Dict[str, Counter]) -> float:
        """Calculate aggregate imbalance score."""
        if not distributions:
            return 0.0
        
        imbalances = []
        for field, counts in distributions.items():
            if len(counts) > 1:
                total = sum(counts.values())
                proportions = [v/total for v in counts.values()]
                max_prop = max(proportions)
                min_prop = min(proportions)
                imbalances.append(max_prop - min_prop)
        
        return sum(imbalances) / len(imbalances) if imbalances else 0.0


class ClinicalValidator:
    """
    Validate medical terminology, diagnostic codes, and clinical logic.
    
    Ensures outputs contain valid medical terms and proper codes.
    
    Medical Terms: By default uses common clinical terms. Users can extend
    with custom terms (e.g., ICD-9 for legacy systems, SNOMED for drugs,
    lab-specific codes). ICD-10 is the default standard for modern healthcare.
    """
    
    def __init__(self, config: SafetyConfig, task_config: Optional[Any] = None):
        """
        Initialize clinical validator.
        
        Args:
            config: Safety configuration
            task_config: Task configuration with valid codes and terminology
        """
        self.config = config
        self.task_config = task_config
        
        # Base medical terminology (ICD-10 era - standard in modern healthcare)
        self.valid_medical_terms = {
            "bmi", "weight", "height", "percentile", "z-score",
            "diagnosis", "assessment", "severity", "mild", "moderate", "severe",
            "treatment", "medication", "therapy", "intervention"
        }
        
        # Add custom medical terms from config
        if config.custom_medical_terms:
            self.valid_medical_terms.update(set(term.lower() for term in config.custom_medical_terms))
        
        # Add task-specific terminology
        if task_config and hasattr(task_config, 'medical_terminology'):
            for category, terms in task_config.medical_terminology.items():
                self.valid_medical_terms.update(set(term.lower() for term in terms))
        
        # ICD-10 pattern (simplified - can be extended for ICD-9, SNOMED, etc.)
        # ICD-10: Standard format is letter + 2 digits + optional decimal + up to 4 chars
        self.icd10_pattern = r"\b[A-TV-Z][0-9]{2}\.?[0-9A-TV-Z]{0,4}\b"
        
        # ICD-9 pattern (if users need legacy support)
        self.icd9_pattern = r"\b[0-9]{3}\.?[0-9]{0,2}\b"
    
    def validate(self, text: str, expected_fields: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate clinical text.
        
        Args:
            text: Clinical text to validate
            expected_fields: Expected fields in output
            
        Returns:
            ValidationResult with validation details
        """
        if not self.config.enable_clinical_validation:
            return ValidationResult(True, [], [], [])
        
        errors = []
        warnings = []
        validated_codes = []
        
        # Check for medical terminology
        text_lower = text.lower()
        found_medical_terms = [term for term in self.valid_medical_terms if term in text_lower]
        
        if not found_medical_terms and self.config.require_icd_validation:
            warnings.append("No recognized medical terminology found")
        
        # Validate ICD codes if required
        if self.config.require_icd_validation:
            # Check ICD-10
            icd10_codes = re.findall(self.icd10_pattern, text)
            
            # Check ICD-9 (for legacy systems)
            icd9_codes = re.findall(self.icd9_pattern, text)
            
            all_codes = icd10_codes + icd9_codes
            
            if all_codes:
                validated_codes = all_codes
                # Here you could validate against actual ICD database
                # For now, we just verify format
            else:
                warnings.append("No ICD codes found (set require_icd_validation=False if not needed)")
        
        # Check expected fields
        if expected_fields:
            for field in expected_fields:
                if field.lower() not in text_lower and field.replace('_', ' ').lower() not in text_lower:
                    errors.append(f"Missing expected field: {field}")
        
        # Validate severity levels if present
        if self.task_config and hasattr(self.task_config, 'severity_levels'):
            if "severity" in text_lower:
                severity_found = False
                for severity in self.task_config.severity_levels:
                    if severity.lower() in text_lower:
                        severity_found = True
                        break
                
                if not severity_found:
                    warnings.append(f"Severity mentioned but no valid level found. Expected: {self.task_config.severity_levels}")
        
        is_valid = len(errors) == 0
        
        if errors:
            logger.error(f"Clinical validation errors: {errors}")
        if warnings and self.config.log_safety_events:
            logger.warning(f"Clinical validation warnings: {warnings}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validated_codes=validated_codes
        )


class SafetyChecker:
    """
    Unified safety checker combining all safety modules.
    
    Provides a single interface for running all safety checks.
    Use this as the main entry point for safety validation.
    """
    
    def __init__(self, config: SafetyConfig, task_config: Optional[Any] = None):
        """
        Initialize safety checker.
        
        Args:
            config: Safety configuration
            task_config: Task configuration for clinical validation
        """
        self.config = config
        self.pii_detector = PIIDetector(config)
        self.bias_monitor = BiasMonitor(config)
        self.clinical_validator = ClinicalValidator(config, task_config)
        
        logger.info("SafetyChecker initialized")
        logger.info(f"  PII detection: {config.enable_pii_detection}")
        logger.info(f"  Bias monitoring: {config.enable_bias_monitoring}")
        logger.info(f"  Clinical validation: {config.enable_clinical_validation}")
    
    def check_text(self, text: str, expected_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run all safety checks on text.
        
        Args:
            text: Text to check
            expected_fields: Expected fields in text
            
        Returns:
            Dictionary with all safety check results
        """
        pii_result = self.pii_detector.detect(text)
        validation_result = self.clinical_validator.validate(text, expected_fields)
        
        passed = True
        if self.config.block_on_safety_failure:
            passed = not pii_result.has_pii and validation_result.is_valid
        else:
            # Log warnings but don't block
            if pii_result.has_pii:
                logger.warning("PII detected but not blocking (block_on_safety_failure=False)")
            if not validation_result.is_valid:
                logger.warning("Validation failed but not blocking (block_on_safety_failure=False)")
        
        return {
            "passed": passed,
            "pii_detected": pii_result.has_pii,
            "pii_types": pii_result.pii_types,
            "pii_confidence": pii_result.confidence,
            "validation_errors": validation_result.errors,
            "validation_warnings": validation_result.warnings,
            "validated_codes": validation_result.validated_codes,
            "anonymized_text": pii_result.anonymized_text
        }
    
    def check_dataset(self, data: pd.DataFrame, text_column: str = "clinical_note",
                     label_column: str = "label") -> Dict[str, Any]:
        """
        Run safety checks on entire dataset.
        
        Args:
            data: DataFrame with training data
            text_column: Column name containing clinical text
            label_column: Column name containing labels
            
        Returns:
            Dictionary with dataset-level safety results
        """
        logger.info(f"Running safety checks on dataset with {len(data)} samples...")
        
        # Check for PII in dataset
        if text_column in data.columns:
            pii_results = self.pii_detector.batch_detect(data[text_column].tolist())
            pii_count = sum(1 for r in pii_results if r.has_pii)
            anonymized_texts = [r.anonymized_text for r in pii_results]
        else:
            pii_results = []
            pii_count = 0
            anonymized_texts = []
            logger.warning(f"Text column '{text_column}' not found in dataset")
        
        # Check for bias
        bias_metrics = self.bias_monitor.analyze(data, label_field=label_column)
        
        # Determine if safe for training
        safe_for_training = True
        
        if self.config.block_on_safety_failure:
            if pii_count > 0:
                safe_for_training = False
                logger.error(f"PII detected in {pii_count} samples")
            
            if not bias_metrics.is_balanced:
                logger.warning(f"Dataset imbalance detected (score: {bias_metrics.imbalance_score:.2f})")
                # Note: We don't block on bias by default as some imbalance may be expected
        
        results = {
            "total_samples": len(data),
            "pii_detected_count": pii_count,
            "pii_percentage": (pii_count / len(data) * 100) if len(data) > 0 else 0,
            "bias_metrics": {
                "imbalance_score": bias_metrics.imbalance_score,
                "is_balanced": bias_metrics.is_balanced,
                "warnings": bias_metrics.warnings,
                "distribution": {k: dict(v) for k, v in bias_metrics.demographic_distribution.items()}
            },
            "safe_for_training": safe_for_training,
            "anonymized_texts": anonymized_texts
        }
        
        logger.info(f"Safety checks completed:")
        logger.info(f"  PII detected: {pii_count}/{len(data)} samples ({results['pii_percentage']:.1f}%)")
        logger.info(f"  Bias score: {bias_metrics.imbalance_score:.2f}")
        logger.info(f"  Safe for training: {safe_for_training}")
        
        return results