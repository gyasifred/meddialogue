#!/usr/bin/env python3
"""
Malnutrition Model Evaluation Script - v3.1.0 (Single-Turn JSON Evaluation)
=============================================================================
Evaluates trained MedDialogue malnutrition models on test datasets.

CRITICAL CHANGES:
  - v3.1.0: Single-turn evaluation with comprehensive JSON response (fixes context issues)
  - v3.0.0: Streamlined 3-step chain-of-thought matching training reasoning
  - v2.0.0: 5-step logical multi-turn matching training order + garbage collection

Uses single-turn inference with comprehensive JSON response:
  - One question requesting complete assessment
  - Returns: growth_assessment, diagnosis_reasoning, malnutrition_status
  - Avoids context accumulation issues from multi-turn conversations

Why Single-Turn:
  - Prevents model from echoing/spilling clinical notes in responses
  - Avoids context explosion in multi-turn conversations
  - Cleaner JSON extraction without accumulated context
  - More efficient inference (single pass)

Forces JSON output with fields:
  - growth_assessment: anthropometric measurements and trends
  - diagnosis_reasoning: clinical reasoning and evidence synthesis
  - malnutrition_status: "Malnutrition Present" or "Malnutrition Absent"

Required CSV columns:
  - txt: Clinical note text
  - DEID: Patient identifier (for tracking)
  - label: Binary label (1=Malnutrition Present, 0=Absent)

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 3.1.0
"""
import os
import sys
import logging
import argparse
import json
import re
import gc
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    matthews_corrcoef, roc_auc_score, roc_curve
)

# MedDialogue imports
from meddialogue import MedDialogue, TaskConfig
from meddialogue.models import load_model, ModelConfig
from meddialogue.config import OutputFormat
from meddialogue.inference import InferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class MalnutritionEvaluator:
    """
    Evaluator for malnutrition classification models with single-turn JSON evaluation.

    Handles:
    - Model loading (adapter or merged)
    - Single-turn inference with comprehensive JSON response
    - Batch inference with progress tracking
    - JSON response enforcement and extraction
    - Comprehensive metrics calculation
    - Results saving (CSV, JSON, plots)
    - Proper inference isolation (no history carryover between samples)
    - Memory optimization with garbage collection
    """

    # Patterns to extract malnutrition status from responses (fallback if JSON fails)
    PRESENT_PATTERNS = [
        r'\bmalnutrition\s+(is\s+)?present\b',
        r'\byes\b.*\bmalnutrition\b',
        r'\bmalnutrition\s+confirmed\b',
        r'\bmalnourished\b',
        r'\bdiagnosis:?\s+malnutrition\b',
        r'\bmalnutrition_status:?\s+(present|yes|1|true)\b',
        r'\b(present|positive)\b.*\bmalnutrition\b'
    ]

    ABSENT_PATTERNS = [
        r'\bmalnutrition\s+(is\s+)?absent\b',
        r'\bno\b.*\bmalnutrition\b',
        r'\bnot\s+malnourished\b',
        r'\bno\s+evidence\s+of\s+malnutrition\b',
        r'\bdoes\s+not\s+meet\s+criteria\b',
        r'\bmalnutrition_status:?\s+(absent|no|0|false)\b',
        r'\b(absent|negative)\b.*\bmalnutrition\b'
    ]

    def __init__(
        self,
        model_path: str,
        model_type: str = "llama",
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
        cuda_device: Optional[int] = None
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.cuda_device = cuda_device

        self.model = None
        self.tokenizer = None
        self.device = None
        self.inference_pipeline = None

        # Task configuration for malnutrition
        self.task_config = TaskConfig(
            task_name="pediatric_malnutrition_evaluation",
            task_description="Evaluate malnutrition status with clinical reasoning",
            input_field="txt",
            output_fields=["malnutrition_status"],
            question_templates={
                "malnutrition_status": [
                    "Is malnutrition present or absent?",
                    "Classify malnutrition status"
                ]
            },
            output_formats=[OutputFormat.JSON, OutputFormat.TEXT]
        )

        logger.info(f"Evaluator initialized for {model_type} (Single-Turn JSON v3.1)")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Max tokens: {max_new_tokens}")
        logger.info(f"Evaluation: Single-turn with comprehensive JSON response")

    def load_model(self, max_seq_length: int = 16384):
        """Load trained model for inference."""
        logger.info("="*80)
        logger.info("Loading model for evaluation...")
        logger.info("="*80)

        # Device
        if self.cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)
            self.device = torch.device("cuda:0")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Device: {self.device}")

        # Model registry
        from meddialogue.models import ModelRegistry
        try:
            model_info = ModelRegistry.get_config(self.model_type)
        except KeyError:
            available = ModelRegistry.list_models()
            raise ValueError(f"Unknown model type: {self.model_type}. Available: {available}")

        model_config = ModelConfig(
            model_name=self.model_path,
            model_type=self.model_type,
            chat_template=model_info["chat_template"],
            max_seq_length=max_seq_length
        )

        logger.info(f"Loading from: {self.model_path}")
        self.model, self.tokenizer = load_model(model_config, max_seq_length)

        # Handle quantized models
        is_quantized = getattr(self.model, "is_loaded_in_8bit", False) or \
                       getattr(self.model, "is_loaded_in_4bit", False)

        if not is_quantized:
            self.model.to(self.device)
        else:
            logger.info("Quantized model (8-bit/4-bit) – skipping .to()")

        self.model.eval()
        logger.info("Model loaded successfully")

        # Initialize inference pipeline
        self.inference_pipeline = InferencePipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task_config=self.task_config,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device=self.device
        )
        logger.info("Inference pipeline initialized for single-turn JSON evaluation")
        logger.info("="*80)

    def infer_single(self, clinical_note: str) -> Tuple[str, int, float]:
        """
        Run inference on single clinical note using single-turn evaluation.

        Uses a single comprehensive question that returns all assessment in one JSON response.
        This avoids context accumulation issues in multi-turn conversations.

        Args:
            clinical_note: Clinical text

        Returns:
            Tuple of (combined_response, predicted_label, confidence)
        """
        print("\n" + "="*80)
        print("SINGLE-TURN EVALUATION (JSON Template with Questions)")
        print("="*80)

        # Provide JSON template with field names as keys and questions as values
        # Model completes by replacing questions with answers
        json_template = {
            "growth_assessment": "What are ALL anthropometric measurements with DATES? Calculate trends and trajectories.",
            "diagnosis_reasoning": "What's your diagnosis with complete clinical reasoning? Synthesize ALL evidence temporally.",
            "malnutrition_status": "Is malnutrition present or absent? State 'Malnutrition Present' or 'Malnutrition Absent'."
        }

        evaluation_question = (
            "Complete this JSON by replacing each question with your answer:\n\n"
            f"{json.dumps(json_template, indent=2)}"
        )

        print(f"\nQ: {evaluation_question}\n")
        print("=" * 80)
        print("Running single-turn inference...\n")

        # Use single-turn inference
        response = self.inference_pipeline.infer(
            clinical_note=clinical_note,
            question=evaluation_question,
            output_format=OutputFormat.JSON,
            return_full_response=False
        )

        # Convert response to string for processing
        if isinstance(response, dict):
            response_str = json.dumps(response)
            json_response = response
        else:
            response_str = str(response)
            json_response = self._extract_json_response(response_str)

        print(f"RESPONSE:\n{response_str}\n")
        print("JSON OUTPUT:")
        print(json.dumps(json_response, indent=2))
        print("="*80 + "\n")

        # Parse classification from JSON
        predicted_label, confidence = self._parse_from_json_or_fallback(
            json_response, response_str
        )

        # If JSON parsing fails, try regex on full response
        if confidence < 0.7:
            predicted_label, confidence = self._parse_classification(response_str)

        return response_str, predicted_label, confidence

    def _extract_json_response(self, response: str) -> Dict:
        """
        Extract JSON from response with robust handling.

        Handles field name keys and extracts malnutrition status from values.
        Uses proper JSON parsing with brace matching for nested content.
        """
        def is_valid_assessment(parsed: Dict) -> bool:
            """Check if parsed dict is a valid assessment JSON."""
            if not isinstance(parsed, dict) or len(parsed) == 0:
                return False
            # Accept any dict with at least one key (model response)
            return True

        def normalize_json(parsed: Dict) -> Dict:
            """Normalize JSON to standard field names and extract status."""
            normalized = {}
            malnutrition_value = None

            for key, value in parsed.items():
                key_lower = key.lower()
                value_str = str(value).lower() if value else ""

                # Map to standard field names
                if key == "growth_assessment" or "anthropometric" in key_lower or "measurement" in key_lower or "growth" in key_lower:
                    normalized["growth_assessment"] = value
                elif key == "diagnosis_reasoning" or "diagnosis" in key_lower or "reasoning" in key_lower:
                    normalized["diagnosis_reasoning"] = value
                elif key == "malnutrition_status" or "malnutrition" in key_lower:
                    normalized["malnutrition_status"] = value
                    malnutrition_value = value_str
                else:
                    normalized[key] = value
                    # Check if value contains malnutrition status
                    if "malnutrition" in value_str and ("present" in value_str or "absent" in value_str):
                        if "malnutrition_status" not in normalized:
                            if "present" in value_str:
                                normalized["malnutrition_status"] = "Malnutrition Present"
                            else:
                                normalized["malnutrition_status"] = "Malnutrition Absent"

            # Ensure malnutrition_status exists
            if "malnutrition_status" not in normalized:
                # Search all values for status
                for value in parsed.values():
                    value_str = str(value).lower()
                    if "malnutrition present" in value_str:
                        normalized["malnutrition_status"] = "Malnutrition Present"
                        break
                    elif "malnutrition absent" in value_str or "no malnutrition" in value_str:
                        normalized["malnutrition_status"] = "Malnutrition Absent"
                        break

            return normalized

        def extract_json_from_text(text: str) -> Optional[Dict]:
            """Extract JSON object from text using brace matching."""
            start_idx = text.find('{')
            if start_idx == -1:
                return None

            brace_count = 0
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            return None
            return None

        # First try: Parse the entire response as JSON
        try:
            parsed = json.loads(response.strip())
            if is_valid_assessment(parsed):
                return normalize_json(parsed)
        except json.JSONDecodeError:
            pass

        # Second try: Extract first JSON object
        parsed = extract_json_from_text(response)
        if parsed and is_valid_assessment(parsed):
            return normalize_json(parsed)

        # Third try: Look for last JSON object (in case first was an example)
        last_brace = response.rfind('{')
        first_brace = response.find('{')
        if last_brace != first_brace and last_brace != -1:
            parsed = extract_json_from_text(response[last_brace:])
            if parsed and is_valid_assessment(parsed):
                return normalize_json(parsed)

        # Fourth try: Extract status from plain text using regex
        response_lower = response.lower()
        if "malnutrition present" in response_lower:
            return {"malnutrition_status": "Malnutrition Present"}
        elif "malnutrition absent" in response_lower or "no malnutrition" in response_lower:
            return {"malnutrition_status": "Malnutrition Absent"}

        # Fallback: return default with warning
        logger.warning(f"Could not extract valid JSON from response: {response[:300]}")
        return {"malnutrition_status": "Malnutrition Absent"}

    def _parse_from_json_or_fallback(self, json_obj: Dict, raw_response: str) -> Tuple[int, float]:
        """Parse label from JSON, fallback to regex if invalid."""
        status = json_obj.get("malnutrition_status", "")

        if not status:
            # Try to find status in any field value
            for value in json_obj.values():
                value_str = str(value).lower()
                if "malnutrition present" in value_str:
                    return 1, 0.90
                elif "malnutrition absent" in value_str or "no malnutrition" in value_str:
                    return 0, 0.90
            # Fallback to regex
            return self._parse_classification(raw_response)

        status_lower = str(status).lower()

        # Check for various status formats
        if "present" in status_lower and "absent" not in status_lower:
            return 1, 0.95
        elif "absent" in status_lower or "no malnutrition" in status_lower or "not malnourished" in status_lower:
            return 0, 0.95
        elif status_lower in ["yes", "1", "true", "positive"]:
            return 1, 0.90
        elif status_lower in ["no", "0", "false", "negative"]:
            return 0, 0.90
        else:
            # Fallback to regex
            return self._parse_classification(raw_response)

    def _parse_classification(self, response: str) -> Tuple[int, float]:
        """Legacy regex parser (fallback only)."""
        response_lower = response.lower()
        present_score = sum(1 for pattern in self.PRESENT_PATTERNS if re.search(pattern, response_lower))
        absent_score = sum(1 for pattern in self.ABSENT_PATTERNS if re.search(pattern, response_lower))

        if present_score > absent_score:
            return 1, min(0.9, 0.5 + 0.1 * present_score)
        elif absent_score > present_score:
            return 0, min(0.9, 0.5 + 0.1 * absent_score)
        else:
            logger.warning(f"Ambiguous response: {response[:100]}")
            return 0, 0.5

    def evaluate_dataset(
        self,
        test_csv: str,
        output_dir: str,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """Evaluate model on test dataset using single-turn JSON evaluation."""
        logger.info("="*80)
        logger.info("EVALUATION STARTING (SINGLE-TURN JSON EVALUATION)")
        logger.info("="*80)
        logger.info(f"Test CSV: {test_csv}")
        logger.info(f"Output dir: {output_dir}")
        logger.info("="*80)

        df = pd.read_csv(test_csv)
        required_cols = ["txt", "DEID", "label"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Loaded {len(df)} test examples")
        label_counts = df["label"].value_counts()
        logger.info(f"Label distribution:")
        logger.info(f"  Absent (0): {label_counts.get(0, 0)}")
        logger.info(f"  Present (1): {label_counts.get(1, 0)}")
        logger.info("-"*80)

        predictions = []
        confidences = []
        responses = []
        json_outputs = []

        logger.info("Running inference on test set...")
        # IMPORTANT: Each infer_single() call creates fresh conversation (no history carryover)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            try:
                # Each sample processed independently with clean state
                response, pred, conf = self.infer_single(row["txt"])
                json_obj = self._extract_json_response(response)
                predictions.append(pred)
                confidences.append(conf)
                responses.append(response)
                json_outputs.append(json_obj)

                # Memory cleanup: Periodic garbage collection every 10 samples
                if (idx + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error on row {idx} (DEID: {row['DEID']}): {e}")
                predictions.append(-1)
                confidences.append(0.0)
                responses.append(f"ERROR: {str(e)}")
                json_outputs.append({"malnutrition_status": "ERROR"})

        df["predicted_label"] = predictions
        df["confidence"] = confidences
        df["model_response"] = responses
        df["json_output"] = [json.dumps(j) for j in json_outputs]
        df["correct"] = df["label"] == df["predicted_label"]

        valid_df = df[df["predicted_label"] != -1].copy()
        error_count = len(df) - len(valid_df)

        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during inference")

        logger.info(f"Completed inference on {len(valid_df)} valid examples")

        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Final memory cleanup completed")
        logger.info("="*80)

        metrics = self._calculate_metrics(
            valid_df["label"].tolist(),
            valid_df["predicted_label"].tolist()
        )

        results = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "test_csv": test_csv,
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(df),
            "valid_examples": len(valid_df),
            "error_examples": error_count,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "output_mode": "Single-Turn JSON Evaluation",
            "reasoning_steps": 1,
            "reasoning_process": "Single comprehensive assessment with JSON response",
            "json_fields": "growth_assessment, diagnosis_reasoning, malnutrition_status",
            "evaluation_method": "Single-turn inference (avoids context accumulation)",
            "memory_optimization": "Garbage collection every 10 samples + final cleanup",
            "metrics": metrics
        }

        os.makedirs(output_dir, exist_ok=True)

        if save_predictions:
            pred_path = os.path.join(output_dir, "predictions.csv")
            df.to_csv(pred_path, index=False)
            logger.info(f"Predictions saved: {pred_path}")

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metrics saved: {metrics_path}")

        self._save_summary_report(results, valid_df, output_dir)

        return results

    def _calculate_metrics(self, true_labels: List[int], predictions: List[int]) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        mcc = matthews_corrcoef(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall
        try:
            auc = roc_auc_score(true_labels, predictions)
        except ValueError:
            auc = None

        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(true_labels, predictions, average=None, zero_division=0)

        class_report = classification_report(
            true_labels, predictions,
            target_names=["Absent", "Present"],
            output_dict=True, zero_division=0
        )

        return {
            "overall": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "mcc": float(mcc),
                "auc_roc": float(auc) if auc is not None else None
            },
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "matrix": cm.tolist()
            },
            "per_class": {
                "absent": {
                    "precision": float(precision_per_class[0]),
                    "recall": float(recall_per_class[0]),
                    "f1_score": float(f1_per_class[0]),
                    "support": int(support_per_class[0])
                },
                "present": {
                    "precision": float(precision_per_class[1]),
                    "recall": float(recall_per_class[1]),
                    "f1_score": float(f1_per_class[1]),
                    "support": int(support_per_class[1])
                }
            },
            "classification_report": class_report
        }

    def _save_summary_report(self, results: Dict[str, Any], df: pd.DataFrame, output_dir: str):
        """Save human-readable summary report."""
        report_path = os.path.join(output_dir, "evaluation_summary.txt")
        metrics = results["metrics"]
        overall = metrics["overall"]
        cm = metrics["confusion_matrix"]
        per_class = metrics["per_class"]

        with open(report_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("MALNUTRITION MODEL EVALUATION SUMMARY (SINGLE-TURN JSON)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {results['model_path']}\n")
            f.write(f"Model Type: {results['model_type']}\n")
            f.write(f"Test Data: {results['test_csv']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Temperature: {results['temperature']}\n")
            f.write(f"Output Mode: {results['output_mode']}\n")
            f.write(f"Evaluation Method: {results['evaluation_method']}\n")
            f.write(f"JSON Fields: {results['json_fields']}\n")
            f.write("\n" + "-"*80 + "\n\n")
            f.write("DATASET STATISTICS:\n")
            f.write(f"  Total examples: {results['total_examples']}\n")
            f.write(f"  Valid examples: {results['valid_examples']}\n")
            f.write(f"  Errors: {results['error_examples']}\n")
            f.write("\n")
            label_dist = df["label"].value_counts()
            f.write("TRUE LABEL DISTRIBUTION:\n")
            f.write(f"  Absent (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(df)*100:.1f}%)\n")
            f.write(f"  Present (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(df)*100:.1f}%)\n")
            f.write("\n" + "-"*80 + "\n\n")
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"  Accuracy: {overall['accuracy']:.4f}\n")
            f.write(f"  Precision: {overall['precision']:.4f}\n")
            f.write(f"  Recall/Sensitivity: {overall['recall']:.4f}\n")
            f.write(f"  Specificity: {overall['specificity']:.4f}\n")
            f.write(f"  F1-Score: {overall['f1_score']:.4f}\n")
            f.write(f"  MCC: {overall['mcc']:.4f}\n")
            if overall['auc_roc']:
                f.write(f"  AUC-ROC: {overall['auc_roc']:.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")
            f.write("CONFUSION MATRIX:\n")
            f.write("                Predicted\n")
            f.write("               Absent Present\n")
            f.write(f"  Actual Absent  {cm['true_negative']:4d}   {cm['false_positive']:4d}\n")
            f.write(f"         Present {cm['false_negative']:4d}   {cm['true_positive']:4d}\n")
            f.write("\n" + "-"*80 + "\n\n")
            f.write("PER-CLASS METRICS:\n")
            f.write(f"  Absent (0):\n")
            f.write(f"    Precision: {per_class['absent']['precision']:.4f}\n")
            f.write(f"    Recall: {per_class['absent']['recall']:.4f}\n")
            f.write(f"    F1-Score: {per_class['absent']['f1_score']:.4f}\n")
            f.write(f"    Support: {per_class['absent']['support']}\n")
            f.write("\n")
            f.write(f"  Present (1):\n")
            f.write(f"    Precision: {per_class['present']['precision']:.4f}\n")
            f.write(f"    Recall: {per_class['present']['recall']:.4f}\n")
            f.write(f"    F1-Score: {per_class['present']['f1_score']:.4f}\n")
            f.write(f"    Support: {per_class['present']['support']}\n")
            f.write("\n" + "="*80 + "\n")

        logger.info(f"Summary report saved: {report_path}")

    def print_results(self, results: Dict[str, Any]):
        """Print formatted results to console."""
        metrics = results["metrics"]
        overall = metrics["overall"]
        cm = metrics["confusion_matrix"]

        print("\n" + "="*80)
        print("EVALUATION RESULTS (SINGLE-TURN JSON EVALUATION)")
        print("="*80)
        print(f"\nModel: {results['model_path']}")
        print(f"Test examples: {results['valid_examples']}")
        print(f"Output Mode: {results['output_mode']}")
        print(f"Method: {results['evaluation_method']}")
        print("\nOVERALL PERFORMANCE:")
        print(f"  Accuracy: {overall['accuracy']:.4f}")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall (Sens): {overall['recall']:.4f}")
        print(f"  Specificity: {overall['specificity']:.4f}")
        print(f"  F1-Score: {overall['f1_score']:.4f}")
        print(f"  MCC: {overall['mcc']:.4f}")
        if overall['auc_roc']:
            print(f"  AUC-ROC: {overall['auc_roc']:.4f}")
        print("\nCONFUSION MATRIX:")
        print("                Predicted")
        print("               Absent Present")
        print(f"  Actual Absent  {cm['true_negative']:4d}   {cm['false_positive']:4d}")
        print(f"         Present {cm['false_negative']:4d}   {cm['true_positive']:4d}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained malnutrition model on test set (Single-Turn JSON v3.1.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluate_malnutrition.py \\
    --model ./malnutrition_models_v1_1/llama/merged_llama_20241125_143022 \\
    --csv test_data.csv \\
    --output ./eval_results

Single-Turn JSON Evaluation:
  - One comprehensive question requesting complete assessment
  - Returns JSON with: growth_assessment, diagnosis_reasoning, malnutrition_status
  - Avoids context accumulation issues from multi-turn conversations
  - More efficient and reliable than multi-turn approach
        """
    )

    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--csv", required=True, help="Path to test CSV")
    parser.add_argument("--output", default="./evaluation_results", help="Output directory")
    parser.add_argument("--model_type", default="llama", choices=["llama", "phi-4", "mistral", "qwen"])
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--cuda_device", type=int, help="CUDA device index")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error(f"Model path not found: {args.model}")
        return 1
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        return 1

    logger.info("")
    logger.info("="*80)
    logger.info("MALNUTRITION MODEL EVALUATION v3.1.0 (SINGLE-TURN JSON)")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Test CSV: {args.csv}")
    logger.info(f"Output: {args.output}")
    logger.info("-"*80)
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Max seq length: {args.max_seq_length}")
    if args.cuda_device is not None:
        logger.info(f"CUDA device: {args.cuda_device}")
    logger.info("="*80)
    logger.info("")

    try:
        evaluator = MalnutritionEvaluator(
            model_path=args.model,
            model_type=args.model_type,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            cuda_device=args.cuda_device
        )

        evaluator.load_model(max_seq_length=args.max_seq_length)

        results = evaluator.evaluate_dataset(
            test_csv=args.csv,
            output_dir=args.output,
            save_predictions=True
        )

        evaluator.print_results(results)

        logger.info(f"\nEvaluation completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        logger.info(f"  - predictions.csv: Includes model responses and json_output")
        logger.info(f"  - metrics.json: Comprehensive metrics with evaluation metadata")
        logger.info(f"  - evaluation_summary.txt: Human-readable report")
        logger.info("")
        logger.info("Single-Turn JSON Evaluation Summary:")
        logger.info(f"  - One comprehensive question per clinical note")
        logger.info(f"  - Returns JSON with: growth_assessment, diagnosis_reasoning, malnutrition_status")
        logger.info(f"  - Avoids context accumulation issues from multi-turn approach")

        return 0

    except Exception as e:
        logger.error("="*80)
        logger.error("EVALUATION FAILED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
