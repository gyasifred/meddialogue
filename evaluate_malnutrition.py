#!/usr/bin/env python3
"""
Malnutrition Model Evaluation Script - v1.7.0 (Optimized - 2 Key Aspects Only)
==================================================================================
Evaluates trained MedDialogue malnutrition models on test datasets.

CRITICAL FIXES:
  - v1.7.0: Reduced Step 1 to 2 key aspects (growth/anthropometrics + clinical signs) for speed
  - v1.6.0: Shortened questions for faster inference; verified clean state between samples
  - v1.5.0: Fixed JSON extraction bug (was extracting wrong status from question examples)
  - v1.4.0: Now uses TRUE multi-turn conversations matching training format

Uses 2-step multi-turn clinical reasoning (OPTIMIZED - Only 2 key aspects per step):
  1. Clinical Assessment: Focus on growth/anthropometrics + clinical signs only
  2. Final Classification: JSON output based on Step 1

Training/Inference Match:
  - Turn 1: Question + Clinical Note → Assessment
  - Turn 2: Question only (with Turn 1 context) → Classification

JSON Extraction (v1.5.0 FIX):
  - Handles single quotes ('Malnutrition Present') and double quotes
  - Extracts LAST valid JSON object (actual answer, not question examples)
  - Non-greedy regex prevents matching across multiple JSON objects
  - Robust handling of embedded examples in prompt

Forces JSON output: {"malnutrition_status": "Malnutrition Present/Absent"}
Prints reasoning steps to terminal for transparency.

Required CSV columns:
  - txt: Clinical note text
  - DEID: Patient identifier (for tracking)
  - label: Binary label (1=Malnutrition Present, 0=Absent)

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.7.0
"""
import os
import sys
import logging
import argparse
import json
import re
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
    Evaluator for malnutrition classification models with 2-step reasoning.

    Handles:
    - Model loading (adapter or merged)
    - Streamlined clinical reasoning (2 questions)
    - Batch inference with progress tracking
    - JSON response enforcement
    - Comprehensive metrics calculation
    - Results saving (CSV, JSON, plots)
    - Proper inference isolation (no history carryover between samples)
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

        logger.info(f"Evaluator initialized for {model_type} (True Multi-Turn Reasoning v1.4)")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Max tokens: {max_new_tokens}")
        logger.info(f"Reasoning: 2-step multi-turn (Assessment → Classification with context)")

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
        logger.info("Inference pipeline initialized for 2-step multi-turn reasoning")
        logger.info("="*80)

    def infer_single(self, clinical_note: str) -> Tuple[str, int, float]:
        """
        Run inference on single clinical note using TRUE 2-step multi-turn reasoning.

        Uses streamlined 2-step clinical reasoning process that MATCHES TRAINING:
        1. Clinical Assessment: Identify key evidence and apply diagnostic criteria
        2. Final Classification: Make final malnutrition status determination with JSON

        CRITICAL: Uses multi-turn conversation where Step 2 sees Step 1's response
        (matching training format, NOT independent calls).

        IMPORTANT: Each call creates a FRESH conversation with NO history from
        previous calls. This ensures clean state between different patient samples.

        Args:
            clinical_note: Clinical text

        Returns:
            Tuple of (combined_response, predicted_label, confidence)
        """
        print("\n" + "="*80)
        print("2-STEP MULTI-TURN REASONING PROCESS (Training-Matched)")
        print("="*80)

        # Define questions for both turns (SHORTENED FOR SPEED - Only 2 key aspects)
        assessment_question = (
            "Assess for malnutrition. Focus on: (1) Growth/anthropometric data, (2) Clinical signs. "
            "Then state if diagnostic criteria are met."
        )

        classification_question = (
            "Provide classification as JSON: "
            '{"malnutrition_status": "Malnutrition Present"} or '
            '{"malnutrition_status": "Malnutrition Absent"}'
        )

        questions = [assessment_question, classification_question]
        output_formats = [OutputFormat.TEXT, OutputFormat.JSON]

        print("\nSTEP 1 - Clinical Assessment:")
        print(f"Q: {assessment_question}\n")
        print("STEP 2 - Final Classification:")
        print(f"Q: {classification_question}\n")
        print("=" * 80)
        print("Running MULTI-TURN inference (Step 2 sees Step 1's answer)...\n")

        # Use multi-turn inference (matches training!)
        responses = self.inference_pipeline.infer_multi_turn(
            clinical_note=clinical_note,
            questions=questions,
            output_formats=output_formats,
            return_full_response=False
        )

        assessment_response = responses[0]
        classification_response = responses[1]

        print(f"STEP 1 RESPONSE:\n{assessment_response}\n")
        print(f"STEP 2 RESPONSE:\n{classification_response}\n")

        # Try to extract JSON from classification response
        json_response = self._extract_json_response(str(classification_response))
        print("JSON OUTPUT:")
        print(json.dumps(json_response, indent=2))
        print("="*80 + "\n")

        # Parse classification from JSON (fallback to regex if needed)
        predicted_label, confidence = self._parse_from_json_or_fallback(
            json_response, str(classification_response)
        )

        # If JSON parsing fails, try parsing from assessment response
        if confidence < 0.7:
            full_text = f"{assessment_response} {classification_response}"
            predicted_label, confidence = self._parse_classification(full_text)

        # Combine responses for logging
        combined_response = (
            f"[ASSESSMENT]\n{assessment_response}\n\n"
            f"[CLASSIFICATION]\n{classification_response}"
        )

        return combined_response, predicted_label, confidence

    def _extract_json_response(self, response: str) -> Dict:
        """
        Extract JSON from response with robust handling.

        CRITICAL FIX: Handles multiple JSON objects, single quotes, and embedded examples.
        Takes the LAST valid JSON object (most likely to be the actual answer).
        """
        # Normalize: Replace single quotes with double quotes for Python dict-style JSON
        response_normalized = response.replace("'", '"')

        # Find ALL JSON blocks (non-greedy)
        # Use non-greedy .*? to avoid matching across multiple objects
        json_matches = re.findall(r'\{[^{}]*?"malnutrition_status"[^{}]*?\}', response_normalized, re.DOTALL)

        if json_matches:
            # Try parsing from LAST match (most likely to be the actual answer, not an example)
            for json_str in reversed(json_matches):
                try:
                    parsed = json.loads(json_str)
                    # Validate it has the expected key
                    if "malnutrition_status" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue

            # If all fail, try first match
            for json_str in json_matches:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        # Fallback: return default
        logger.warning(f"Could not extract valid JSON from response: {response[:200]}")
        return {"malnutrition_status": "Malnutrition Absent"}

    def _parse_from_json_or_fallback(self, json_obj: Dict, raw_response: str) -> Tuple[int, float]:
        """Parse label from JSON, fallback to regex if invalid."""
        status = json_obj.get("malnutrition_status", "").strip().lower()

        if "present" in status:
            return 1, 0.95
        elif "absent" in status:
            return 0, 0.95
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
        """Evaluate model on test dataset using TRUE 2-step multi-turn reasoning."""
        logger.info("="*80)
        logger.info("EVALUATION STARTING (2-STEP MULTI-TURN REASONING)")
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
            "output_mode": "2-Step Multi-Turn Reasoning + JSON",
            "reasoning_steps": 2,
            "reasoning_process": "Clinical Assessment → Final Classification (with context)",
            "training_inference_match": "True (multi-turn conversations)",
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
            f.write("MALNUTRITION MODEL EVALUATION SUMMARY (2-STEP MULTI-TURN)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {results['model_path']}\n")
            f.write(f"Model Type: {results['model_type']}\n")
            f.write(f"Test Data: {results['test_csv']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Temperature: {results['temperature']}\n")
            f.write(f"Output Mode: {results['output_mode']}\n")
            f.write(f"Reasoning Steps: {results['reasoning_steps']}\n")
            f.write(f"Reasoning Process: {results['reasoning_process']}\n")
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
        print("EVALUATION RESULTS (2-STEP MULTI-TURN REASONING)")
        print("="*80)
        print(f"\nModel: {results['model_path']}")
        print(f"Test examples: {results['valid_examples']}")
        print(f"Output Mode: {results['output_mode']}")
        print(f"Reasoning: {results['reasoning_process']}")
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
        description="Evaluate trained malnutrition model on test set (True Multi-Turn v1.4.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluate_malnutrition.py \\
    --model ./malnutrition_models_v1_1/llama/merged_llama_20241125_143022 \\
    --csv test_data.csv \\
    --output ./eval_results

TRUE Multi-Turn 2-Step Reasoning Process (Matches Training!):
  Turn 1: Clinical Assessment + Clinical Note → Assessment Response
  Turn 2: Classification Question (with Turn 1 context) → JSON Output

CRITICAL FIX: Step 2 now sees Step 1's answer (multi-turn conversation),
matching the training format. No more independent single-turn calls!
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
    logger.info("MALNUTRITION MODEL EVALUATION v1.4.0 (TRUE MULTI-TURN)")
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
        logger.info(f"  - predictions.csv: Includes 2-step multi-turn responses and json_output")
        logger.info(f"  - metrics.json: Comprehensive metrics with reasoning metadata")
        logger.info(f"  - evaluation_summary.txt: Human-readable report with reasoning details")
        logger.info("")
        logger.info("True Multi-Turn 2-Step Reasoning Summary:")
        logger.info(f"  Turn 1: Clinical Assessment (with clinical note)")
        logger.info(f"  Turn 2: Final Classification (with Turn 1 context) → JSON")
        logger.info(f"CRITICAL: Matches training format - Turn 2 sees Turn 1's response!")
        logger.info(f"This approach provides efficient, contextual clinical decision-making.")

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
