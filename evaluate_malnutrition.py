#!/usr/bin/env python3
"""
Malnutrition Model Evaluation Script - v1.2.0 (Multi-Step Reasoning)
====================================================================
Evaluates trained MedDialogue malnutrition models on test datasets.
Uses multi-step clinical reasoning:
  1. Evidence Gathering: Anthropometrics, symptoms, intake data
  2. Clinical Reasoning: Apply diagnostic criteria and explain rationale
  3. Final Classification: JSON output with malnutrition status

Forces JSON output: {"malnutrition_status": "Malnutrition Present/Absent"}
Prints all reasoning steps to terminal for transparency.

Required CSV columns:
  - txt: Clinical note text
  - DEID: Patient identifier (for tracking)
  - label: Binary label (1=Malnutrition Present, 0=Absent)

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.2.0
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
    Evaluator for malnutrition classification models with multi-step reasoning.

    Handles:
    - Model loading (adapter or merged)
    - Multi-step clinical reasoning (3 questions)
    - Batch inference with progress tracking
    - JSON response enforcement
    - Comprehensive metrics calculation
    - Results saving (CSV, JSON, plots)
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

        logger.info(f"Evaluator initialized for {model_type} (Multi-Step Reasoning v1.2)")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Max tokens: {max_new_tokens}")
        logger.info(f"Reasoning: 3-step process (Evidence → Diagnosis → Classification)")

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
        logger.info("Inference pipeline initialized for multi-step reasoning")
        logger.info("="*80)

    def infer_single(self, clinical_note: str) -> Tuple[str, int, float]:
        """
        Run inference on single clinical note using multi-step reasoning.

        Uses 3-step clinical reasoning process:
        1. Evidence Gathering: Identify key anthropometric, clinical, and intake findings
        2. Diagnosis & Reasoning: Synthesize evidence and explain clinical reasoning
        3. Final Classification: Make final malnutrition status determination with JSON

        Args:
            clinical_note: Clinical text

        Returns:
            Tuple of (combined_response, predicted_label, confidence)
        """
        print("\n" + "="*80)
        print("MULTI-STEP REASONING PROCESS")
        print("="*80)

        # STEP 1: Evidence Gathering
        evidence_question = (
            "What are the key pieces of evidence for assessing malnutrition in this patient? "
            "Specifically identify: (1) Anthropometric data (weight, BMI, percentiles, z-scores, trends), "
            "(2) Clinical symptoms and physical exam findings, and (3) Nutritional intake patterns. "
            "Be concise and focus on the most relevant findings."
        )
        print(f"\nSTEP 1 - Evidence Gathering:")
        print(f"Q: {evidence_question}")

        evidence_response = self.inference_pipeline.infer(
            clinical_note=clinical_note,
            question=evidence_question,
            output_format=OutputFormat.TEXT,
            return_full_response=False
        )
        print(f"A: {evidence_response}\n")

        # STEP 2: Diagnosis and Clinical Reasoning
        reasoning_question = (
            "Based on the evidence you identified, what is your diagnostic assessment? "
            "Apply clinical criteria (ASPEN, WHO, or AND guidelines) and explain your reasoning: "
            "Does this patient meet diagnostic criteria for malnutrition? If so, what severity? "
            "What is your clinical rationale? Be specific about which criteria are met or not met."
        )
        print(f"STEP 2 - Diagnosis & Clinical Reasoning:")
        print(f"Q: {reasoning_question}")

        reasoning_response = self.inference_pipeline.infer(
            clinical_note=clinical_note,
            question=reasoning_question,
            output_format=OutputFormat.TEXT,
            return_full_response=False
        )
        print(f"A: {reasoning_response}\n")

        # STEP 3: Final Classification with JSON
        final_question = (
            "Based on your assessment and reasoning, is malnutrition present or absent in this patient? "
            "Answer in strict JSON format ONLY: "
            "{\"malnutrition_status\": \"Malnutrition Present\"} or "
            "{\"malnutrition_status\": \"Malnutrition Absent\"}. "
            "Provide ONLY the JSON object, no additional text."
        )
        print(f"STEP 3 - Final Classification:")
        print(f"Q: {final_question}")

        final_response = self.inference_pipeline.infer(
            clinical_note=clinical_note,
            question=final_question,
            output_format=OutputFormat.JSON,
            return_full_response=False
        )
        print(f"A: {final_response}\n")

        # Try to extract JSON from final response
        json_response = self._extract_json_response(str(final_response))
        print("JSON OUTPUT:")
        print(json.dumps(json_response, indent=2))
        print("="*80 + "\n")

        # Parse classification from JSON (fallback to regex on all responses)
        predicted_label, confidence = self._parse_from_json_or_fallback(
            json_response, str(final_response)
        )

        # If JSON parsing fails, try reasoning from the full conversation
        if confidence < 0.7:
            full_reasoning = f"{evidence_response} {reasoning_response} {final_response}"
            predicted_label, confidence = self._parse_classification(full_reasoning)

        # Combine all responses for logging
        combined_response = (
            f"[EVIDENCE]\n{evidence_response}\n\n"
            f"[REASONING]\n{reasoning_response}\n\n"
            f"[CLASSIFICATION]\n{final_response}"
        )

        return combined_response, predicted_label, confidence

    def _extract_json_response(self, response: str) -> Dict:
        """Extract JSON from response using regex (robust to extra text)."""
        # Find JSON block
        json_match = re.search(r'\{.*"malnutrition_status".*?\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return default
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
        """Evaluate model on test dataset using multi-step reasoning."""
        logger.info("="*80)
        logger.info("EVALUATION STARTING (MULTI-STEP REASONING)")
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
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            try:
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
            "output_mode": "Multi-Step Reasoning + JSON",
            "reasoning_steps": 3,
            "reasoning_process": "Evidence Gathering → Diagnosis & Reasoning → Final Classification",
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
            f.write("MALNUTRITION MODEL EVALUATION SUMMARY (MULTI-STEP REASONING)\n")
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
        print("EVALUATION RESULTS (MULTI-STEP REASONING)")
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
        description="Evaluate trained malnutrition model on test set (Multi-Step Reasoning v1.2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluate_malnutrition.py \\
    --model ./malnutrition_models_v1_1/llama/merged_llama_20241125_143022 \\
    --csv test_data.csv \\
    --output ./eval_results

Multi-Step Reasoning Process:
  Step 1: Evidence Gathering - Identify key anthropometric, clinical, and intake findings
  Step 2: Diagnosis & Reasoning - Apply clinical criteria and explain reasoning
  Step 3: Final Classification - Make final determination with JSON output

This approach mirrors clinical decision-making: Gather → Reason → Classify
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
    logger.info("MALNUTRITION MODEL EVALUATION v1.2.0 (MULTI-STEP REASONING)")
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
        logger.info(f"  - predictions.csv: Includes multi-step reasoning responses and json_output")
        logger.info(f"  - metrics.json: Comprehensive metrics with reasoning metadata")
        logger.info(f"  - evaluation_summary.txt: Human-readable report with reasoning details")
        logger.info("")
        logger.info("Multi-Step Reasoning Summary:")
        logger.info(f"  Step 1: Evidence Gathering")
        logger.info(f"  Step 2: Clinical Diagnosis & Reasoning")
        logger.info(f"  Step 3: Final Classification (JSON)")
        logger.info(f"This approach provides transparent, explainable clinical decision-making.")

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
