#!/usr/bin/env python3
"""
Malnutrition Model Evaluation Script - v4.0.0 (Multi-Turn Conversation)
==========================================================================
Evaluates trained MedDialogue malnutrition models using multi-turn conversation
pattern similar to gradio_chat_v1.py.

Key Features:
  - Multi-turn conversation per patient (maintains history within sample)
  - Clean conversation reset between patients (no history leakage)
  - Clinical note included ONLY in first message
  - Questions asked in training order
  - Last question is malnutrition status
  - Only model responses returned (no clinical text spillage)
  - Comprehensive classification metrics

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 4.0.0
"""

import os
import sys
import logging
import argparse
import json
import re
import gc
import torch
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    matthews_corrcoef, roc_auc_score
)

# Unsloth imports
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
except ImportError:
    print("Error: unsloth not installed. Install with: pip install unsloth")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =================================================================
# Evaluation Questions (from training templates - Priority Order)
# =================================================================
# Priority 1: case_presentation, growth_and_anthropometrics, physical_exam, labs_and_screening
# Priority 2: diagnosis_and_reasoning
# Priority 3: malnutrition_status (MUST BE LAST)

EVALUATION_QUESTIONS = [
    # Priority 1 - Question 1: Case Presentation
    "Document the clinical presentation including chief concern, temporal context, and assessment type (single-point vs serial/longitudinal). What additional history would strengthen the assessment?",

    # Priority 1 - Question 2: Growth & Anthropometrics
    "What are ALL anthropometric measurements with DATES? Calculate trends and trajectories.",

    # Priority 1 - Question 3: Physical Exam
    "Document nutrition-focused physical exam findings: muscle mass, subcutaneous fat, edema, micronutrient deficiency signs. Do findings meet ASPEN criteria?",

    # Priority 1 - Question 4: Labs & Screening
    "Extract ALL nutrition-relevant laboratory values with dates: visceral proteins, CBC, micronutrients, inflammatory markers. Calculate trends and identify deficiencies.",

    # Priority 2 - Question 5: Diagnosis & Reasoning
    "What's your diagnosis with complete clinical reasoning? Synthesize ALL evidence (case presentation, anthropometrics, physical exam, labs) temporally.",

    # Priority 3 - Question 6: Malnutrition Status (FINAL)
    "Is malnutrition present or absent? State 'Malnutrition Present' or 'Malnutrition Absent'."
]


# =================================================================
# Multi-Turn Conversation Evaluator
# =================================================================

class MalnutritionEvaluator:
    """
    Evaluator using multi-turn conversation pattern.

    Key behaviors:
    - Maintains conversation history WITHIN each patient evaluation
    - Resets conversation history BETWEEN patients (no leakage)
    - Clinical note included ONLY in first message
    - Processes questions in order, last one is malnutrition status
    """

    # Patterns to extract malnutrition status from final response
    PRESENT_PATTERNS = [
        r'\bmalnutrition\s+(is\s+)?present\b',
        r'\byes\b.*(malnutrition|malnourished)\b',
        r'\bmalnutrition\s+confirmed\b',
        r'(?<!\bnot\s)(?<!\bno\s)\b(is\s+)?malnourished\b',  # malnourished but NOT preceded by "not" or "no"
        r'\bdiagnosis:?\s+malnutrition\b',
        r'\bhas\s+malnutrition\b',
        r'\bpositive\s+for\s+malnutrition\b',
        r'\bpatient\s+(is\s+)?malnourished\b',
        r'\bchild\s+(is\s+)?malnourished\b'
    ]

    ABSENT_PATTERNS = [
        r'\bmalnutrition\s+(is\s+)?absent\b',
        r'\bno\b.*(malnutrition|malnourished)\b',
        r'\b(is\s+)?not\s+malnourished\b',
        r'\b(is\s+)?not\s+malnutrition\b',
        r'\bno\s+evidence\s+of\s+malnutrition\b',
        r'\bdoes\s+not\s+meet\s+criteria\b',
        r'\bwell[\s-]nourished\b',
        r'\bnormal\s+nutritional\s+status\b',
        r'\bno\s*[-–—]\s*.*(malnutrition|malnourished)\b'
    ]

    def __init__(
        self,
        model_path: str,
        model_type: str = "llama",
        chat_template: str = None,
        temperature: float = 0.1,
        max_seq_length: int = 4096,
        max_new_tokens: int = 2048,
        cuda_device: Optional[int] = None
    ):
        """Initialize evaluator with model configuration."""
        self.model_path = model_path
        self.model_type = model_type
        self.chat_template = chat_template
        self.temperature = temperature
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.cuda_device = cuda_device

        self.model = None
        self.tokenizer = None
        self.device = self._get_device()

        # Conversation history for current patient (reset between patients)
        self.current_conversation: List[Dict[str, str]] = []
        self.clinical_note_included = False

        logger.info(f"Evaluator initialized: {model_type}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Device: {self.device}")

    def _get_device(self) -> torch.device:
        """Get optimal CUDA device."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available - using CPU")
            return torch.device("cpu")

        if self.cuda_device is not None:
            device = torch.device(f"cuda:{self.cuda_device}")
            logger.info(f"Using specified GPU {self.cuda_device}")
            return device

        device = torch.device("cuda:0")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    def load_model(self):
        """Load model and tokenizer."""
        logger.info("="*80)
        logger.info("Loading model...")
        logger.info("="*80)

        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Load model with Unsloth
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
                device_map={"": self.device.index if self.device.type == 'cuda' else 'cpu'},
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN")
            )

            # Set chat template
            if self.chat_template:
                template = self.chat_template
            else:
                template_map = {
                    "llama": "llama-3.1",
                    "phi": "phi-3",
                    "phi-4": "phi-4",
                    "mistral": "mistral",
                    "qwen": "qwen2.5"
                }
                template = template_map.get(self.model_type, "llama-3.1")

            self.tokenizer = get_chat_template(
                tokenizer=self.tokenizer,
                chat_template=template
            )

            # Set to inference mode
            FastLanguageModel.for_inference(self.model)

            logger.info("✓ Model loaded successfully")

            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"GPU Memory: {allocated:.2f}GB allocated")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {e}")

    def reset_conversation(self):
        """Reset conversation history for new patient."""
        self.current_conversation = []
        self.clinical_note_included = False
        logger.debug("Conversation reset")

    def generate_response(self, user_message: str, clinical_note: str = None) -> str:
        """
        Generate response to user message.

        Clinical note is included ONLY in first message if provided.
        Maintains full conversation history for multi-turn context.
        """
        # Include clinical note in FIRST message only
        actual_message = user_message
        if clinical_note and not self.clinical_note_included:
            actual_message = f"CLINICAL NOTE:\n{clinical_note.strip()}\n\n{user_message}"
            self.clinical_note_included = True
            logger.debug("Clinical note included in first message")

        # Add user message to conversation history
        self.current_conversation.append({"role": "user", "content": actual_message})

        try:
            # Apply chat template to full conversation history
            formatted_text = self.tokenizer.apply_chat_template(
                self.current_conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize with truncation
            max_input_length = self.max_seq_length - self.max_new_tokens
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.95,
                    use_cache=True
                )

            # Decode only the new tokens (not the input)
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Clean up tensors
            del inputs
            del outputs

            # Add assistant response to conversation history
            self.current_conversation.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def evaluate_single_patient(
        self,
        clinical_note: str,
        patient_id: str = None
    ) -> Tuple[str, int, float, List[str]]:
        """
        Evaluate single patient using multi-turn conversation.

        Returns:
            Tuple of (final_response, predicted_label, confidence, all_responses)
        """
        # Reset conversation for this patient
        self.reset_conversation()

        all_responses = []

        print(f"\n{'='*80}")
        print(f"Patient: {patient_id} | Note Length: {len(clinical_note)} chars")
        print(f"{'='*80}")

        # Ask questions in order
        for i, question in enumerate(EVALUATION_QUESTIONS, 1):
            print(f"\n{'='*80}")
            print(f"TURN {i}/{len(EVALUATION_QUESTIONS)}")
            print(f"{'='*80}")

            # Show what's being passed
            if i == 1:
                print(f"\n📝 Turn {i} Input:")
                print(f"Clinical note + Question {i}")
                print(f"Question: {question[:100]}...")
            else:
                print(f"\n📝 Turn {i} Input:")
                print(f"Question {i} (with conversation history)")
                print(f"Question: {question[:100]}...")

            # Include clinical note only in first message
            clinical_note_to_pass = clinical_note if i == 1 else None

            response = self.generate_response(question, clinical_note_to_pass)
            all_responses.append(response)

            print(f"✅ Response {i}: {response[:150]}..." if len(response) > 150 else f"✅ Response {i}: {response}")

        # Extract malnutrition status from final response
        final_response = all_responses[-1]
        predicted_label, confidence = self._parse_malnutrition_status(final_response)

        logger.info(f"\nPredicted: {'Present' if predicted_label == 1 else 'Absent'} (confidence: {confidence:.2f})")
        logger.info(f"{'='*80}\n")

        return final_response, predicted_label, confidence, all_responses

    def _parse_malnutrition_status(self, response: str) -> Tuple[int, float]:
        """Parse malnutrition status from response."""
        response_lower = response.lower().strip()

        # Count pattern matches
        present_score = sum(1 for pattern in self.PRESENT_PATTERNS
                          if re.search(pattern, response_lower))
        absent_score = sum(1 for pattern in self.ABSENT_PATTERNS
                         if re.search(pattern, response_lower))

        if present_score > absent_score:
            return 1, min(0.95, 0.5 + 0.1 * present_score)
        elif absent_score > present_score:
            return 0, min(0.95, 0.5 + 0.1 * absent_score)
        else:
            # Fallback: Check for explicit keywords
            if "present" in response_lower and "absent" not in response_lower:
                return 1, 0.85
            elif "absent" in response_lower and "present" not in response_lower:
                return 0, 0.85

            # Check for simple yes/no at start
            response_words = response_lower.split()
            if response_words:
                first_word = response_words[0].rstrip('.,!?;:-')
                if first_word == "yes":
                    return 1, 0.75
                elif first_word == "no":
                    return 0, 0.75

            # Last resort: default to absent with low confidence
            logger.warning(f"Ambiguous response (defaulting to Absent): {response[:100]}")
            return 0, 0.5

    def evaluate_dataset(
        self,
        test_csv: str,
        output_dir: str,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """Evaluate model on test dataset."""
        logger.info("="*80)
        logger.info("EVALUATION STARTING (MULTI-TURN CONVERSATION)")
        logger.info("="*80)
        logger.info(f"Test CSV: {test_csv}")
        logger.info(f"Output dir: {output_dir}")
        logger.info("="*80)

        # Load test data
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

        # Evaluate each patient
        predictions = []
        confidences = []
        final_responses = []
        all_responses_list = []

        logger.info("Running multi-turn evaluation on test set...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            try:
                # CRITICAL: Each patient gets fresh conversation history
                final_response, pred, conf, all_resp = self.evaluate_single_patient(
                    clinical_note=row["txt"],
                    patient_id=str(row["DEID"])
                )

                predictions.append(pred)
                confidences.append(conf)
                final_responses.append(final_response)
                all_responses_list.append(all_resp)

                # Periodic garbage collection
                if (idx + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error on row {idx} (DEID: {row['DEID']}): {e}")
                predictions.append(-1)
                confidences.append(0.0)
                final_responses.append(f"ERROR: {str(e)}")
                all_responses_list.append([])

        # Add results to dataframe
        df["predicted_label"] = predictions
        df["confidence"] = confidences
        df["final_response"] = final_responses
        df["all_responses"] = [json.dumps(resp) for resp in all_responses_list]
        df["correct"] = df["label"] == df["predicted_label"]

        # Filter valid predictions
        valid_df = df[df["predicted_label"] != -1].copy()
        error_count = len(df) - len(valid_df)

        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during evaluation")

        logger.info(f"Completed evaluation on {len(valid_df)} valid examples")

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Final memory cleanup completed")
        logger.info("="*80)

        # Calculate metrics
        metrics = self._calculate_metrics(
            valid_df["label"].tolist(),
            valid_df["predicted_label"].tolist()
        )

        # Prepare results
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
            "evaluation_method": "Multi-turn conversation (gradio_chat pattern)",
            "num_questions": len(EVALUATION_QUESTIONS),
            "questions": EVALUATION_QUESTIONS,
            "metrics": metrics
        }

        # Save results
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

        try:
            auc = roc_auc_score(true_labels, predictions)
        except ValueError:
            auc = None

        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(true_labels, predictions, average=None, zero_division=0)

        return {
            "overall": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "sensitivity": float(recall),
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
            }
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
            f.write("MALNUTRITION MODEL EVALUATION SUMMARY (MULTI-TURN)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {results['model_path']}\n")
            f.write(f"Model Type: {results['model_type']}\n")
            f.write(f"Test Data: {results['test_csv']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Temperature: {results['temperature']}\n")
            f.write(f"Evaluation Method: {results['evaluation_method']}\n")
            f.write(f"Number of Questions: {results['num_questions']}\n")
            f.write("\n" + "-"*80 + "\n\n")
            f.write("QUESTIONS ASKED:\n")
            for i, q in enumerate(results['questions'], 1):
                f.write(f"  {i}. {q}\n")
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
        print("EVALUATION RESULTS (MULTI-TURN CONVERSATION)")
        print("="*80)
        print(f"\nModel: {results['model_path']}")
        print(f"Test examples: {results['valid_examples']}")
        print(f"Method: {results['evaluation_method']}")
        print(f"Questions: {results['num_questions']}")
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


# =================================================================
# Main Application Entry Point
# =================================================================

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate malnutrition model using multi-turn conversation (v4.0.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluate_malnutrition.py \\
    --model ./trained_model \\
    --csv test_data.csv \\
    --output ./eval_results

Multi-Turn Conversation Evaluation:
  - Maintains conversation history within each patient
  - Resets conversation between patients (no leakage)
  - Clinical note included only in first message
  - Questions asked in training order
  - Last question is malnutrition status
        """
    )

    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--csv", required=True, help="Path to test CSV")
    parser.add_argument("--output", default="./evaluation_results", help="Output directory")
    parser.add_argument("--model_type", default="llama", choices=["llama", "phi", "phi-4", "mistral", "qwen"])
    parser.add_argument("--chat_template", help="Custom chat template")
    parser.add_argument("--max_seq_length", type=int, default=4096)
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
    logger.info("MALNUTRITION MODEL EVALUATION v4.0.0 (MULTI-TURN)")
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
        # Initialize evaluator
        evaluator = MalnutritionEvaluator(
            model_path=args.model,
            model_type=args.model_type,
            chat_template=args.chat_template,
            temperature=args.temperature,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_tokens,
            cuda_device=args.cuda_device
        )

        # Load model
        evaluator.load_model()

        # Run evaluation
        results = evaluator.evaluate_dataset(
            test_csv=args.csv,
            output_dir=args.output,
            save_predictions=True
        )

        # Print results
        evaluator.print_results(results)

        logger.info(f"\n✓ Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        logger.info(f"  - predictions.csv: Full predictions with all responses")
        logger.info(f"  - metrics.json: Comprehensive metrics")
        logger.info(f"  - evaluation_summary.txt: Human-readable report")
        logger.info("")
        logger.info("Multi-Turn Conversation Summary:")
        logger.info(f"  - {len(EVALUATION_QUESTIONS)} questions per patient")
        logger.info(f"  - Conversation history maintained within patient")
        logger.info(f"  - Clean reset between patients (no leakage)")
        logger.info(f"  - Clinical note included only in first message")

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
