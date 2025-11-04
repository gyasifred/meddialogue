"""
Command-Line Interface for MedDialogue
======================================

Provides CLI commands for training and inference.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import argparse
import logging
import sys
from pathlib import Path

from .core import MedDialogue
from .config import TaskConfig, SafetyConfig, TrainingConfig, DataMultiplicationConfig, OutputFormat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def train_command():
    """CLI command for training models."""
    parser = argparse.ArgumentParser(
        description="Train a MedDialogue model (v1.0.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  meddialogue-train \\
    --task malnutrition_assessment \\
    --csv data.csv \\
    --model llama \\
    --epochs 3 \\
    --output ./models/malnutrition \\
    --validation_split 0.2 \\
    --logical_style_ratio 0.4
    
Note: v1.0.0 uses 1:1 row-to-example mapping (no artificial multiplication)
      16 question styles (7 grammatical + 9 logical reasoning)
        """
    )
    
    # Required arguments
    parser.add_argument("--task", required=True, help="Task name (e.g., malnutrition_assessment)")
    parser.add_argument("--csv", required=True, help="Path to training CSV file")
    
    # Model arguments
    parser.add_argument("--model", default="llama", choices=["llama", "phi-4", "mistral", "qwen"],
                       help="Model type (default: llama)")
    parser.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--max_steps", type=int, help="Max training steps (overrides epochs)")
    
    # Data preparation arguments (v1.0.0)
    parser.add_argument("--single_turn_ratio", type=float, default=0.5, 
                       help="Single-turn conversation ratio (default: 0.5)")
    parser.add_argument("--validation_split", type=float, default=0.0, 
                       help="Validation split ratio (default: 0.0 = no validation)")
    parser.add_argument("--logical_style_ratio", type=float, default=0.4, 
                       help="Logical style ratio (default: 0.4 = 40%% logical, 60%% grammatical)")
    parser.add_argument("--typo_ratio", type=float, default=0.15, 
                       help="Typo ratio for robustness (default: 0.15)")
    parser.add_argument("--max_multi_turns", type=int, default=10, 
                       help="Max multi-turn conversation turns (default: 10)")
    
    # Safety arguments
    parser.add_argument("--disable_safety", action="store_true", help="Disable safety checks")
    parser.add_argument("--disable_pii", action="store_true", help="Disable PII detection")
    parser.add_argument("--disable_bias", action="store_true", help="Disable bias monitoring")
    
    # Other arguments
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device (default: 0)")
    parser.add_argument("--quick_test", action="store_true", help="Quick test with subsampled data")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.csv).exists():
        logger.error(f"CSV file not found: {args.csv}")
        sys.exit(1)
    
    # Validate ratios
    if not 0.0 <= args.single_turn_ratio <= 1.0:
        logger.error("single_turn_ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not 0.0 <= args.validation_split < 1.0:
        logger.error("validation_split must be between 0.0 and 1.0 (exclusive)")
        sys.exit(1)
    
    if not 0.0 <= args.logical_style_ratio <= 1.0:
        logger.error("logical_style_ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not 0.0 <= args.typo_ratio <= 1.0:
        logger.error("typo_ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    logger.info("="*80)
    logger.info("MEDDIALOGUE TRAINING v1.0.0")
    logger.info("="*80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model}")
    logger.info(f"CSV: {args.csv}")
    logger.info(f"Output: {args.output}")
    logger.info("-"*80)
    logger.info("Data Configuration:")
    logger.info(f"  Single-turn ratio: {args.single_turn_ratio * 100:.0f}%")
    logger.info(f"  Validation split: {args.validation_split * 100:.0f}%")
    logger.info(f"  Logical style ratio: {args.logical_style_ratio * 100:.0f}%")
    logger.info(f"  Max multi-turns: {args.max_multi_turns}")
    logger.info("="*80)
    
    # Create configurations
    task_config = TaskConfig(
        task_name=args.task,
        task_description=f"{args.task} clinical assessment"
    )
    
    safety_config = SafetyConfig(
        enable_pii_detection=not args.disable_pii,
        enable_bias_monitoring=not args.disable_bias,
        block_on_safety_failure=False
    )
    
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        use_full_dataset=not args.quick_test
    )
    
    mult_config = DataMultiplicationConfig(
        single_turn_ratio=args.single_turn_ratio,
        validation_split=args.validation_split,
        logical_style_ratio=args.logical_style_ratio,
        typo_ratio=args.typo_ratio,
        max_multi_turns=args.max_multi_turns
    )
    
    # Initialize MedDialogue
    meddialogue = MedDialogue(
        task_config=task_config,
        model_type=args.model,
        safety_config=safety_config,
        training_config=training_config,
        mult_config=mult_config,
        output_dir=args.output,
        enable_safety=not args.disable_safety,
        cuda_device=args.cuda_device
    )
    
    # Train
    try:
        results = meddialogue.train_from_csv(args.csv)
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)
        logger.info(f"Model saved to: {results['save_paths']}")
        logger.info(f"Train examples: {results['num_train_examples']}")
        logger.info(f"Has validation: {results['has_validation']}")
        logger.info(f"Training time: {results['training_time_minutes']:.2f} minutes")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def infer_command():
    """CLI command for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained MedDialogue model (v1.0.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single inference
  meddialogue-infer --model ./models/malnutrition/merged_llama_20250101_120000 \\
    --task malnutrition_assessment \\
    --text "Patient with BMI < 5th percentile..."
  
  # Batch inference from CSV
  meddialogue-infer --model ./models/malnutrition/merged_llama_20250101_120000 \\
    --task malnutrition_assessment \\
    --csv input.csv --output output.csv
  
  # Interactive mode
  meddialogue-infer --model ./models/malnutrition/merged_llama_20250101_120000 \\
    --task malnutrition_assessment \\
    --interactive
        """
    )
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--task", required=True, help="Task name")
    
    # Input arguments (one of these required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Clinical text for single inference")
    group.add_argument("--csv", help="CSV file for batch inference")
    group.add_argument("--interactive", action="store_true", help="Start interactive session")
    
    # Output arguments
    parser.add_argument("--output", help="Output CSV path (for batch inference)")
    parser.add_argument("--text_column", default="clinical_note", help="Text column name (default: clinical_note)")
    parser.add_argument("--format", default="text", choices=["text", "json", "xml", "markdown"],
                       help="Output format (default: text)")
    parser.add_argument("--question", help="Custom question for inference")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model).exists():
        logger.error(f"Model directory not found: {args.model}")
        sys.exit(1)
    
    if args.csv and not Path(args.csv).exists():
        logger.error(f"CSV file not found: {args.csv}")
        sys.exit(1)
    
    logger.info("="*80)
    logger.info("MEDDIALOGUE INFERENCE v1.0.0")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task}")
    logger.info("="*80)
    
    # Create task config
    task_config = TaskConfig(
        task_name=args.task,
        task_description=f"{args.task} clinical assessment"
    )
    
    # Initialize MedDialogue
    meddialogue = MedDialogue(task_config=task_config)
    meddialogue.load_trained_model(args.model)
    
    try:
        # Single inference
        if args.text:
            logger.info("Running single inference...")
            response = meddialogue.infer(
                clinical_note=args.text,
                question=args.question,
                format=args.format
            )
            
            print("\n" + "="*80)
            print("RESPONSE:")
            print("="*80)
            if isinstance(response, dict):
                import json
                print(json.dumps(response, indent=2))
            else:
                print(response)
            print("="*80 + "\n")
        
        # Batch inference
        elif args.csv:
            if not args.output:
                logger.error("--output required for batch inference")
                sys.exit(1)
            
            logger.info(f"Running batch inference on {args.csv}...")
            
            import pandas as pd
            from .inference import BatchInference
            
            df = pd.read_csv(args.csv)
            batch_inference = BatchInference(meddialogue.inference_pipeline)
            
            df = batch_inference.process_dataframe(
                df,
                text_column=args.text_column,
                output_format=OutputFormat(args.format)
            )
            
            df.to_csv(args.output, index=False)
            logger.info(f"Results saved to: {args.output}")
        
        # Interactive mode
        elif args.interactive:
            logger.info("Starting interactive session...")
            logger.info("Type 'exit' or 'quit' to end the session")
            logger.info("-"*80)
            
            while True:
                try:
                    clinical_note = input("\nEnter clinical note (or 'exit'): ")
                    if clinical_note.lower() in ['exit', 'quit']:
                        break
                    
                    response = meddialogue.infer(
                        clinical_note=clinical_note,
                        question=args.question,
                        format=args.format
                    )
                    
                    print("\n" + "-"*80)
                    print("RESPONSE:")
                    print("-"*80)
                    if isinstance(response, dict):
                        import json
                        print(json.dumps(response, indent=2))
                    else:
                        print(response)
                    print("-"*80)
                    
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
                except Exception as e:
                    logger.error(f"Inference error: {e}")
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # For direct execution
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_command()
    elif len(sys.argv) > 1 and sys.argv[1] == "infer":
        infer_command()
    else:
        print("Usage: python -m meddialogue.cli [train|infer] [options]")
        sys.exit(1)