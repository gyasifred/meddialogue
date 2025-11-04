"""
Core MedDialogue Interface
==========================

Main high-level interface for the MedDialogue framework.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import os
import logging
import pandas as pd
from typing import Optional, Dict, Any, Union, List
from datetime import datetime

from .config import (
    TaskConfig, SafetyConfig, TrainingConfig, 
    ModelConfig, ConversationConfig, OutputFormat
)
from .data_prep import DataPrep
from .safety import SafetyChecker
from .train import Trainer
from .inference import InferencePipeline
from .models import ModelRegistry, get_device, optimize_memory

logger = logging.getLogger(__name__)


class MedDialogue:
    """
    Main interface for MedDialogue framework.
    
    Version 1.0.0 features:
    - True 1:1 mapping: Each clinical note → ONE training example
    - 16 question combination styles (7 grammatical + 9 logical reasoning)
    - Intelligent field ordering for reasoning questions
    - Balanced single-turn and multi-turn conversations
    - Optional validation split (set to 0.0 to use all data for training)
    - Optimized for modern LLMs (16K-32K token context)
    - Enhanced safety and validation
    - Comprehensive logging and statistics
    
    Example:
        >>> from meddialogue import MedDialogue, TaskConfig, ConversationConfig
        >>> 
        >>> # Define task with field ordering for reasoning
        >>> task_config = TaskConfig(
        ...     task_name="malnutrition_assessment",
        ...     output_fields=["growth_and_anthropometrics", "diagnosis_and_reasoning", "care_plan"],
        ...     field_ordering={
        ...         "growth_and_anthropometrics": 1,  # Assessment first
        ...         "diagnosis_and_reasoning": 2,      # Diagnosis second
        ...         "care_plan": 3                     # Action third
        ...     },
        ...     input_field="clinical_note"
        ... )
        >>> 
        >>> # Configure data preparation (v1.1.0)
        >>> conversation_config = ConversationConfig(
        ...     single_turn_ratio=0.7,          # 70% single-turn, 30% multi-turn
        ...     max_multi_turns=3,               # Up to 4 conversation turns
        ...     include_typos=True,
        ...     typo_ratio=0.15,
        ...     validation_split=0.0,            # Use all data for training
        ...     logical_style_ratio=0.4,         # 40% logical, 60% grammatical styles
        ...     context_window_size=64000        # 16K tokens
        ... )
        >>> 
        >>> # Initialize framework
        >>> trainer = MedDialogue(
        ...     task_config=task_config,
        ...     conversation_config=conversation_config,
        ...     model_type="llama"
        ... )
        >>> 
        >>> # Train (10 rows → 10 examples)
        >>> results = trainer.train_from_csv("data.csv", epochs=3)
        >>> 
        >>> # Inference
        >>> response = trainer.infer(
        ...     clinical_note="Patient presents with...",
        ...     format="json"
        ... )
    """
    
    def __init__(
        self,
        task_config: TaskConfig,
        model_type: str = "llama",
        safety_config: Optional[SafetyConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        conversation_config: Optional[ConversationConfig] = None,
        output_dir: str = "./output",
        enable_safety: bool = True,
        cuda_device: int = 0,
        verbose: bool = True
    ):
        """
        Initialize MedDialogue framework.
        
        Args:
            task_config: Task configuration (required)
            model_type: Model type from ["llama", "phi-4", "mistral", "qwen"]
            safety_config: Safety configuration (optional, uses defaults if None)
            training_config: Training configuration (optional, uses defaults if None)
            conversation_config: Data multiplication configuration (optional, uses defaults if None)
            output_dir: Output directory for models and logs
            enable_safety: Enable safety checks (PII detection, bias monitoring)
            cuda_device: CUDA device index (default: 0)
            verbose: Enable verbose logging (default: True)
        """
        self.task_config = task_config
        self.model_type = model_type
        self.safety_config = safety_config or SafetyConfig()
        self.training_config = training_config or TrainingConfig()
        self.conversation_config = conversation_config or ConversationConfig()
        self.output_dir = output_dir
        self.enable_safety = enable_safety
        self.cuda_device = cuda_device
        self.verbose = verbose
        
        # Robust logging setup
        if verbose:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()],
                force=True
            )
        else:
            logging.getLogger().setLevel(logging.CRITICAL)

        # Validate model type
        try:
            model_info = ModelRegistry.get_config(model_type)
        except KeyError:
            available = ", ".join(ModelRegistry.list_available_models())
            raise ValueError(
                f"Invalid model_type '{model_type}'. "
                f"Available models: {available}"
            )
        
        # Get model configuration
        self.model_config = ModelConfig(
            model_name=model_info["model_name"],
            model_type=model_type,
            chat_template=model_info["chat_template"],
            max_seq_length=model_info["max_seq_length"]
        )
        
        # Initialize components
        self.data_prep = DataPrep(task_config, conversation_config)
        self.safety_checker = SafetyChecker(self.safety_config, task_config) if enable_safety else None
        self.trainer = None
        self.inference_pipeline = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Log initialization
        logger.info("=" * 80)
        logger.info(f"MedDialogue v1.0.0 initialized")
        logger.info("=" * 80)
        logger.info(f"Task: {task_config.task_name}")
        logger.info(f"Model: {model_type} ({self.model_config.model_name})")
        logger.info(f"Output fields: {', '.join(task_config.output_fields)}")
        logger.info(f"Safety enabled: {enable_safety}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("-" * 80)
        logger.info("Data Preparation Settings (v1.0.0):")
        logger.info(f"  Single-turn ratio: {conversation_config.single_turn_ratio * 100:.0f}%")
        logger.info(f"  Multi-turn ratio: {(1 - conversation_config.single_turn_ratio) * 100:.0f}%")
        logger.info(f"  Max conversation turns: {conversation_config.max_multi_turns + 1}")
        logger.info(f"  Logical style ratio: {conversation_config.logical_style_ratio * 100:.0f}%")
        logger.info(f"  Context window: {conversation_config.context_window_size} chars (~{conversation_config.context_window_size // 4000}K tokens)")
        logger.info(f"  Question length: {conversation_config.min_question_length}-{conversation_config.max_question_length} chars")
        logger.info(f"  Typos enabled: {conversation_config.include_typos} ({conversation_config.typo_ratio * 100:.0f}% if enabled)")
        
        # Validation split messaging
        if conversation_config.validation_split > 0:
            logger.info(f"  Validation split: {conversation_config.validation_split * 100:.0f}% (validation enabled)")
        else:
            logger.info(f"  Validation split: DISABLED (all data used for training)")
        
        logger.info("-" * 80)
        logger.info("  • 16 question combination styles (7 grammatical + 9 logical reasoning)")
        logger.info("  • Intelligent field ordering for reasoning questions")
        logger.info("  • Truly optional validation split")
        logger.info("-" * 80)
        logger.info("NOTE: 1:1 mapping - Each row generates ONE example")
        logger.info("      No artificial multiplication. 10 rows = 10 examples.")
        logger.info("=" * 80)
        
    def prepare_data(
        self,
        data: pd.DataFrame,
        run_safety_checks: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare training data with safety checks.
        
        Args:
            data: Input DataFrame with clinical notes and output fields
            run_safety_checks: Run safety checks (PII detection, bias monitoring)
            
        Returns:
            Dictionary containing:
                - train_examples: List of training examples
                - val_examples: List of validation examples (empty if validation_split=0)
                - safety_results: Safety check results (if enabled)
                - num_train: Number of training examples
                - num_val: Number of validation examples
                - has_validation: Boolean indicating if validation set exists
                - single_turn_count: Number of single-turn examples
                - multi_turn_count: Number of multi-turn examples
                - statistics: Detailed statistics
                
        Raises:
            ValueError: If safety checks fail and block_on_safety_failure=True
        """
        optimize_memory(aggressive=True)
        
        logger.info("=" * 80)
        logger.info("Preparing training data...")
        logger.info("=" * 80)
        logger.info(f"Input data: {len(data)} rows")
        logger.info(f"Expected output: ~{len(data)} examples (1:1 mapping)")
        
        if self.conversation_config.validation_split > 0:
            expected_train = int(len(data) * (1 - self.conversation_config.validation_split))
            expected_val = len(data) - expected_train
            logger.info(f"Expected split: ~{expected_train} train, ~{expected_val} validation")
        else:
            logger.info(f"Validation disabled: All {len(data)} examples for training")
        
        logger.info("-" * 80)
        
        # Validate input data
        required_cols = [self.task_config.input_field] + self.task_config.output_fields
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Validated columns: {', '.join(required_cols)}")
        
        # Run safety checks if enabled
        safety_results = None
        if self.enable_safety and run_safety_checks and self.safety_checker:
            logger.info("-" * 80)
            logger.info("Running safety checks...")
            
            safety_results = self.safety_checker.check_dataset(
                data, 
                text_column=self.task_config.input_field
            )
            
            logger.info(f"PII detected: {safety_results.get('pii_detected', 'N/A')}")
            logger.info(f"Bias issues: {safety_results.get('bias_detected', 'N/A')}")
            logger.info(f"Safe for training: {safety_results.get('safe_for_training', False)}")
            
            if not safety_results.get("safe_for_training", False) and self.safety_config.block_on_safety_failure:
                logger.error("Safety checks failed! Training blocked.")
                raise ValueError(
                    "Safety checks failed. Set safety_config.block_on_safety_failure=False "
                    "to continue with warnings only."
                )
            
            # Apply anonymization if PII detected
            if self.safety_config.enable_pii_detection and safety_results.get("anonymized_texts"):
                data = data.copy()
                data[self.task_config.input_field] = safety_results["anonymized_texts"]
                logger.info("Applied PII redaction to clinical notes")
        
        optimize_memory()
        
        # Prepare data
        logger.info("-" * 80)
        logger.info("Generating training examples...")
        
        train_examples, val_examples = self.data_prep.prepare_data(data)
        
        optimize_memory()
        
        # Calculate statistics
        single_turn_train = sum(1 for ex in train_examples if ex.metadata['type'] == 'single_turn')
        multi_turn_train = len(train_examples) - single_turn_train
        
        single_turn_val = sum(1 for ex in val_examples if ex.metadata['type'] == 'single_turn')
        multi_turn_val = len(val_examples) - single_turn_val
        
        has_validation = len(val_examples) > 0
        
        # Context utilization stats
        train_context_util = [ex.metadata.get('context_utilization', 0) for ex in train_examples]
        avg_train_context = sum(train_context_util) / len(train_context_util) if train_context_util else 0
        
        train_note_lengths = [ex.metadata.get('note_length', 0) for ex in train_examples]
        avg_note_length = sum(train_note_lengths) / len(train_note_lengths) if train_note_lengths else 0
        
        logger.info("=" * 80)
        logger.info("Data Preparation Complete!")
        logger.info("=" * 80)
        logger.info(f"Training examples: {len(train_examples)}")
        logger.info(f"  Single-turn: {single_turn_train} ({single_turn_train/len(train_examples)*100:.1f}%)")
        logger.info(f"  Multi-turn: {multi_turn_train} ({multi_turn_train/len(train_examples)*100:.1f}%)")
        
        if has_validation:
            logger.info(f"Validation examples: {len(val_examples)}")
            logger.info(f"  Single-turn: {single_turn_val} ({single_turn_val/len(val_examples)*100:.1f}%)")
            logger.info(f"  Multi-turn: {multi_turn_val} ({multi_turn_val/len(val_examples)*100:.1f}%)")
        else:
            logger.info("Validation examples: NONE (validation_split=0.0)")
        
        logger.info("-" * 80)
        logger.info(f"Average clinical note length: {avg_note_length:.0f} chars")
        logger.info(f"Average context utilization: {avg_train_context:.1f}%")
        logger.info("-" * 80)
        logger.info(f"TRUE DATASET SIZE: {len(data)} rows → {len(train_examples)} train examples")
        logger.info(f"Ratio: {len(train_examples)/len(data):.2f}x (should be ~1.0 for v1.1.0)")
        logger.info("=" * 80)
        
        return {
            "train_examples": train_examples,
            "val_examples": val_examples,
            "safety_results": safety_results,
            "num_train": len(train_examples),
            "num_val": len(val_examples),
            "has_validation": has_validation,
            "single_turn_count": single_turn_train + single_turn_val,
            "multi_turn_count": multi_turn_train + multi_turn_val,
            "statistics": {
                "train_single_turn": single_turn_train,
                "train_multi_turn": multi_turn_train,
                "val_single_turn": single_turn_val,
                "val_multi_turn": multi_turn_val,
                "avg_note_length": avg_note_length,
                "avg_context_utilization": avg_train_context,
                "data_efficiency_ratio": len(train_examples) / len(data)
            }
        }
        
    
    def train(
        self,
        data: pd.DataFrame,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train model on prepared data.
        
        Args:
            data: Input DataFrame with clinical notes and output fields
            epochs: Number of training epochs (overrides config)
            batch_size: Batch size (overrides config)
            learning_rate: Learning rate (overrides config)
            max_steps: Maximum training steps (overrides config)
            
        Returns:
            Dictionary containing:
                - training_results: Training metrics and loss
                - save_paths: Paths to saved models
                - num_train_examples: Number of training examples
                - has_validation: Whether validation was used
                - training_time: Total training time
                - statistics: Detailed statistics
                
        Example:
            >>> results = trainer.train(
            ...     data=df,
            ...     epochs=3,
            ...     batch_size=2,
            ...     learning_rate=2e-4
            ... )
            >>> print(f"Model saved to: {results['save_paths']['adapter']}")
        """
        start_time = datetime.now()
        
        # Prepare data
        prepared_data = self.prepare_data(data)
        
        # Override training config if specified
        if epochs is not None:
            self.training_config.num_epochs = epochs
            logger.info(f"Overriding epochs: {epochs}")
        if batch_size is not None:
            self.training_config.batch_size = batch_size
            logger.info(f"Overriding batch_size: {batch_size}")
        if learning_rate is not None:
            self.training_config.learning_rate = learning_rate
            logger.info(f"Overriding learning_rate: {learning_rate}")
        if max_steps is not None:
            self.training_config.max_steps = max_steps
            logger.info(f"Overriding max_steps: {max_steps}")
        
        # Initialize trainer
        logger.info("=" * 80)
        logger.info("Initializing trainer...")
        logger.info("=" * 80)
        
        self.trainer = Trainer(
            model_config=self.model_config,
            training_config=self.training_config,
            safety_config=self.safety_config,
            task_config=self.task_config,
            output_dir=self.output_dir
        )
        
        # Setup model
        self.trainer.setup()
        
        optimize_memory()
        
        # Prepare datasets
        logger.info("Preparing datasets for tokenizer...")
        train_dataset = self.data_prep.prepare_dataset(
            prepared_data["train_examples"],
            self.trainer.tokenizer,
            num_proc=4 
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} examples")
        
        if prepared_data["has_validation"]:
            logger.info("Validation is DISABLED in this version (all data used for training)")
            logger.info("Set validation_split=0.0 to avoid this message")
        
        optimize_memory()
        
        # Train
        training_results = self.trainer.train(train_dataset)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save model
        logger.info("=" * 80)
        logger.info("Saving model...")
        save_paths = self.trainer.save_model()
        
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Training time: {training_time / 60:.2f} minutes")
        logger.info(f"Train loss: {training_results.get('train_loss', 'N/A')}")
        logger.info(f"Adapter saved: {save_paths.get('adapter', 'N/A')}")
        logger.info(f"Merged model saved: {save_paths.get('merged', 'N/A')}")
        logger.info("=" * 80)
        
        return {
            "training_results": training_results,
            "save_paths": save_paths,
            "num_train_examples": len(train_dataset),
            "has_validation": False,  # Always False in current version
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "statistics": prepared_data["statistics"]
        }
    
    def train_from_csv(
        self,
        csv_path: str,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train from CSV file.
        
        Args:
            csv_path: Path to CSV file with clinical notes and output fields
            epochs: Number of training epochs (overrides config)
            batch_size: Batch size (overrides config)
            learning_rate: Learning rate (overrides config)
            max_steps: Maximum training steps (overrides config)
            
        Returns:
            Training results dictionary
            
        Example:
            >>> results = trainer.train_from_csv(
            ...     "malnutrition_data.csv",
            ...     epochs=3
            ... )
        """
        logger.info(f"Loading data from: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        data = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(data)} rows from CSV")
        
        return self.train(data, epochs, batch_size, learning_rate, max_steps)
    
    def infer(
        self,
        clinical_note: str,
        question: Optional[str] = None,
        format: str = "text",
        return_full_response: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Run inference on a single clinical note.
        
        Args:
            clinical_note: Clinical text to analyze
            question: Optional specific question (auto-generated if None)
            format: Output format: "text", "json", "xml", or "markdown"
            return_full_response: Return full response dict with metadata
            
        Returns:
            Model response (str) or full response dict
            
        Example:
            >>> response = trainer.infer(
            ...     clinical_note="Patient presents with weight loss...",
            ...     format="json"
            ... )
            >>> print(response)
            {
                "diagnosis": "Malnutrition",
                "severity": "moderate",
                "recommendations": "..."
            }
        """
        if self.inference_pipeline is None:
            if self.trainer is None or self.trainer.model is None:
                raise ValueError(
                    "No trained model available. Train a model first or load one with load_trained_model()"
                )
            
            self.inference_pipeline = InferencePipeline(
                model=self.trainer.model,
                tokenizer=self.trainer.tokenizer,
                task_config=self.task_config
            )
        
        try:
            output_format = OutputFormat(format)
        except ValueError:
            raise ValueError(
                f"Invalid format '{format}'. Must be one of: text, json, xml, markdown"
            )
        
        return self.inference_pipeline.infer(
            clinical_note=clinical_note,
            question=question,
            output_format=output_format,
            return_full_response=return_full_response
        )
    
    def batch_infer(
        self,
        clinical_notes: List[str],
        questions: Optional[List[str]] = None,
        format: str = "text",
        show_progress: bool = True
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Batch inference on multiple clinical notes.
        
        Args:
            clinical_notes: List of clinical texts
            questions: Optional list of questions (one per note)
            format: Output format: "text", "json", "xml", or "markdown"
            show_progress: Show progress bar
            
        Returns:
            List of model responses
            
        Example:
            >>> notes = ["Patient 1 note...", "Patient 2 note..."]
            >>> responses = trainer.batch_infer(notes, format="json")
        """
        if self.inference_pipeline is None:
            if self.trainer is None or self.trainer.model is None:
                raise ValueError(
                    "No trained model available. Train a model first or load one with load_trained_model()"
                )
            
            self.inference_pipeline = InferencePipeline(
                model=self.trainer.model,
                tokenizer=self.trainer.tokenizer,
                task_config=self.task_config
            )
        
        try:
            output_format = OutputFormat(format)
        except ValueError:
            raise ValueError(
                f"Invalid format '{format}'. Must be one of: text, json, xml, markdown"
            )
        
        logger.info(f"Running batch inference on {len(clinical_notes)} notes...")
        
        results = self.inference_pipeline.batch_infer(
            clinical_notes=clinical_notes,
            questions=questions,
            output_format=output_format
        )
        
        logger.info(f"Batch inference complete: {len(results)} results")
        
        return results
    
    def load_trained_model(self, model_path: str):
        """
        Load a previously trained model for inference.
        
        Args:
            model_path: Path to saved model (adapter or merged)
            
        Example:
            >>> trainer = MedDialogue(task_config=config, model_type="llama")
            >>> trainer.load_trained_model("./output/adapter_llama_20241125_143022")
            >>> response = trainer.infer("Clinical note...")
        """
        from .models import load_model
        
        logger.info("=" * 80)
        logger.info(f"Loading trained model from: {model_path}")
        logger.info("=" * 80)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        self.model_config.model_name = model_path
        
        model, tokenizer = load_model(self.model_config)
        
        self.inference_pipeline = InferencePipeline(
            model=model,
            tokenizer=tokenizer,
            task_config=self.task_config
        )
        
        logger.info("Model loaded successfully!")
        logger.info("=" * 80)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of all configurations.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "version": "1.1.0",
            "task": self.task_config.to_dict(),
            "model": self.model_config.to_dict(),
            "training": self.training_config.__dict__,
            "data_preparation": {
                "single_turn_ratio": self.conversation_config.single_turn_ratio,
                "max_multi_turns": self.conversation_config.max_multi_turns,
                "logical_style_ratio": self.conversation_config.logical_style_ratio,
                "context_window": self.conversation_config.context_window_size,
                "include_typos": self.conversation_config.include_typos,
                "typo_ratio": self.conversation_config.typo_ratio,
                "validation_split": self.conversation_config.validation_split,
                "has_validation": self.conversation_config.validation_split > 0
            },
            "safety": {
                "enabled": self.enable_safety,
                "pii_detection": self.safety_config.enable_pii_detection if self.enable_safety else False,
                "bias_monitoring": self.safety_config.enable_bias_monitoring if self.enable_safety else False
            }
        }