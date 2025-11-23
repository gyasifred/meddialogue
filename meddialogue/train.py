"""
Training Orchestration for MedDialogue
======================================

Handles model training with LoRA fine-tuning.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import os
import unsloth
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import json
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig

from .config import TrainingConfig, ModelConfig, SafetyConfig, TaskConfig
from .models import load_model, apply_lora, optimize_memory

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer for fine-tuning medical language models.
    
    Features:
    - LoRA fine-tuning for memory efficiency
    - Optional validation split (controlled by config)
    - Automatic fallback for OOM errors
    - Comprehensive logging and metrics
    - Saves both adapter and merged models
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        safety_config: SafetyConfig,
        task_config: TaskConfig,
        output_dir: str = "./output"
    ):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training hyperparameters
            safety_config: Safety configuration
            task_config: Task configuration
            output_dir: Output directory for models and logs
        """
        self.model_config = model_config
        self.training_config = training_config
        self.safety_config = safety_config
        self.task_config = task_config
        self.output_dir = output_dir
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info(f"Trainer initialized for: {task_config.task_name}")
        logger.info(f"Model: {model_config.model_name}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)
    
    def setup(self, max_seq_length: Optional[int] = None):
        """
        Setup model and tokenizer with LoRA adapters.
        
        Args:
            max_seq_length: Maximum sequence length (uses config default if None)
        """
        optimize_memory(aggressive=True)
        
        logger.info("=" * 80)
        logger.info("Setting up model and tokenizer...")
        logger.info("=" * 80)
        
        max_seq_length = max_seq_length or self.model_config.max_seq_length
        
        logger.info(f"Loading base model: {self.model_config.model_name}")
        logger.info(f"Max sequence length: {max_seq_length}")
        
        self.model, self.tokenizer = load_model(
            self.model_config,
            max_seq_length=max_seq_length
        )
        
        logger.info("Base model loaded successfully")
        logger.info("-" * 80)
        logger.info("Applying LoRA adapters...")
        logger.info(f"  LoRA rank (r): {self.model_config.lora_config.r}")
        logger.info(f"  LoRA alpha: {self.model_config.lora_config.lora_alpha}")
        logger.info(f"  Target modules: {', '.join(self.model_config.lora_config.target_modules)}")
        
        # Apply LoRA
        self.model = apply_lora(
            self.model,
            self.model_config.lora_config,
            self.model_config.lora_config.target_modules
        )
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = 100 * trainable_params / total_params
        
        logger.info("-" * 80)
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info("=" * 80)
        logger.info("Setup complete!")
        logger.info("=" * 80)
        optimize_memory()
        
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """
        Train the model on prepared datasets.
        
        Args:
            train_dataset: Training dataset (prepared with tokenizer)
            eval_dataset: Optional evaluation dataset (can be None if validation_split=0)
            
        Returns:
            Dictionary with training metrics:
                - train_loss: Final training loss
                - eval_loss: Final eval loss (if validation used)
                - train_runtime: Total training time
                - train_samples_per_second: Training throughput
                - has_validation: Whether validation was used
                - fallback_mode: Whether fallback mode was used
                
        Raises:
            ValueError: If model not setup
            RuntimeError: If training fails unrecoverably
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not setup. Call setup() first")
        
        has_validation = eval_dataset is not None and len(eval_dataset) > 0
        
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)
        logger.info(f"Train examples: {len(train_dataset)}")
        
        if has_validation:
            logger.info(f"Eval examples: {len(eval_dataset)}")
            logger.info("Validation: ENABLED")
        else:
            logger.info("Validation: DISABLED (using all data for training)")
        
        logger.info("-" * 80)
        logger.info("Training Configuration:")
        logger.info(f"  Epochs: {self.training_config.num_epochs}")
        logger.info(f"  Batch size: {self.training_config.batch_size}")
        logger.info(f"  Gradient accumulation: {self.training_config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.training_config.batch_size * self.training_config.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {self.training_config.learning_rate}")
        logger.info(f"  Max steps: {self.training_config.max_steps if self.training_config.max_steps else 'Not set'}")
        logger.info(f"  FP16: {self.training_config.fp16}")
        logger.info(f"  BF16: {self.training_config.bf16}")
        logger.info("=" * 80)
        
        # Subsample if configured (for quick testing)
        if not self.training_config.use_full_dataset:
            logger.warning("=" * 80)
            logger.warning(f"USING SUBSAMPLE MODE: {self.training_config.subsample_size} examples")
            logger.warning("This is for quick testing only. Set use_full_dataset=True for production.")
            logger.warning("=" * 80)
            
            subsample_size = min(len(train_dataset), self.training_config.subsample_size)
            train_dataset = train_dataset.select(range(subsample_size))
            
            if has_validation:
                eval_size = min(len(eval_dataset), self.training_config.subsample_size // 5)
                eval_dataset = eval_dataset.select(range(eval_size))
            
            logger.info(f"Subsampled to: {len(train_dataset)} train, {len(eval_dataset) if has_validation else 0} eval")
        
        # Create training args (handles validation conditionally)
        training_args = self._create_training_args(has_validation=has_validation)
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.model_config.max_seq_length,
            return_tensors="pt",
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        logger.info("Data collator created")
        logger.info("-" * 80)
        
        try:
            # Create trainer
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if has_validation else None,
                dataset_text_field="text",
                max_seq_length=self.model_config.max_seq_length,
                data_collator=data_collator,
                packing=False,
                args=training_args,
                dataset_num_proc=4
            )
            
            logger.info("SFTTrainer created")
            logger.info("=" * 80)
            logger.info("Beginning training loop...")
            logger.info("=" * 80)
            
            try:
                optimize_memory(aggressive=True) 
                train_result = self.trainer.train()
                
                logger.info("=" * 80)
                logger.info("Training completed successfully!")
                logger.info("=" * 80)
                logger.info(f"Train loss: {train_result.training_loss if hasattr(train_result, 'training_loss') else 'N/A'}")
                logger.info(f"Train runtime: {train_result.metrics.get('train_runtime', 'N/A'):.2f}s")
                logger.info(f"Samples/second: {train_result.metrics.get('train_samples_per_second', 'N/A'):.2f}")
                
                # Evaluate if validation enabled
                eval_loss = None
                if has_validation:
                    logger.info("-" * 80)
                    logger.info("Running final evaluation...")
                    eval_results = self.trainer.evaluate()
                    eval_loss = eval_results.get('eval_loss')
                    logger.info(f"Eval loss: {eval_loss}")
                
                logger.info("=" * 80)
                
                return {
                    "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
                    "eval_loss": eval_loss,
                    "train_runtime": train_result.metrics.get('train_runtime'),
                    "train_samples_per_second": train_result.metrics.get('train_samples_per_second'),
                    "train_steps_per_second": train_result.metrics.get('train_steps_per_second'),
                    "total_flos": train_result.metrics.get('total_flos'),
                    "has_validation": has_validation,
                    "fallback_mode": False
                }
                
            except RuntimeError as e:
                if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                    logger.error("=" * 80)
                    logger.error("OUT OF MEMORY ERROR DETECTED!")
                    logger.error("=" * 80)
                    logger.error(f"Error: {str(e)}")
                    logger.warning("Attempting recovery with fallback settings...")
                    logger.warning("=" * 80)
                    return self._train_with_fallback(train_dataset, eval_dataset)
                else:
                    logger.error("=" * 80)
                    logger.error("TRAINING FAILED WITH UNEXPECTED ERROR")
                    logger.error("=" * 80)
                    logger.error(f"Error: {str(e)}")
                    raise
        
        finally:
            logger.info("Cleaning up training resources...")
            
            # Clean up trainer references
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.train_dataset = None
                self.trainer.eval_dataset = None
                self.trainer = None
            
            # Delete dataset references
            if 'train_dataset' in locals():
                del train_dataset
            if 'eval_dataset' in locals():
                del eval_dataset
            
            optimize_memory(aggressive=True)
        
    def _train_with_fallback(
        self, 
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset]
    ) -> Dict[str, Any]:
        """
        Fallback training with aggressive memory reduction.
        
        Applied when OOM error occurs:
        - Reduces batch size to 1
        - Increases gradient accumulation to 64
        - Limits dataset to 1000 training examples
        - Reduces sequence length if needed
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Training metrics with fallback_mode=True
        """
        logger.warning("=" * 80)
        logger.warning("APPLYING FALLBACK SETTINGS")
        logger.warning("=" * 80)
        
        optimize_memory(aggressive=True)
        
        has_validation = eval_dataset is not None and len(eval_dataset) > 0
        
        # Reduce memory footprint
        original_batch_size = self.training_config.batch_size
        original_grad_accum = self.training_config.gradient_accumulation_steps
        original_seq_len = self.model_config.max_seq_length
        
        self.training_config.batch_size = 1
        self.training_config.gradient_accumulation_steps = 64
        
        logger.warning(f"Batch size: {original_batch_size} → 1")
        logger.warning(f"Gradient accumulation: {original_grad_accum} → 64")
        logger.warning(f"Effective batch size maintained at: {1 * 64}")
        
        # Subsample dataset
        max_train = min(len(train_dataset), 1000)
        train_subset = train_dataset.select(range(max_train))
        
        eval_subset = None
        if has_validation:
            max_eval = min(len(eval_dataset), 200)
            eval_subset = eval_dataset.select(range(max_eval))
            logger.warning(f"Dataset: {len(train_dataset)} → {len(train_subset)} train, {len(eval_dataset)} → {len(eval_subset)} eval")
        else:
            logger.warning(f"Dataset: {len(train_dataset)} → {len(train_subset)} train examples")
        
        # Reduce sequence length if very long
        if self.model_config.max_seq_length > 8192:
            self.model_config.max_seq_length = 2048
            logger.warning(f"Sequence length: {original_seq_len} → 2048")
        
        logger.warning("=" * 80)
        
        # Clear memory
        optimize_memory(aggressive=True) 
        
        # Create new training args with fallback settings
        training_args = self._create_training_args(has_validation=has_validation)
        
        # Create data collator with reduced sequence length
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.model_config.max_seq_length,
            return_tensors="pt",
            label_pad_token_id=-100
        )
        
        try:
            # Create new trainer
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_subset,
                eval_dataset=eval_subset if has_validation else None,
                dataset_text_field="text",
                max_seq_length=self.model_config.max_seq_length,
                data_collator=data_collator,
                packing=False,
                args=training_args,
                dataset_num_proc=4
            )
            
            logger.warning("Restarting training with fallback settings...")
            logger.warning("=" * 80)
            
            # Train with fallback
            train_result = self.trainer.train()
            
            logger.warning("=" * 80)
            logger.warning("Training completed with FALLBACK MODE")
            logger.warning("=" * 80)
            logger.warning(f"Train loss: {train_result.training_loss if hasattr(train_result, 'training_loss') else 'N/A'}")
            logger.warning(f"Trained on: {len(train_subset)} examples (subsampled)")
            logger.warning("=" * 80)
            logger.warning("IMPORTANT: Fallback mode used reduced dataset and settings.")
            logger.warning("Consider using a GPU with more memory for full training.")
            logger.warning("=" * 80)
            
            return {
                "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
                "fallback_mode": True,
                "has_validation": has_validation,
                "train_samples": len(train_subset),
                "eval_samples": len(eval_subset) if has_validation else 0,
                "original_batch_size": original_batch_size,
                "fallback_batch_size": 1,
                "original_grad_accum": original_grad_accum,
                "fallback_grad_accum": 64
            }
        
        finally:
            # Cleanup for fallback
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.train_dataset = None
                self.trainer.eval_dataset = None
            
            if 'train_subset' in locals():
                del train_subset
            if 'eval_subset' in locals():
                del eval_subset
            
            optimize_memory(aggressive=True)
            
    def _create_training_args(self, has_validation: bool = False) -> SFTConfig:
        """
        Create training arguments from configuration.
        
        Args:
            has_validation: Whether validation dataset is being used
        
        Returns:
            SFTConfig object for SFTTrainer
        """
        cfg = self.training_config
        
        # Evaluation strategy based on whether we have validation
        if has_validation:
            eval_strategy = cfg.evaluation_strategy
            eval_steps = cfg.eval_steps if cfg.eval_steps else 500
            load_best_model = True if eval_strategy != "no" else False
        else:
            eval_strategy = "no"
            eval_steps = None
            load_best_model = False
        
        return SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size if has_validation else None,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_ratio=cfg.warmup_ratio,
            num_train_epochs=cfg.num_epochs if cfg.max_steps is None else 1,
            max_steps=cfg.max_steps if cfg.max_steps else -1,
            learning_rate=cfg.learning_rate,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            logging_steps=cfg.logging_steps,
            optim="adamw_8bit",
            weight_decay=cfg.weight_decay,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
            save_strategy=cfg.save_strategy,
            save_steps=cfg.save_steps if cfg.save_steps else 500,
            save_total_limit=cfg.save_total_limit,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            load_best_model_at_end=load_best_model,
            dataloader_num_workers=0,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, 
            max_grad_norm=cfg.max_grad_norm,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False 
        )
    
    def save_model(
        self, 
        save_merged: bool = True, 
        save_adapter: bool = True
    ) -> Dict[str, str]:
        """
        Save trained model (adapter and/or merged).
        
        Args:
            save_merged: Save merged model (base + adapter)
            save_adapter: Save adapter only (lightweight, recommended)
            
        Returns:
            Dictionary with save paths:
                - adapter: Path to adapter (if saved)
                - merged: Path to merged model (if saved)
                
        Note:
            Adapter-only saves are much faster and smaller (~100MB vs several GB).
            Merged models are convenient for deployment but take longer to save.
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        logger.info("=" * 80)
        logger.info("Saving model...")
        logger.info("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}
        
        try:
            optimize_memory()
            
            # Save adapter (fast, small, recommended)
            if save_adapter:
                adapter_path = os.path.join(
                    self.output_dir,
                    f"adapter_{self.model_config.model_type}_{timestamp}"
                )
                os.makedirs(adapter_path, exist_ok=True)
                
                logger.info(f"Saving adapter to: {adapter_path}")
                
                self.model.save_pretrained(adapter_path)
                self.tokenizer.save_pretrained(adapter_path)
                paths['adapter'] = adapter_path
                
                logger.info(f"✓ Adapter saved: {adapter_path}")
                logger.info(f"  Size: ~100-200MB (LoRA weights only)")
            
            # Save merged model (slow, large, convenient for deployment)
            if save_merged:
                merged_path = os.path.join(
                    self.output_dir,
                    f"merged_{self.model_config.model_type}_{timestamp}"
                )
                os.makedirs(merged_path, exist_ok=True)
                
                logger.info(f"Saving merged model to: {merged_path}")
                logger.info("  This may take several minutes...")
                
                try:
                    self.model.save_pretrained_merged(
                        merged_path,
                        self.tokenizer,
                        save_method="merged_16bit",
                        max_shard_size="2GB",
                        push_to_hub=False
                    )
                    paths['merged'] = merged_path
                    logger.info(f"✓ Merged model saved: {merged_path}")
                    logger.info(f"  Size: Several GB (full model)")
                    
                except Exception as e:
                    logger.warning(f"Merged save with 'merged_16bit' failed: {e}")
                    logger.warning("Trying alternative method with 'lora'...")
                    
                    self.model.save_pretrained_merged(
                        merged_path,
                        self.tokenizer,
                        save_method="lora",
                        max_shard_size="2GB"
                    )
                    paths['merged'] = merged_path
                    logger.info(f"✓ Merged model saved (LoRA method): {merged_path}")
            
            # Save training metadata
            self._save_metadata(timestamp)
            
            logger.info("-" * 80)
            logger.info("Model saving complete!")
            logger.info("=" * 80)
            
            return paths
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error("MODEL SAVE FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}")
            raise
            
        finally:
            optimize_memory()
    
    def _save_metadata(self, timestamp: str):
        """
        Save training metadata and configuration.
        
        Args:
            timestamp: Timestamp string for filename
        """
        metadata = {
            "framework": "MedDialogue",
            "version": "1.0.0",
            "author": "Frederick Gyasi (gyasi@musc.edu)",
            "institution": "Medical University of South Carolina, Biomedical Informatics Center",
            "timestamp": timestamp,
            "date": datetime.now().isoformat(),
            "task": self.task_config.to_dict(),
            "model": self.model_config.to_dict(),
            "training": {
                "num_epochs": self.training_config.num_epochs,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
                "max_steps": self.training_config.max_steps,
                "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
                "warmup_ratio": self.training_config.warmup_ratio,
                "weight_decay": self.training_config.weight_decay,
                "fp16": self.training_config.fp16,
                "bf16": self.training_config.bf16,
                "use_full_dataset": self.training_config.use_full_dataset
            },
            "notes": {
                "data_mapping": "1:1 row to example (v1.0.0)",
                "no_artificial_multiplication": True,
                "reasoning_styles": "16 styles (7 grammatical + 9 logical)",
                "intelligent_field_ordering": True,
                "optional_validation": True,
                "lora_fine_tuning": True,
                "context_optimized": True
            }
        }
        
        metadata_path = os.path.join(self.output_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved: {metadata_path}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics and model info.
        
        Returns:
            Dictionary with training statistics
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "model_name": self.model_config.model_name,
            "model_type": self.model_config.model_type,
            "task_name": self.task_config.task_name,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": 100 * trainable_params / total_params,
            "lora_rank": self.model_config.lora_config.r,
            "lora_alpha": self.model_config.lora_config.lora_alpha,
            "max_seq_length": self.model_config.max_seq_length,
            "training_config": {
                "epochs": self.training_config.num_epochs,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
                "gradient_accumulation": self.training_config.gradient_accumulation_steps
            }
        }