"""
Model Management for MedDialogue
================================

Handles model loading, LoRA application, and model registry.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
"""
import  unsloth
import logging
import torch
import gc
from typing import Tuple, List, Optional, Any
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from .config import ModelConfig, LoRAConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry of supported model configurations.
    
    Includes pre-configured settings for popular medical LLMs.
    Users can also add custom models using the custom_model parameter.
    """
    
    CONFIGS = {
        "llama": {
            "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "chat_template": "llama-3.1",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_seq_length": 32768,
            "supports_system_role": True,
        },
        "phi-4": {
            "model_name": "unsloth/Phi-4-bnb-4bit",
            "chat_template": "phi-4",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_seq_length": 16384,
            "supports_system_role": True,
        },
        "mistral": {
            "model_name": "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
            "chat_template": "mistral",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_seq_length": 32768,
            "supports_system_role": False,
        },
        "qwen": {
            "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "chat_template": "qwen2.5",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_seq_length": 32768,
            "supports_system_role": True,
        }
    }
    
    @classmethod
    def get_config(cls, model_type: str) -> dict:
        """Get model configuration."""
        if model_type not in cls.CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls.CONFIGS.keys())}")
        return cls.CONFIGS[model_type].copy()
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available model types."""
        return list(cls.CONFIGS.keys())


def get_supported_models() -> List[str]:
    """Get list of supported model types."""
    return ModelRegistry.list_models()


def detect_chat_template(model_name: str) -> str:
    """
    Auto-detect chat template from model name.
    
    Args:
        model_name: Hugging Face model name
        
    Returns:
        Chat template name
    """
    model_name_lower = model_name.lower()
    
    if 'llama' in model_name_lower:
        if '3.1' in model_name_lower or '3.2' in model_name_lower:
            return "llama-3.1"
        elif '3' in model_name_lower:
            return "llama-3"
        else:
            return "llama-2"
    elif 'phi-4' in model_name_lower:
        return "phi-4"
    elif 'phi-3' in model_name_lower:
        return "phi-3"
    elif 'mistral' in model_name_lower:
        return "mistral"
    elif 'qwen' in model_name_lower:
        return "qwen2.5" if '2.5' in model_name_lower else "qwen2"
    elif 'gemma' in model_name_lower:
        return "gemma"
    else:
        logger.warning(f"Unknown model type, using default 'chatml' template")
        return "chatml"


def load_model(model_config: ModelConfig, max_seq_length: Optional[int] = None) -> Tuple[Any, Any]:
    """
    Load model and tokenizer with Unsloth optimizations.
    
    Args:
        model_config: Model configuration
        max_seq_length: Maximum sequence length (uses config default if None)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    max_seq_length = max_seq_length or model_config.max_seq_length
    
    logger.info(f"Loading model: {model_config.model_name}")
    logger.info(f"Max sequence length: {max_seq_length}")
    
    # Determine device and dtype
    if torch.cuda.is_available():
        device_map = {"": torch.cuda.current_device()}
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device_map = {"": "cpu"}
        dtype = torch.float32
        logger.info("CUDA not available, using CPU")
    
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
        device_map=device_map,
        trust_remote_code=True,
        use_cache=False,  # ✅ FIXED: Changed from True to False
        low_cpu_mem_usage=True
    )
    
    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=model_config.chat_template
    )
    
    # Configure tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.model_max_length = max_seq_length
    
    # Configure model
    if hasattr(model, 'config'):
        model.config.use_cache = False 
        model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info("Model and tokenizer loaded successfully")
    
    return model, tokenizer

def apply_lora(model: Any, lora_config: LoRAConfig, target_modules: List[str]) -> Any:
    """
    Apply LoRA adapters to model.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        target_modules: List of modules to apply LoRA to
        
    Returns:
        Model with LoRA adapters
    """
    logger.info("Applying LoRA adapters...")
    logger.info(f"LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
    
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=lora_config.r,
        target_modules=target_modules,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        use_rslora=lora_config.use_rslora,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        loftq_config=None,
        init_lora_weights=lora_config.init_lora_weights
    )
    
    logger.info("LoRA adapters applied successfully")
    
    return model


def optimize_memory(aggressive: bool = False):
    """
    Optimize memory usage by clearing cache and collecting garbage.
    
    Args:
        aggressive: If True, perform multiple cleanup passes
    
    This is especially important when training multiple models sequentially
    or when dealing with memory constraints.
    """
    # Standard garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if aggressive:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                torch.cuda.empty_cache()
                gc.collect()
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.debug(f"GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    if aggressive:
        for _ in range(3):
            gc.collect()
            


def get_device(cuda_device: int = 0) -> torch.device:
    """
    Get appropriate device for training/inference.
    
    Args:
        cuda_device: CUDA device index
        
    Returns:
        torch.device
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return torch.device("cpu")
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    return torch.device("cuda:0")