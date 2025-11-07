#!/usr/bin/env python
"""
Pediatric Malnutrition Assessment - Gradio Chat Interface

Author: Frederick Gyasi (gyasi@musc.edu)
Affiliation: Medical University of South Carolina, Biomedical Informatics Center, Clinical NLP Lab
Version: 1.2.0
License: MIT
Date: November 2025

CHANGELOG v1.2.0 (2025-11-07):
------------------------------
- UX: Added STOP button to cancel generation mid-process ⏹️
- UX: Text input remains usable while processing (type next question while waiting)
- UX: Concurrent operations enabled - no blocking
- PERF: Queue-based processing for better responsiveness

CHANGELOG v1.1.0 (2025-11-07):
------------------------------
- PERFORMANCE: Optimized cache clearing (10x less frequent) - saves ~10-50ms per generation
- PERFORMANCE: Added explicit use_cache=True for KV cache - faster generation
- PERFORMANCE: Immediate tensor deletion (del inputs, del outputs) - better memory management
- PERFORMANCE: Smart cache clearing based on generation count
- BUG FIX: Fixed CSV path input bug (was passing None instead of csv_path_input)
- DOCS: Added explicit conversation history documentation
- DOCS: Clarified that conversation history is properly maintained ✅

CONVERSATION HISTORY CONFIRMED: Follow-up questions see full context! ✅
"""

import os
import sys
import json
import torch
import logging
import argparse
import shutil
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Third-party imports
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import gradio as gr
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not found - {e}")
    print("Install required packages: pip install unsloth gradio pandas")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class MessageRole(Enum):
    """Enumeration of conversation message roles."""
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Data structure representing a single message in a clinical conversation."""
    role: str
    content: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary format for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Reconstruct message object from dictionary data."""
        return cls(**data)


@dataclass
class ConversationSession:
    """Complete conversation session data structure for clinical consultations."""
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Message]
    clinical_note: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert complete session to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [msg.to_dict() for msg in self.messages],
            "clinical_note": self.clinical_note,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationSession':
        """Reconstruct session object from dictionary data."""
        messages = [Message.from_dict(msg) for msg in data.get("messages", [])]
        return cls(
            session_id=data["session_id"],
            title=data["title"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=messages,
            clinical_note=data.get("clinical_note", ""),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# Conversation Memory Manager
# ============================================================================

class ConversationMemoryManager:
    """Persistent storage and retrieval system for clinical conversation sessions."""
    
    def __init__(self, storage_dir: str = "./conversation_history"):
        """Initialize the conversation memory management system."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.backup_dir = self.storage_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.current_session_id: Optional[str] = None
        
        logger.info(f"Memory manager initialized: {self.storage_dir}")
    
    def create_session(self, title: str = None) -> str:
        """Create a new conversation session with unique identifier."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
        
        if not title:
            title = f"Consultation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = ConversationSession(
            session_id=session_id,
            title=title,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            messages=[],
            clinical_note="",
            metadata={"message_count": 0}
        )
        
        self.active_sessions[session_id] = session
        self.current_session_id = session_id
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to an existing conversation session."""
        if session_id not in self.active_sessions:
            return False
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        self.active_sessions[session_id].messages.append(message)
        self.active_sessions[session_id].updated_at = datetime.now().isoformat()
        
        return True
    
    def update_clinical_note(self, session_id: str, clinical_note: str) -> bool:
        """Update the clinical note associated with a session."""
        if session_id not in self.active_sessions:
            return False
        
        self.active_sessions[session_id].clinical_note = clinical_note
        self.active_sessions[session_id].updated_at = datetime.now().isoformat()
        return True
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a session object by its identifier."""
        return self.active_sessions.get(session_id)
    
    def save_session(self, session_id: str, create_backup: bool = True) -> Optional[str]:
        """Persist session data to disk with optional backup creation."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        file_path = self.storage_dir / f"{session_id}.json"
        
        if create_backup and file_path.exists():
            backup_path = self.backup_dir / f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(file_path, backup_path)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session saved: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return None
    
    def load_session(self, file_path: str) -> Optional[str]:
        """Load a session from disk into active memory."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = ConversationSession.from_dict(data)
            self.active_sessions[session.session_id] = session
            self.current_session_id = session.session_id
            
            logger.info(f"Session loaded: {session.session_id}")
            return session.session_id
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None
    
    def get_all_saved_sessions(self) -> List[Dict[str, str]]:
        """Retrieve metadata for all saved sessions."""
        sessions = []
        
        for file_path in sorted(self.storage_dir.glob("*.json"), reverse=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sessions.append({
                    "session_id": data["session_id"],
                    "title": data["title"],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "message_count": len(data.get("messages", [])),
                    "file_path": str(file_path)
                })
            except Exception as e:
                logger.warning(f"Could not read session file {file_path}: {e}")
                continue
        
        return sessions
    
    def export_session_txt(self, session_id: str) -> str:
        """Export session to formatted plain text representation."""
        if session_id not in self.active_sessions:
            return "Session not found"
        
        session = self.active_sessions[session_id]
        lines = []
        
        lines.append("=" * 80)
        lines.append(f"CLINICAL CONSULTATION RECORD")
        lines.append("=" * 80)
        lines.append(f"Title: {session.title}")
        lines.append(f"Session ID: {session.session_id}")
        lines.append(f"Created: {session.created_at}")
        lines.append(f"Messages: {len(session.messages)}")
        lines.append("=" * 80)
        lines.append("")
        
        if session.clinical_note:
            lines.append("CLINICAL NOTE:")
            lines.append("-" * 80)
            lines.append(session.clinical_note)
            lines.append("-" * 80)
            lines.append("")
        
        lines.append("CONVERSATION:")
        lines.append("-" * 80)
        
        for i, msg in enumerate(session.messages, 1):
            role_label = "ASSISTANT" if msg.role == "assistant" else "USER"
            lines.append(f"\n[{i}] {role_label} ({msg.timestamp}):")
            lines.append(msg.content)
            lines.append("")
        
        lines.append("=" * 80)
        lines.append(f"END OF CONSULTATION RECORD")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Calculate and return memory usage statistics."""
        total_messages = sum(len(s.messages) for s in self.active_sessions.values())
        total_sessions = len(self.active_sessions)
        saved_sessions = len(list(self.storage_dir.glob("*.json")))
        
        return {
            "active_sessions": total_sessions,
            "total_messages": total_messages,
            "saved_sessions": saved_sessions,
            "storage_dir": str(self.storage_dir)
        }


# ============================================================================
# CSV Patient Data Manager
# ============================================================================

class CSVPatientDataManager:
    """Manager for batch processing of patient clinical data from CSV files."""
    
    def __init__(self, csv_path: str = None, deid_col: str = "DEID", text_col: str = "txt"):
        """Initialize CSV patient data manager with optional immediate loading."""
        self.csv_path = csv_path
        self.deid_col = deid_col
        self.text_col = text_col
        self.df = None
        self.patient_ids = []
        
        if csv_path and os.path.exists(csv_path):
            self.load_csv(csv_path, deid_col, text_col)
    
    def load_csv(self, csv_path: str, deid_col: str = None, text_col: str = None) -> Tuple[bool, str]:
        """Load and parse CSV file with patient clinical data."""
        try:
            self.csv_path = csv_path
            if deid_col:
                self.deid_col = deid_col
            if text_col:
                self.text_col = text_col
            
            self.df = pd.read_csv(csv_path)
            
            if self.deid_col not in self.df.columns:
                return False, f"Column '{self.deid_col}' not found. Available: {list(self.df.columns)}"
            
            if self.text_col not in self.df.columns:
                return False, f"Column '{self.text_col}' not found. Available: {list(self.df.columns)}"
            
            self.patient_ids = self.df[self.deid_col].astype(str).tolist()
            
            logger.info(f"CSV loaded: {len(self.patient_ids)} patients")
            return True, f"✅ Loaded {len(self.patient_ids)} patients from CSV"
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return False, f"❌ Error loading CSV: {str(e)}"
    
    def get_patient_ids(self) -> List[str]:
        """Retrieve list of all patient identifiers from loaded CSV."""
        return self.patient_ids
    
    def get_clinical_note(self, patient_id: str) -> str:
        """Retrieve clinical note text for a specific patient."""
        if self.df is None:
            return "No CSV loaded"
        
        try:
            patient_row = self.df[self.df[self.deid_col].astype(str) == str(patient_id)]
            
            if patient_row.empty:
                return f"Patient ID '{patient_id}' not found in CSV"
            
            clinical_text = patient_row[self.text_col].iloc[0]
            
            if pd.isna(clinical_text):
                return f"No clinical note available for patient '{patient_id}'"
            
            return str(clinical_text)
        except Exception as e:
            logger.error(f"Error retrieving note for patient {patient_id}: {e}")
            return f"Error retrieving clinical note: {str(e)}"
    
    def get_info(self) -> Dict[str, Any]:
        """Get metadata about loaded CSV dataset."""
        if self.df is None:
            return {"loaded": False, "patient_count": 0}
        
        return {
            "loaded": True,
            "csv_path": self.csv_path,
            "patient_count": len(self.patient_ids),
            "deid_column": self.deid_col,
            "text_column": self.text_col
        }


# ============================================================================
# GPU Device Manager
# ============================================================================

def get_optimal_device(cuda_device: int = None) -> torch.device:
    """Select optimal CUDA device for model inference with intelligent fallback."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available - using CPU")
        return torch.device("cpu")
    
    device_count = torch.cuda.device_count()
    logger.info(f"Detected {device_count} CUDA device(s)")
    
    if cuda_device is not None:
        if 0 <= cuda_device < device_count:
            device = torch.device(f"cuda:{cuda_device}")
            logger.info(f"Using specified GPU {cuda_device}")
            return device
    
    if device_count == 1:
        device = torch.device("cuda:0")
        logger.info(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        return device
    
    best_device = 0
    max_free_memory = 0
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        total = props.total_memory
        free = total - (allocated + reserved)
        
        if free > max_free_memory:
            max_free_memory = free
            best_device = i
    
    device = torch.device(f"cuda:{best_device}")
    logger.info(f"Selected GPU {best_device} with most free memory")
    
    return device


# ============================================================================
# Chat Template Detection
# ============================================================================

def detect_chat_template(model_name: str) -> str:
    """Automatically detect appropriate chat template from model name."""
    model_lower = model_name.lower()
    
    template_map = {
        'llama-3.1': ['llama', '3.1'],
        'llama-3': ['llama', '3'],
        'phi-4': ['phi-4'],
        'phi-3': ['phi-3', 'phi'],
        'mistral': ['mistral'],
        'qwen2.5': ['qwen', '2.5'],
        'chatml': ['yi', 'chat']
    }
    
    for template, keywords in template_map.items():
        if all(kw in model_lower for kw in keywords):
            logger.info(f"Detected chat template: {template}")
            return template
    
    logger.warning("Could not detect chat template, defaulting to 'chatml'")
    return 'chatml'


# ============================================================================
# Model Type Detection
# ============================================================================

def detect_model_type(model_path: str, override_type: str = None) -> str:
    """Detect model architecture type from path or apply manual override."""
    if override_type:
        return override_type
    
    path_lower = model_path.lower()
    
    if "phi-4" in path_lower:
        return "phi-4"
    elif "phi" in path_lower:
        return "phi"
    elif "llama" in path_lower:
        return "llama"
    elif "mistral" in path_lower:
        return "mistral"
    elif "qwen" in path_lower:
        return "qwen"
    else:
        return "llama"


# ============================================================================
# Conversational Clinical AI Engine
# ============================================================================

class ClinicalConversationEngine:
    """
    Core AI engine for clinical malnutrition assessment conversations.

    CRITICAL: Clinical text NEVER passed unless explicitly loaded by user.

    CONVERSATION HISTORY: Properly maintained across turns!
    - self.current_conversation stores full conversation history
    - Each user message appended before generation
    - Each assistant response appended after generation
    - Full history passed to model on every turn
    - Follow-up questions see complete context ✅
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = None,
        custom_chat_template: str = None,
        cuda_device: int = None,
        temperature: float = 0.3,
        max_seq_length: int = 4096,
        csv_manager: CSVPatientDataManager = None
    ):
        """Initialize clinical conversation engine with configuration."""
        self.model_path = model_path
        self.model_type = detect_model_type(model_path, model_type)
        self.custom_chat_template = custom_chat_template
        self.temperature = temperature
        self.max_seq_length = max_seq_length
        self.csv_manager = csv_manager

        self.model = None
        self.tokenizer = None
        self.device = get_optimal_device(cuda_device)

        self.memory = ConversationMemoryManager()

        # CRITICAL: Conversation history maintained here
        self.current_conversation: List[Dict[str, str]] = []
        self.clinical_note_included = False

        # Performance tracking
        self.generation_count = 0  # Track number of generations for cache optimization

        logger.info(f"Clinical Conversation Engine initialized")
    
    def load_model(self):
        """Load language model and tokenizer for inference."""
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
                device_map={"": self.device.index if self.device.type == 'cuda' else 'cpu'},
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN")
            )
            
            if self.custom_chat_template:
                template = self.custom_chat_template
            else:
                template_map = {
                    "llama": "llama-3.1",
                    "phi": "phi-3",
                    "phi-4": "phi-4",
                    "mistral": "mistral",
                    "qwen": "qwen2.5"
                }
                template = template_map.get(self.model_type, detect_chat_template(self.model_path))
            
            self.tokenizer = get_chat_template(
                tokenizer=self.tokenizer,
                chat_template=template
            )
            
            FastLanguageModel.for_inference(self.model)
            
            logger.info(f"✓ Model loaded successfully")
            
            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"GPU Memory: {allocated:.2f}GB allocated")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {e}")
    
    def new_conversation(self, title: str = None) -> str:
        """Initialize a new conversation session."""
        session_id = self.memory.create_session(title)
        self.current_conversation = []
        self.clinical_note_included = False
        self.generation_count = 0  # Reset performance counter
        return session_id

    def load_conversation(self, session_id: str) -> bool:
        """Load an existing conversation session into active context."""
        session = self.memory.get_session(session_id)
        if not session:
            return False

        self.current_conversation = []
        for msg in session.messages:
            self.current_conversation.append({
                "role": msg.role,
                "content": msg.content
            })

        self.memory.current_session_id = session_id
        self.clinical_note_included = bool(session.clinical_note and session.clinical_note.strip())
        self.generation_count = 0  # Reset performance counter
        return True

    def reset_conversation(self):
        """Reset current conversation context to empty state."""
        self.current_conversation = []
        self.clinical_note_included = False
        self.generation_count = 0  # Reset performance counter
    
    def generate_response(
        self,
        user_message: str,
        clinical_note: str = None,
        session_id: str = None,
        max_new_tokens: int = 2048
    ) -> Tuple[str, float]:
        """
        Generate AI response to user message.
        
        CRITICAL: Clinical note ONLY included if:
        1. Not None
        2. Not empty string
        3. Has actual content after stripping
        4. Hasn't been included yet
        """
        import time
        start_time = time.time()
        
        # CRITICAL CHECK: Only include clinical note if it has REAL content
        actual_message = user_message
        
        # Strict validation: clinical note must exist AND have content
        has_clinical_content = (
            clinical_note is not None and 
            isinstance(clinical_note, str) and 
            len(clinical_note.strip()) > 0
        )
        
        if has_clinical_content and not self.clinical_note_included:
            actual_message = f"CLINICAL NOTE:\n{clinical_note.strip()}\n\n{user_message}"
            self.clinical_note_included = True
            logger.info("✓ Clinical note included with first message")
            
            if session_id:
                self.memory.update_clinical_note(session_id, clinical_note.strip())
        
        # Add user message to conversation
        self.current_conversation.append({"role": "user", "content": actual_message})
        
        if session_id:
            self.memory.add_message(session_id, "user", user_message)
        
        try:
            # NO system prompt - pure conversation
            formatted_text = self.tokenizer.apply_chat_template(
                self.current_conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            max_input_length = self.max_seq_length - max_new_tokens
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.95,
                    use_cache=True  # CRITICAL: Enable KV cache for faster generation
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # CRITICAL: Delete tensors immediately to free memory
            del inputs
            del outputs

            # Track generation count
            self.generation_count += 1

            # PERFORMANCE OPTIMIZATION: Only clear cache periodically (expensive ~10-50ms)
            # Clear every 10th generation OR when conversation gets very long
            # This reduces overhead while preventing memory accumulation
            should_clear_cache = (
                self.device.type == 'cuda' and (
                    self.generation_count % 10 == 0 or  # Every 10 generations
                    len(self.current_conversation) > 50  # Or if conversation very long
                )
            )

            if should_clear_cache:
                torch.cuda.empty_cache()
                logger.debug(f"Cache cleared at generation {self.generation_count}")

            self.current_conversation.append({"role": "assistant", "content": response})

            if session_id:
                self.memory.add_message(session_id, "assistant", response)

            inference_time = time.time() - start_time

            return response, inference_time
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, time.time() - start_time
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retrieve comprehensive model and system information."""
        info = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "temperature": self.temperature,
            "max_seq_length": self.max_seq_length,
            "device": str(self.device),
            "loaded": self.model is not None
        }
        
        if self.device.type == 'cuda' and self.model:
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1024**3
        
        if self.csv_manager:
            info["csv_data"] = self.csv_manager.get_info()
        
        return info


# ============================================================================
# Gradio Interface Builder
# ============================================================================

def create_professional_interface(engine: ClinicalConversationEngine) -> gr.Blocks:
    """Create compact, professional Gradio interface with proper component sizing."""
    
    EXAMPLE_CASES = {
        "Severe Malnutrition - Cerebral Palsy": """7 yo male with cerebral palsy presents for malnutrition evaluation. Growth parameters: Height 115 cm (5th percentile, z-score -1.65), Weight 16.2 kg (below 3rd percentile, z-score -2.3), BMI 12.3 kg/m² (z-score -2.5). Previous visit 3 months ago: Weight 17.1 kg (z-score -1.8), BMI 13.1 kg/m² (z-score -1.9). Patient has difficulty with oral intake due to dysphagia. Reports decreased appetite and feeding takes 45+ minutes. Physical exam: thin appearance, decreased subcutaneous fat, muscle wasting noted in extremities. Pale conjunctiva. Labs: Albumin 3.2 g/dL (low), Hemoglobin 10.1 g/dL (anemia). Assessment: Moderate malnutrition with progressive weight loss and z-score decline over past 3 months.""",
        
        "No Malnutrition - Down Syndrome": """11 yo girl with Down syndrome presents for nutrition follow-up. Growth parameters: Height 142 cm (10th percentile), Weight 45 kg (75th percentile), BMI 22.3 kg/m² (90th percentile). Previous visit 6 months ago showed BMI 21.8 kg/m² (85th percentile). Patient eating well with good appetite. No concerns about nutritional intake. Physical exam: well-appearing, alert, interactive, no signs of malnutrition. Assessment: Nutritional status appropriate for age and condition.""",
        
        "Moderate Malnutrition - Cystic Fibrosis": """5 yo female with cystic fibrosis presents for nutrition assessment. Height 102 cm (15th percentile, z-score -1.0), Weight 14.8 kg (5th percentile, z-score -1.7), BMI 14.2 kg/m² (z-score -1.4). Six months ago: Weight 15.5 kg (z-score -1.2), BMI 14.8 kg/m² (z-score -1.0). Patient has frequent respiratory infections affecting appetite. Pancreatic insufficiency on enzyme replacement. Physical exam shows mild muscle wasting, adequate energy level. Labs: Albumin 3.5 g/dL (normal), Hemoglobin 11.5 g/dL."""
    }
    
    initial_session_id = engine.new_conversation("Initial Consultation")
    
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1800px;
        margin: auto;
    }
    
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .status-loaded {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 13px;
    }
    
    .status-empty {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 13px;
    }
    
    .compact-section {
        margin: 10px 0;
        padding: 8px;
        background: #f8f9fa;
        border-radius: 6px;
    }
    
    .footer-box {
        text-align: center;
        padding: 15px;
        margin-top: 20px;
        border-top: 2px solid #e0e0e0;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Pediatric Malnutrition Assessment", theme=gr.themes.Soft()) as interface:
        
        session_state = gr.State({
            "session_id": initial_session_id,
            "title": "Initial Consultation",
            "created_at": datetime.now().isoformat(),
            "message_count": 0,
            "total_inference_time": 0.0,
            "clinical_note_loaded": False
        })
        
        doc_panel_visible = gr.State(True)
        
        gr.HTML("""
            <div class="header-title">
                <h1>🏥 Pediatric Malnutrition Assessment System</h1>
                <h3>Clinical Decision Support</h3>
                <p style="margin-top: 8px; font-size: 13px;">
                    ✨ Pure chat mode - Clinical notes optional
                </p>
            </div>
        """)
        
        with gr.Tabs() as tabs:
            
            with gr.TabItem("💬 Clinical Consultation", id=0):
                
                with gr.Row():
                    with gr.Column(scale=1, visible=True) as clinical_doc_column:
                        gr.Markdown("### 📋 Clinical Documentation")
                        
                        clinical_note_status = gr.Markdown(
                            '<div class="status-empty">💬 Pure chat mode - No clinical note loaded</div>'
                        )
                        
                        with gr.Accordion("📂 Load from CSV Database", open=False):
                            csv_status = gr.Markdown(
                                "No CSV file loaded" if not engine.csv_manager or engine.csv_manager.df is None else 
                                f"✅ {len(engine.csv_manager.patient_ids)} patients loaded",
                                elem_classes="compact-section"
                            )
                            
                            csv_file_upload = gr.File(
                                label="Upload CSV File",
                                file_types=[".csv"],
                                type="filepath",
                                scale=1,
                                height=80
                            )
                            
                            csv_path_input = gr.Textbox(
                                label="Or enter CSV path",
                                placeholder="/path/to/file.csv",
                                lines=1,
                                scale=1
                            )
                            
                            with gr.Row():
                                deid_col_input = gr.Textbox(
                                    label="ID Column",
                                    value="DEID",
                                    scale=1
                                )
                                text_col_input = gr.Textbox(
                                    label="Text Column",
                                    value="txt",
                                    scale=1
                                )
                            
                            load_csv_btn = gr.Button("📥 Load CSV", size="sm", variant="primary")
                            
                            patient_dropdown = gr.Dropdown(
                                choices=engine.csv_manager.get_patient_ids() if engine.csv_manager and engine.csv_manager.df is not None else [],
                                label="Select Patient",
                                scale=1
                            )
                            
                            load_patient_btn = gr.Button("👤 Load Patient Note", size="sm")
                        
                        gr.Markdown("---")
                        
                        clinical_note = gr.Textbox(
                            label="Clinical Note (Optional)",
                            placeholder="Clinical note will be sent ONLY when loaded and ONLY with first message...",
                            lines=10,
                            max_lines=15
                        )
                        
                        with gr.Accordion("📝 Example Cases", open=False):
                            example_dropdown = gr.Dropdown(
                                choices=list(EXAMPLE_CASES.keys()),
                                label="Select Example",
                                scale=1
                            )
                            load_example_btn = gr.Button("📥 Load Example", size="sm")
                        
                        with gr.Row():
                            clear_note_btn = gr.Button("🗑️ Clear Note", size="sm", variant="secondary")
                            clear_note_and_reset_btn = gr.Button("🔄 Clear Note & Reset Chat", size="sm", variant="stop")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            toggle_doc_btn = gr.Button(
                                "◀️ Hide Panel",
                                size="sm"
                            )
                        
                        chatbot = gr.Chatbot(
                            label="Clinical Conversation",
                            height=520,
                            type="messages",
                            show_copy_button=True
                        )
                        
                        with gr.Row():
                            user_input = gr.Textbox(
                                label="Your Message",
                                placeholder="Ask anything - works without clinical notes!",
                                lines=2,
                                scale=5,
                                show_label=False,
                                interactive=True  # Always keep interactive
                            )

                            with gr.Column(scale=1, min_width=120):
                                with gr.Row():
                                    send_btn = gr.Button("📤 Send", variant="primary", size="sm")
                                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="sm")

                        with gr.Row():
                            new_conversation_btn = gr.Button("🆕 New", size="sm")
                            reset_btn = gr.Button("🔄 Reset", size="sm")
                            save_btn = gr.Button("💾 Save", size="sm")

                        status_message = gr.Markdown("")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**Messages**")
                                message_count_display = gr.Number(value=0, label="", interactive=False, show_label=False)
                            
                            with gr.Column(scale=1):
                                gr.Markdown("**Avg Time (s)**")
                                avg_time_display = gr.Number(value=0.0, label="", interactive=False, show_label=False, precision=2)
            
            with gr.TabItem("📂 Session Management", id=1):
                
                gr.Markdown("## 💾 Conversation History")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        session_info_display = gr.Markdown("")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🆕 New Session")
                        
                        new_session_title = gr.Textbox(
                            label="Title",
                            placeholder="Session title...",
                            lines=1
                        )
                        
                        create_session_btn = gr.Button("✨ Create", variant="primary")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 📁 Load Session")
                        
                        refresh_sessions_btn = gr.Button("🔄 Refresh", size="sm")
                        
                        saved_sessions_dropdown = gr.Dropdown(
                            label="Saved Sessions",
                            choices=[],
                            scale=1
                        )
                        
                        load_session_btn = gr.Button("📥 Load", variant="primary")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 📤 Export")
                        
                        export_format = gr.Radio(
                            choices=["Text (TXT)", "JSON"],
                            value="Text (TXT)",
                            label="Format"
                        )
                        
                        export_btn = gr.Button("📄 Export")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Statistics")
                        
                        stats_display = gr.JSON(
                            label="Memory Stats",
                            value=engine.memory.get_memory_stats()
                        )
                        
                        gr.Markdown("### 📝 Export Preview")
                        
                        export_output = gr.Textbox(
                            label="Exported Content",
                            lines=18,
                            max_lines=25,
                            interactive=False,
                            show_copy_button=True
                        )
                
                session_status = gr.Markdown("")
            
            with gr.TabItem("ℹ️ System Info", id=2):
                
                gr.Markdown("## 🖥️ System Status")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🤖 Model")
                        model_info_display = gr.JSON(
                            label="Configuration",
                            value=engine.get_model_info()
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 💾 Memory")
                        memory_stats_display = gr.JSON(
                            label="Statistics",
                            value=engine.memory.get_memory_stats()
                        )
                
                refresh_info_btn = gr.Button("🔄 Refresh")
                
                gr.Markdown("""
                ## 📖 Version 1.2.0

                **Key Features:**
                - ✅ Clinical text NEVER passed when empty
                - ✅ LLM works in pure chat mode without clinical context
                - ✅ Clinical note only sent when explicitly loaded
                - ✅ Clear notes functionality - clears note and resets chat
                - ✅ New conversation properly clears previous clinical notes
                - ⏹️ **NEW**: Stop button to cancel generation mid-process
                - ⌨️ **NEW**: Type next question while current one is processing
                - ⚡ **NEW**: Non-blocking interface - no waiting for responses

                **Usage:**
                1. **Pure Chat**: Start chatting immediately without any clinical note
                2. **With Clinical Note**: Load a note - it will be sent only with your first message
                3. **Clear Note**: Use "Clear Note" to remove note or "Clear Note & Reset Chat" to start fresh
                4. **New Conversation**: Clinical notes are NOT carried over to new conversations
                5. **Stop Generation**: Click ⏹️ Stop button to cancel long-running generation
                6. **Queue Questions**: Type next question while waiting for response
                """)
        
        gr.HTML("""
            <div class="footer-box">
                <p style="font-size: 13px; color: #555; margin: 4px 0;">
                    <strong>Frederick Gyasi</strong> | gyasi@musc.edu
                </p>
                <p style="font-size: 12px; color: #666; margin: 4px 0;">
                    Medical University of South Carolina
                </p>
                <p style="font-size: 11px; color: #999; margin: 8px 0 0 0;">
                    © 2025 | MIT License
                </p>
            </div>
        """)
        
        # ====================================================================
        # EVENT HANDLERS
        # ====================================================================
        
        def toggle_documentation_panel(current_visibility):
            new_visibility = not current_visibility
            button_text = "▶️ Show" if not new_visibility else "◀️ Hide"
            return new_visibility, gr.update(visible=new_visibility), gr.update(value=button_text)
        
        def update_session_display(session_state_dict):
            msg_count = session_state_dict.get("message_count", 0)
            total_time = session_state_dict.get("total_inference_time", 0.0)
            avg_time = (total_time / msg_count) if msg_count > 0 else 0.0
            
            clinical_loaded = session_state_dict.get("clinical_note_loaded", False)
            
            display_text = f"""
**Session Info**

- **Title**: {session_state_dict.get('title', 'Unknown')}
- **ID**: `{session_state_dict.get('session_id', 'N/A')[:12]}...`
- **Created**: {session_state_dict.get('created_at', 'N/A')[:16]}
- **Messages**: {msg_count}
- **Clinical Note**: {"✅ Loaded" if clinical_loaded else "❌ Not loaded"}
            """
            
            return display_text, msg_count, avg_time
        
        def update_clinical_note_status(clinical_note_text):
            """CRITICAL: Update status indicator with strict validation."""
            # Strict check: must be non-None, non-empty string with actual content
            has_content = (
                clinical_note_text is not None and 
                isinstance(clinical_note_text, str) and 
                len(clinical_note_text.strip()) > 0
            )
            
            if has_content:
                return '<div class="status-loaded">✅ Clinical note loaded - will be sent with first message</div>'
            else:
                return '<div class="status-empty">💬 Pure chat mode - No clinical note loaded</div>'
        
        def process_user_message(user_msg, clinical_note_text, chat_history, session_state_dict):
            if not user_msg.strip():
                return chat_history, "", session_state_dict, "⚠️ Please enter a message", *update_session_display(session_state_dict), gr.update()
            
            session_id = session_state_dict["session_id"]
            
            # CRITICAL: Only pass clinical note if it has REAL content and this is first message
            clinical_note_to_pass = None
            
            # Strict validation
            has_clinical_content = (
                clinical_note_text is not None and 
                isinstance(clinical_note_text, str) and 
                len(clinical_note_text.strip()) > 0
            )
            
            # Only pass if: has content AND first message in conversation
            if has_clinical_content and len(chat_history) == 0:
                clinical_note_to_pass = clinical_note_text.strip()
                session_state_dict["clinical_note_loaded"] = True
                logger.info(f"✓ Passing clinical note to LLM (length: {len(clinical_note_to_pass)})")
            else:
                logger.info("✓ Pure chat mode - no clinical note passed")
            
            try:
                response, inference_time = engine.generate_response(
                    user_msg, 
                    clinical_note=clinical_note_to_pass,
                    session_id=session_id
                )
                
                chat_history.append({"role": "user", "content": user_msg})
                chat_history.append({"role": "assistant", "content": response})
                
                session_state_dict["message_count"] = len(chat_history) // 2
                session_state_dict["total_inference_time"] += inference_time
                
                status_msg = f"✅ Response in {inference_time:.2f}s"
                
                return (chat_history, "", session_state_dict, status_msg, 
                        *update_session_display(session_state_dict),
                        update_clinical_note_status(clinical_note_text))
            
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                logger.error(f"Error in process_user_message: {e}", exc_info=True)
                return (chat_history, user_msg, session_state_dict, error_msg, 
                        *update_session_display(session_state_dict),
                        update_clinical_note_status(clinical_note_text))
        
        def create_new_conversation(title_text):
            """Create new conversation and CLEAR clinical note."""
            if not title_text.strip():
                title_text = f"Consultation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            session_id = engine.new_conversation(title_text)
            
            new_state = {
                "session_id": session_id,
                "title": title_text,
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
                "total_inference_time": 0.0,
                "clinical_note_loaded": False
            }
            
            status_msg = f"✅ New: {title_text}"
            
            # CRITICAL: Clear clinical note when creating new conversation
            return [], "", new_state, status_msg, *update_session_display(new_state), update_clinical_note_status("")
        
        def reset_current_conversation(session_state_dict):
            """Reset conversation but KEEP clinical note."""
            engine.reset_conversation()
            session_id = engine.new_conversation(f"Reset {datetime.now().strftime('%H:%M')}")
            
            session_state_dict["session_id"] = session_id
            session_state_dict["message_count"] = 0
            session_state_dict["total_inference_time"] = 0.0
            session_state_dict["clinical_note_loaded"] = False
            
            status_msg = "🔄 Reset - conversation cleared, clinical note kept"
            
            return ([], session_state_dict, status_msg, 
                    *update_session_display(session_state_dict))
        
        def clear_clinical_note():
            """Clear clinical note only."""
            logger.info("✓ Clinical note cleared")
            return "", update_clinical_note_status(""), "🗑️ Clinical note cleared"
        
        def clear_note_and_reset_chat(session_state_dict):
            """Clear clinical note AND reset conversation."""
            engine.reset_conversation()
            session_id = engine.new_conversation(f"Fresh Start {datetime.now().strftime('%H:%M')}")
            
            session_state_dict["session_id"] = session_id
            session_state_dict["message_count"] = 0
            session_state_dict["total_inference_time"] = 0.0
            session_state_dict["clinical_note_loaded"] = False
            
            logger.info("✓ Clinical note cleared and conversation reset")
            status_msg = "🔄 Clinical note cleared and conversation reset"
            
            return ([], "", session_state_dict, status_msg,
                    *update_session_display(session_state_dict),
                    update_clinical_note_status(""))
        
        def save_current_session(session_state_dict):
            session_id = session_state_dict["session_id"]
            file_path = engine.memory.save_session(session_id)
            
            if file_path:
                return f"✅ Saved: {file_path}"
            else:
                return "❌ Save failed"
        
        def load_example_case(example_name):
            if example_name:
                content = EXAMPLE_CASES.get(example_name, "")
                return content, update_clinical_note_status(content)
            return "", update_clinical_note_status("")
        
        def load_csv_file(csv_file, csv_path, deid_col, text_col):
            file_to_load = csv_file if csv_file else csv_path
            
            if not file_to_load:
                return "⚠️ No file provided", gr.Dropdown(choices=[])
            
            if not engine.csv_manager:
                engine.csv_manager = CSVPatientDataManager()
            
            success, message = engine.csv_manager.load_csv(file_to_load, deid_col, text_col)
            patient_ids = engine.csv_manager.get_patient_ids() if success else []
            
            return message, gr.Dropdown(choices=patient_ids)
        
        def load_patient_note(patient_id):
            if not engine.csv_manager or not patient_id:
                return "⚠️ Select patient", update_clinical_note_status("")
            
            note = engine.csv_manager.get_clinical_note(patient_id)
            logger.info(f"✓ Loaded patient note (length: {len(note)})")
            return note, update_clinical_note_status(note)
        
        def get_saved_sessions_list():
            sessions = engine.memory.get_all_saved_sessions()
            return [f"{s['title']} - {s['created_at'][:10]} ({s['message_count']} msgs)" for s in sessions]
        
        def load_saved_session(selected_session_str, session_state_dict):
            if not selected_session_str:
                return ([], "", session_state_dict, "⚠️ No session selected", 
                        *update_session_display(session_state_dict), gr.update())
            
            sessions = engine.memory.get_all_saved_sessions()
            selected_session = None
            
            for session in sessions:
                if session['title'] in selected_session_str:
                    selected_session = session
                    break
            
            if not selected_session:
                return ([], "", session_state_dict, "❌ Not found", 
                        *update_session_display(session_state_dict), gr.update())
            
            session_id = engine.memory.load_session(selected_session['file_path'])
            
            if not session_id:
                return ([], "", session_state_dict, "❌ Load failed", 
                        *update_session_display(session_state_dict), gr.update())
            
            engine.load_conversation(session_id)
            session = engine.memory.get_session(session_id)
            
            chat_history = []
            for msg in session.messages:
                chat_history.append({"role": msg.role, "content": msg.content})
            
            # Strict check for clinical note
            clinical_note_loaded = (
                session.clinical_note is not None and
                isinstance(session.clinical_note, str) and
                len(session.clinical_note.strip()) > 0
            )
            
            session_state_dict = {
                "session_id": session_id,
                "title": session.title,
                "created_at": session.created_at,
                "message_count": len([m for m in session.messages if m.role == "user"]),
                "total_inference_time": 0.0,
                "clinical_note_loaded": clinical_note_loaded
            }
            
            status_msg = f"✅ Loaded: {session.title}"
            
            return (chat_history, session.clinical_note, session_state_dict, status_msg, 
                    *update_session_display(session_state_dict),
                    update_clinical_note_status(session.clinical_note))
        
        def export_current_session(export_format_choice, session_state_dict):
            session_id = session_state_dict["session_id"]
            
            if export_format_choice == "Text (TXT)":
                return engine.memory.export_session_txt(session_id)
            else:
                session = engine.memory.get_session(session_id)
                if session:
                    return json.dumps(session.to_dict(), indent=2)
                return "Session not found"
        
        def refresh_system_info():
            return engine.get_model_info(), engine.memory.get_memory_stats()
        
        # ====================================================================
        # CONNECT EVENTS
        # ====================================================================
        
        toggle_doc_btn.click(
            fn=toggle_documentation_panel,
            inputs=[doc_panel_visible],
            outputs=[doc_panel_visible, clinical_doc_column, toggle_doc_btn]
        )

        # CRITICAL: Make send events cancellable and non-blocking
        # This allows:
        # 1. Stop button to cancel generation
        # 2. User to type next question while current one is processing
        send_event = send_btn.click(
            fn=process_user_message,
            inputs=[user_input, clinical_note, chatbot, session_state],
            outputs=[chatbot, user_input, session_state, status_message,
                    session_info_display, message_count_display, avg_time_display,
                    clinical_note_status],
            concurrency_limit=None  # Allow concurrent operations
        )

        submit_event = user_input.submit(
            fn=process_user_message,
            inputs=[user_input, clinical_note, chatbot, session_state],
            outputs=[chatbot, user_input, session_state, status_message,
                    session_info_display, message_count_display, avg_time_display,
                    clinical_note_status],
            concurrency_limit=None  # Allow concurrent operations
        )

        # Stop button cancels both send and submit events
        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[send_event, submit_event]
        )
        
        new_conversation_btn.click(
            fn=lambda: create_new_conversation(f"Chat {datetime.now().strftime('%H:%M')}"),
            inputs=[],
            outputs=[chatbot, clinical_note, session_state, status_message,
                    session_info_display, message_count_display, avg_time_display,
                    clinical_note_status]
        )
        
        reset_btn.click(
            fn=reset_current_conversation,
            inputs=[session_state],
            outputs=[chatbot, session_state, status_message,
                    session_info_display, message_count_display, avg_time_display]
        )
        
        save_btn.click(
            fn=save_current_session,
            inputs=[session_state],
            outputs=[status_message]
        )
        
        clear_note_btn.click(
            fn=clear_clinical_note,
            outputs=[clinical_note, clinical_note_status, status_message]
        )
        
        clear_note_and_reset_btn.click(
            fn=clear_note_and_reset_chat,
            inputs=[session_state],
            outputs=[chatbot, clinical_note, session_state, status_message,
                    session_info_display, message_count_display, avg_time_display,
                    clinical_note_status]
        )
        
        clinical_note.change(
            fn=update_clinical_note_status,
            inputs=[clinical_note],
            outputs=[clinical_note_status]
        )
        
        load_example_btn.click(
            fn=load_example_case,
            inputs=[example_dropdown],
            outputs=[clinical_note, clinical_note_status]
        )
        
        example_dropdown.change(
            fn=load_example_case,
            inputs=[example_dropdown],
            outputs=[clinical_note, clinical_note_status]
        )
        
        load_csv_btn.click(
            fn=load_csv_file,
            inputs=[csv_file_upload, csv_path_input, deid_col_input, text_col_input],
            outputs=[csv_status, patient_dropdown]
        )
        
        load_patient_btn.click(
            fn=load_patient_note,
            inputs=[patient_dropdown],
            outputs=[clinical_note, clinical_note_status]
        )
        
        create_session_btn.click(
            fn=create_new_conversation,
            inputs=[new_session_title],
            outputs=[chatbot, clinical_note, session_state, session_status,
                    session_info_display, message_count_display, avg_time_display,
                    clinical_note_status]
        )
        
        refresh_sessions_btn.click(
            fn=get_saved_sessions_list,
            outputs=[saved_sessions_dropdown]
        )
        
        load_session_btn.click(
            fn=load_saved_session,
            inputs=[saved_sessions_dropdown, session_state],
            outputs=[chatbot, clinical_note, session_state, session_status,
                    session_info_display, message_count_display, avg_time_display,
                    clinical_note_status]
        )
        
        export_btn.click(
            fn=export_current_session,
            inputs=[export_format, session_state],
            outputs=[export_output]
        )
        
        refresh_info_btn.click(
            fn=refresh_system_info,
            outputs=[model_info_display, memory_stats_display]
        )
        
        interface.load(
            fn=update_session_display,
            inputs=[session_state],
            outputs=[session_info_display, message_count_display, avg_time_display]
        )
    
    return interface


# ============================================================================
# Main Application Entry Point
# ============================================================================

def main():
    """Main application entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Pediatric Malnutrition Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python malnutrition_chat_v1_0_0_fixed.py --model ./trained_model
  python malnutrition_chat_v1_0_0_fixed.py --model ./model --csv_path ./patients.csv
  python malnutrition_chat_v1_0_0_fixed.py --model ./model --temperature 0.5 --port 8080
        """
    )
    
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--model_type", choices=["llama", "phi", "phi-4", "mistral", "qwen"])
    parser.add_argument("--custom_chat_template", help="Custom chat template")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--cuda_device", type=int)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--auth", nargs=2, metavar=("USER", "PASS"))
    parser.add_argument("--history_dir", default="./conversation_history")
    parser.add_argument("--csv_path", help="CSV with patient data")
    parser.add_argument("--deid_col", default="DEID")
    parser.add_argument("--text_col", default="txt")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("\n" + "="*80)
    print("  PEDIATRIC MALNUTRITION ASSESSMENT")
    print("  ✨ Clinical notes NEVER passed when empty")
    print("  ✨ Pure chat mode works perfectly")
    print("  ✨ New conversations properly isolated")
    print("="*80)
    
    try:
        csv_manager = None
        if args.csv_path:
            logger.info(f"Loading CSV: {args.csv_path}")
            csv_manager = CSVPatientDataManager(args.csv_path, args.deid_col, args.text_col)
        
        logger.info("Initializing engine...")
        engine = ClinicalConversationEngine(
            model_path=args.model,
            model_type=args.model_type,
            custom_chat_template=args.custom_chat_template,
            cuda_device=args.cuda_device,
            temperature=args.temperature,
            max_seq_length=args.max_length,
            csv_manager=csv_manager
        )
        
        engine.memory = ConversationMemoryManager(storage_dir=args.history_dir)
        
        logger.info("Loading model...")
        engine.load_model()
        logger.info("✓ Model loaded")
        
        logger.info("Building interface...")
        interface = create_professional_interface(engine)
        logger.info("✓ Interface ready")

        # CRITICAL: Enable queue for stop button and concurrent operations
        interface.queue(
            default_concurrency_limit=None  # Allow unlimited concurrent operations
        )
        logger.info("✓ Queue enabled - Stop button and concurrent typing active")

        print("\n" + "="*80)
        print("  🚀 LAUNCHING")
        print("="*80)
        print(f"\n🌐 http://localhost:{args.port}")
        if args.share:
            print(f"🌍 Public link will be generated")
        print(f"\n🛑 Ctrl+C to stop")
        print(f"\n✨ Features: Stop button + Type while processing")
        print("="*80 + "\n")

        auth_tuple = tuple(args.auth) if args.auth else None

        interface.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            auth=auth_tuple,
            show_error=True
        )
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped")
        logger.info("Shutdown by user")
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cleared")
        print("\n✓ Cleanup complete")


if __name__ == "__main__":
    main()