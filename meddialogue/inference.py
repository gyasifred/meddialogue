"""
Inference Pipeline for MedDialogue - FIXED
==========================================

Fixed format instruction handling and response parsing.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.1 - Bug Fixes
"""
import logging
from typing import List, Dict, Any, Optional, Union
import torch
import re

from .config import OutputFormat, TaskConfig
from .utils import parse_json_response, parse_xml_response, preprocess_clinical_text

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Inference pipeline for trained MedDialogue models."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        task_config: TaskConfig,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task_config = task_config
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.model.eval()
        
        logger.info("Inference pipeline initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Max new tokens: {max_new_tokens}")
    
    def infer(
        self,
        clinical_note: str,
        question: Optional[str] = None,
        output_format: OutputFormat = OutputFormat.TEXT,
        return_full_response: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Run inference on single clinical note.

        Each call is independent - no history carried over between calls.
        """
        clinical_note = preprocess_clinical_text(clinical_note)

        # FIX: Improved format instruction generation
        if question is None:
            question = self._generate_default_question(output_format)
        else:
            # FIX: Ensure format instruction is in question if not already present
            question = self._ensure_format_instruction(question, output_format)

        # Build conversation (fresh for each call - no history)
        conversation = [
            {"role": "system", "content": self.task_config.get_system_prompt()},
            {"role": "user", "content": f"{question}\n\nCLINICAL NOTE:\n{clinical_note}"}
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        # Clear any cached states before generation to ensure independence
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache for speed, but it's per-generation only
            )

        # Cleanup: Delete inputs to free memory immediately
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = self._extract_assistant_response(full_output, prompt)
        
        # FIX: Improved format parsing with validation
        parsed_response = self._parse_and_validate_format(response, output_format)
        
        if return_full_response:
            return {
                "response": parsed_response,
                "format": output_format.value,
                "question": question,
                "raw_response": response,  # Include raw for debugging
                "metadata": {
                    "input_length": len(clinical_note),
                    "output_length": len(str(parsed_response)),
                    "model": self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else "unknown"
                }
            }
        
        return parsed_response
    
    def _generate_default_question(self, output_format: OutputFormat) -> str:
        """Generate default question based on format - IMPROVED."""
        task_name = self.task_config.task_name.replace('_', ' ')
        
        if output_format == OutputFormat.JSON:
            return (
                f"Provide a complete {task_name} assessment. "
                f"Return your response as a valid JSON object with the following fields: "
                f"{', '.join(self.task_config.output_fields)}."
            )
        elif output_format == OutputFormat.XML:
            return (
                f"Provide a complete {task_name} assessment. "
                f"Return your response as a valid XML document with root element <assessment> "
                f"containing elements for: {', '.join(self.task_config.output_fields)}."
            )
        elif output_format == OutputFormat.MARKDOWN:
            return (
                f"Provide a complete {task_name} assessment. "
                f"Return your response in Markdown format with clear sections for: "
                f"{', '.join(self.task_config.output_fields)}."
            )
        else:  # TEXT
            return f"What is the {task_name} status? Provide a complete assessment."
    
    def _ensure_format_instruction(self, question: str, output_format: OutputFormat) -> str:
        """Ensure format instruction is in question - NEW."""
        question_lower = question.lower()
        
        # Check if format already specified
        format_keywords = {
            OutputFormat.JSON: ['json'],
            OutputFormat.XML: ['xml'],
            OutputFormat.MARKDOWN: ['markdown', 'md']
        }
        
        if output_format in format_keywords:
            for keyword in format_keywords[output_format]:
                if keyword in question_lower:
                    return question  # Already has format instruction
        
        # Add format instruction
        if output_format == OutputFormat.JSON:
            return f"{question} Provide your response in JSON format."
        elif output_format == OutputFormat.XML:
            return f"{question} Provide your response in XML format."
        elif output_format == OutputFormat.MARKDOWN:
            return f"{question} Provide your response in Markdown format."
        else:
            return question
    
    def _parse_and_validate_format(self, response: str, output_format: OutputFormat) -> Union[str, Dict[str, Any]]:
        """Parse and validate response format - IMPROVED."""
        if output_format == OutputFormat.JSON:
            parsed = parse_json_response(response)
            if parsed:
                # Validate has expected fields
                missing_fields = [f for f in self.task_config.output_fields if f not in parsed]
                if missing_fields:
                    logger.warning(f"JSON missing fields: {missing_fields}")
                return parsed
            else:
                logger.warning("Failed to parse JSON, returning raw text")
                return response
        
        elif output_format == OutputFormat.XML:
            parsed = parse_xml_response(response)
            if parsed:
                # Validate has expected fields
                missing_fields = [f for f in self.task_config.output_fields if f not in parsed]
                if missing_fields:
                    logger.warning(f"XML missing fields: {missing_fields}")
                return parsed
            else:
                logger.warning("Failed to parse XML, returning raw text")
                return response
        
        elif output_format == OutputFormat.MARKDOWN:
            # Validate Markdown structure
            if self._validate_markdown(response):
                return response
            else:
                logger.warning("Markdown validation failed, returning raw text")
                return response
        
        else:  # TEXT
            return response
    
    def _validate_markdown(self, response: str) -> bool:
        """Validate Markdown has proper structure - NEW."""
        # Check for headers
        has_headers = bool(re.search(r'^#+\s+', response, re.MULTILINE))
        
        # Check for expected field names in headers
        response_lower = response.lower()
        found_fields = sum(1 for field in self.task_config.output_fields 
                          if field.replace('_', ' ').lower() in response_lower)
        
        # Valid if has headers and at least half the expected fields
        return has_headers and found_fields >= len(self.task_config.output_fields) // 2
    
    def infer_multi_turn(
        self,
        clinical_note: str,
        questions: List[str],
        output_formats: Optional[List[OutputFormat]] = None,
        return_full_response: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Run multi-turn inference on single clinical note.

        Matches training format where:
        - Turn 1: Includes clinical note
        - Turn 2+: Uses conversation context (no repeated note)

        Args:
            clinical_note: Clinical text
            questions: List of questions for each turn
            output_formats: Optional list of output formats (one per question)
            return_full_response: Whether to return full response dict

        Returns:
            List of responses (one per turn)
        """
        clinical_note = preprocess_clinical_text(clinical_note)

        if output_formats is None:
            output_formats = [OutputFormat.TEXT] * len(questions)

        if len(output_formats) != len(questions):
            raise ValueError(f"Number of output_formats ({len(output_formats)}) must match number of questions ({len(questions)})")

        # Build cumulative conversation
        conversation = [
            {"role": "system", "content": self.task_config.get_system_prompt()}
        ]

        responses = []

        for turn_idx, (question, output_format) in enumerate(zip(questions, output_formats)):
            # Ensure format instruction
            question = self._ensure_format_instruction(question, output_format)

            # Turn 1: Include clinical note (matches training)
            if turn_idx == 0:
                user_content = f"{question}\n\nCLINICAL NOTE:\n{clinical_note}"
            else:
                # Turn 2+: Only question (matches training)
                user_content = question

            # Add user message to conversation
            conversation.append({"role": "user", "content": user_content})

            # Apply chat template to current conversation
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )

            # Cleanup
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self._extract_assistant_response(full_output, prompt)

            # Parse and validate format
            parsed_response = self._parse_and_validate_format(response, output_format)

            # Add assistant response to conversation for next turn
            conversation.append({"role": "assistant", "content": response})

            if return_full_response:
                responses.append({
                    "response": parsed_response,
                    "format": output_format.value,
                    "question": question,
                    "raw_response": response,
                    "turn": turn_idx + 1,
                    "metadata": {
                        "output_length": len(str(parsed_response))
                    }
                })
            else:
                responses.append(parsed_response)

        return responses

    def batch_infer(
        self,
        clinical_notes: List[str],
        questions: Optional[List[str]] = None,
        output_format: OutputFormat = OutputFormat.TEXT
    ) -> List[Union[str, Dict[str, Any]]]:
        """Run inference on batch of clinical notes (single-turn)."""
        logger.info(f"Running batch inference on {len(clinical_notes)} examples")

        if questions is None:
            questions = [None] * len(clinical_notes)

        results = []
        for i, (note, question) in enumerate(zip(clinical_notes, questions)):
            try:
                result = self.infer(note, question, output_format)
                results.append(result)
            except Exception as e:
                logger.error(f"Inference failed for note {i}: {e}")
                results.append(None)

        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Batch inference completed: {success_count}/{len(results)} successful")

        return results
    
    def _extract_assistant_response(self, full_output: str, prompt: str) -> str:
        """Extract assistant response from full output."""
        if prompt in full_output:
            response = full_output.replace(prompt, "").strip()
        else:
            markers = ["assistant", "Assistant:", "ASSISTANT:", "<|assistant|>", "<|im_start|>assistant"]
            for marker in markers:
                if marker in full_output:
                    parts = full_output.split(marker)
                    response = parts[-1].strip()
                    break
            else:
                response = full_output
        
        # Clean up special tokens
        response = response.replace("<|im_end|>", "").replace("<|eot_id|>", "").strip()
        
        return response
    
    def interactive_session(self):
        """Start interactive inference session."""
        print(f"\n{'='*60}")
        print(f"MedDialogue Interactive Session")
        print(f"Task: {self.task_config.task_name}")
        print(f"{'='*60}\n")
        print("Commands:")
        print("  'quit' or 'exit' - Exit session")
        print("  'format <type>' - Change output format (text/json/xml/markdown)")
        print("\n")
        
        current_format = OutputFormat.TEXT
        
        while True:
            print("Enter clinical note (or command):")
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if user_input.lower().startswith('format '):
                format_str = user_input.split()[1].lower()
                try:
                    current_format = OutputFormat(format_str)
                    print(f"Output format changed to: {format_str}\n")
                except ValueError:
                    print(f"Invalid format. Use: text, json, xml, or markdown\n")
                continue
            
            if not user_input:
                print("Please enter a valid clinical note.\n")
                continue
            
            print("\nEnter question (or press Enter for default):")
            question = input("> ").strip()
            question = question if question else None
            
            print("\n" + "="*60)
            print(f"Response ({current_format.value}):")
            print("="*60)
            
            try:
                response = self.infer(user_input, question, current_format)
                if isinstance(response, dict):
                    import json
                    print(json.dumps(response, indent=2))
                else:
                    print(response)
            except Exception as e:
                print(f"Error: {e}")
            
            print("\n" + "="*60 + "\n")


class BatchInference:
    """Batch inference with progress tracking and error handling."""
    
    def __init__(self, inference_pipeline: InferencePipeline):
        self.pipeline = inference_pipeline
    
    def process_dataframe(
        self,
        df: Any,
        text_column: str,
        output_column: str = "prediction",
        batch_size: int = 8,
        output_format: OutputFormat = OutputFormat.TEXT
    ) -> Any:
        """Process DataFrame with batch inference."""
        from tqdm import tqdm
        
        logger.info(f"Processing DataFrame with {len(df)} rows")
        
        predictions = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Batch inference"):
            batch = df[text_column].iloc[i:i+batch_size].tolist()
            batch_results = self.pipeline.batch_infer(batch, output_format=output_format)
            predictions.extend(batch_results)
        
        df[output_column] = predictions
        
        success_count = sum(1 for p in predictions if p is not None)
        logger.info(f"Inference completed: {success_count}/{len(predictions)} successful")
        
        return df
    
    def process_csv(
        self,
        input_path: str,
        output_path: str,
        text_column: str,
        batch_size: int = 8,
        output_format: OutputFormat = OutputFormat.TEXT
    ):
        """Process CSV file with batch inference."""
        import pandas as pd
        
        logger.info(f"Loading CSV from: {input_path}")
        df = pd.read_csv(input_path)
        
        df = self.process_dataframe(df, text_column, batch_size=batch_size, output_format=output_format)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")