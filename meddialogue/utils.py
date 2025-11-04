"""
Utility Functions for MedDialogue
==================================

Text processing, formatting, parsing, and weighted selection utilities.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.1 - Added weighted_random_choice for format selection
"""

import re
import json
import logging
import random
from typing import Dict, List, Any, Optional, TypeVar
import xml.etree.ElementTree as ET

from .config import OutputFormat

logger = logging.getLogger(__name__)

T = TypeVar('T')


def weighted_random_choice(choices: List[T], weights: Optional[Dict[T, float]] = None) -> T:
    """
    Select random choice from list with optional weights.
    
    Args:
        choices: List of choices to select from
        weights: Optional dict mapping choices to weights (probabilities)
                If None or empty, uses uniform selection
                Weights must sum to 1.0
    
    Returns:
        Randomly selected choice based on weights
    
    Examples:
        # Uniform selection (no weights)
        >>> weighted_random_choice([1, 2, 3])
        2
        
        # Weighted selection
        >>> weighted_random_choice([1, 2, 3], {1: 0.5, 2: 0.3, 3: 0.2})
        1  # 50% chance
        
        # With OutputFormat enums
        >>> formats = [OutputFormat.TEXT, OutputFormat.JSON]
        >>> weights = {OutputFormat.TEXT: 0.7, OutputFormat.JSON: 0.3}
        >>> weighted_random_choice(formats, weights)
        OutputFormat.TEXT  # 70% chance
    
    Raises:
        ValueError: If choices is empty
        ValueError: If weights provided but don't cover all choices
        ValueError: If weights don't sum to ~1.0
    """
    if not choices:
        raise ValueError("choices list cannot be empty")
    
    # Uniform selection if no weights provided
    if not weights:
        return random.choice(choices)
    
    # Validate weights cover all choices
    missing_weights = set(choices) - set(weights.keys())
    if missing_weights:
        raise ValueError(
            f"Weights must be provided for all choices. "
            f"Missing weights for: {missing_weights}"
        )
    
    # Validate weights sum to ~1.0
    total = sum(weights[choice] for choice in choices)
    if not 0.99 <= total <= 1.01:
        raise ValueError(
            f"Weights must sum to 1.0, got {total:.4f}. "
            f"Weights: {weights}"
        )
    
    # Perform weighted random selection
    r = random.random()
    cumulative = 0.0
    
    for choice in choices:
        cumulative += weights[choice]
        if r <= cumulative:
            return choice
    
    # Fallback (should never reach here due to normalization)
    return choices[-1]


def preprocess_clinical_text(text: str, max_chars: int = 100000) -> str:
    """
    Preprocess clinical text by removing special tokens and normalizing whitespace.
    
    Args:
        text: Raw clinical text
        max_chars: Maximum character length
    
    Returns:
        Cleaned clinical text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove common model special tokens
    special_tokens = [
        '<s>', '</s>', '<pad>', '</pad>', '<eos>', '<bos>', '<|endoftext|>',
        '<|begin_of_text|>', '<|end_of_text|>', '<|eot_id|>',
        '[INST]', '[/INST]', '<|im_start|>', '<|im_end|>'
    ]
    
    for token in special_tokens:
        text = text.replace(token, ' ')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*\.\s*\.', ' ', text)
    text = re.sub(r'\s*-\s*-\s*-', ' ', text)
    
    # Truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars]
        logger.warning(f"Text truncated to {max_chars} characters")
    
    return text.strip()


def format_output(output_data: Dict[str, Any], output_format: OutputFormat, 
                 output_fields: List[str]) -> str:
    """
    Format output data in specified format.
    
    Args:
        output_data: Dictionary of field values
        output_format: Desired output format
        output_fields: List of fields to include
    
    Returns:
        Formatted output string
    """
    if output_format == OutputFormat.TEXT:
        return format_text_output(output_data, output_fields)
    elif output_format == OutputFormat.JSON:
        return format_json_output(output_data, output_fields)
    elif output_format == OutputFormat.XML:
        return format_xml_output(output_data, output_fields)
    elif output_format == OutputFormat.MARKDOWN:
        return format_markdown_output(output_data, output_fields)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def format_text_output(output_data: Dict[str, Any], output_fields: List[str]) -> str:
    """Format output as natural language text."""
    parts = []
    
    if "reasoning" in output_data and output_data["reasoning"]:
        parts.append(f"**Clinical Reasoning:**\n{output_data['reasoning']}")
    
    for field in output_fields:
        if field == "reasoning":
            continue
        
        value = output_data.get(field, "")
        if value:
            field_title = field.replace("_", " ").title()
            parts.append(f"\n**{field_title}:** {value}")
    
    return "\n".join(parts)


def format_json_output(output_data: Dict[str, Any], output_fields: List[str]) -> str:
    """Format output as JSON."""
    json_data = {}
    
    for field in output_fields:
        if field in output_data:
            json_data[field] = output_data[field]
    
    return json.dumps(json_data, indent=2, ensure_ascii=False)


def format_xml_output(output_data: Dict[str, Any], output_fields: List[str]) -> str:
    """
    Format output as XML with proper escaping.
    
    Handles multiline content with CDATA sections.
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<assessment>']
    
    for field in output_fields:
        if field in output_data:
            value = str(output_data[field])
            
            # Use CDATA for multiline content
            if '\n' in value:
                lines.append(f'  <{field}><![CDATA[{value}]]></{field}>')
            else:
                # Proper XML entity encoding for single-line
                value_escaped = (value.replace('&', '&amp;')
                                     .replace('<', '&lt;')
                                     .replace('>', '&gt;')
                                     .replace('"', '&quot;')
                                     .replace("'", '&apos;'))
                lines.append(f'  <{field}>{value_escaped}</{field}>')
    
    lines.append('</assessment>')
    return '\n'.join(lines)


def format_markdown_output(output_data: Dict[str, Any], output_fields: List[str]) -> str:
    """Format output as Markdown with proper heading hierarchy."""
    parts = ["# Clinical Assessment\n"]
    
    for field in output_fields:
        if field in output_data and output_data[field]:
            field_title = field.replace("_", " ").title()
            value = str(output_data[field]).strip()
            
            parts.append(f"## {field_title}\n")
            parts.append(f"{value}\n")
    
    return "\n".join(parts)


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from model response with multiple fallback strategies.
    
    Strategies:
    1. Direct parsing
    2. Extract from markdown code blocks
    3. Find balanced JSON object in text
    4. Remove common prefixes/suffixes
    
    Args:
        response: Model response text
    
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_block_pattern, response, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Find JSON object in text (with balanced braces)
    try:
        cleaned = _extract_json_object(response)
        if cleaned:
            return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Remove common prefixes/suffixes
    prefixes = ["Here's the JSON:", "Here is the JSON:", "JSON output:", 
                "Response:", "```json", "```"]
    
    for prefix in prefixes:
        if prefix in response:
            try:
                cleaned = response.split(prefix, 1)[1].strip()
                cleaned = cleaned.rstrip('`').strip()
                return json.loads(cleaned)
            except (json.JSONDecodeError, IndexError):
                continue
    
    logger.warning("Failed to parse JSON from response")
    return None


def _extract_json_object(text: str) -> Optional[str]:
    """
    Extract balanced JSON object from text.
    
    Handles nested objects and arrays properly by counting braces.
    
    Args:
        text: Text containing JSON object
    
    Returns:
        Extracted JSON string or None
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                
                if brace_count == 0:
                    return text[start_idx:i+1]
    
    return None


def parse_xml_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse XML from model response.
    
    Strategies:
    1. Extract from markdown code blocks
    2. Extract XML document
    3. Find assessment tag
    
    Handles CDATA sections properly.
    
    Args:
        response: Model response text
    
    Returns:
        Parsed XML as dict or None if parsing fails
    """
    # Strategy 1: Extract from markdown code blocks
    xml_block_pattern = r'```(?:xml)?\s*(<?xml.*?</assessment>)\s*```'
    matches = re.findall(xml_block_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        response = matches[0]
    
    # Strategy 2: Extract XML document
    xml_doc_pattern = r'(<\?xml.*?</assessment>)'
    matches = re.findall(xml_doc_pattern, response, re.DOTALL)
    
    if matches:
        response = matches[0]
    
    # Strategy 3: Find assessment tag
    if '<assessment>' in response and '</assessment>' in response:
        start = response.find('<assessment>')
        end = response.find('</assessment>') + len('</assessment>')
        response = response[start:end]
    
    try:
        root = ET.fromstring(response)
        
        # Convert to dict
        result = {}
        for child in root:
            # Handle CDATA sections and regular text
            if child.text:
                result[child.tag] = child.text.strip()
            else:
                result[child.tag] = ""
        
        return result
        
    except ET.ParseError as e:
        logger.warning(f"Failed to parse XML: {e}")
        return None


def validate_medical_terminology(text: str, valid_terms: List[str]) -> Dict[str, Any]:
    """
    Validate presence of medical terminology in text.
    
    Args:
        text: Text to validate
        valid_terms: List of valid medical terms
    
    Returns:
        Validation results dict with found/missing terms and coverage
    """
    text_lower = text.lower()
    found_terms = [term for term in valid_terms if term.lower() in text_lower]
    missing_terms = [term for term in valid_terms if term.lower() not in text_lower]
    
    return {
        "is_valid": len(found_terms) > 0,
        "found_terms": found_terms,
        "missing_terms": missing_terms,
        "coverage": len(found_terms) / len(valid_terms) if valid_terms else 0.0
    }


def add_typo(text: str) -> str:
    """
    Add a realistic typo to text for robustness training.
    
    Common medical term typos and general word typos.
    
    Args:
        text: Original text
    
    Returns:
        Text with one typo added
    """
    typo_patterns = [
        # Common medical term typos
        ("patient", "patien"),
        ("assessment", "assesment"),
        ("severity", "severety"),
        ("recommend", "recomend"),
        ("diagnosis", "diagnois"),
        ("evidence", "evidance"),
        ("analysis", "anaylsis"),
        ("treatment", "treatement"),
        ("malnutrition", "malnutrtion"),
        ("nutritional", "nutitional"),
        
        # Common word typos
        ("the ", "teh "),
        ("with ", "wiht "),
        ("this ", "htis "),
        ("provide", "provde"),
        ("complete", "compelte"),
    ]
    
    # Apply one random typo
    for correct, typo in typo_patterns:
        if correct in text.lower():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(correct), re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(typo, text, count=1)
                break
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()


def extract_field_value(text: str, field_name: str) -> Optional[str]:
    """
    Extract field value from structured text.
    
    Args:
        text: Structured text
        field_name: Field name to extract
    
    Returns:
        Extracted value or None
    """
    patterns = [
        rf"{field_name}:\s*(.+?)(?:\n|$)",
        rf"\*\*{field_name}\*\*:\s*(.+?)(?:\n|$)",
        rf"{field_name.replace('_', ' ')}:\s*(.+?)(?:\n|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def truncate_text(text: str, max_length: int = 512, add_ellipsis: bool = True) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        add_ellipsis: Add "..." at end
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    
    if add_ellipsis:
        truncated += "..."
    
    return truncated


def clean_model_response(response: str) -> str:
    """
    Clean model response of common artifacts.
    
    Removes thinking tags, prefixes, etc.
    
    Args:
        response: Raw model response
    
    Returns:
        Cleaned response
    """
    # Remove common prefixes
    prefixes = [
        "Here is the assessment:",
        "Here's the assessment:",
        "Assessment:",
        "Response:",
        "Output:",
    ]
    
    for prefix in prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    # Remove trailing markers
    response = response.rstrip('`').strip()
    
    return response