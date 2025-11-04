#!/usr/bin/env python3
"""
Data Preparation for MedDialogue
=================================

Question combination with semantic variation for single and multi-turn.
ONE comprehensive example per clinical note.
Optimized for modern LLMs with 16K-32K token context windows.
Balanced mix of single-turn and multi-turn conversations.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0

Key Features:
- 1:1 mapping: Each clinical note → ONE training example
- Semantic variation: Different questions for same fields
- Balanced single-turn and multi-turn conversations
- 16 question combination styles (7 grammatical + 9 logical reasoning)
- Intelligent field ordering for reasoning-oriented questions
- Context-aware question length (optimized for 16K-32K token models)
- Strategic typos for robustness
- Optional validation split
"""

import random
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
from datasets import Dataset

from .config import TaskConfig, OutputFormat, DataMultiplicationConfig
from .utils import preprocess_clinical_text, add_typo, weighted_random_choice

logger = logging.getLogger(__name__)

random.seed(42)

@dataclass
class ConversationExample:
    """
    Single conversation example.
    
    Attributes:
        conversation: List of turn dictionaries with 'role' and 'content'
        metadata: Additional information about the conversation
    """
    conversation: List[Dict[str, str]]
    metadata: Dict[str, Any]


def select_output_format(task_config: TaskConfig) -> OutputFormat:
    """
    Select output format using weighted ratios if provided.
    
    Args:
        task_config: TaskConfig with output_formats and optional output_format_ratios
    
    Returns:
        Selected OutputFormat enum
    """
    if not task_config.output_format_ratios:
        return random.choice(task_config.output_formats)
    
    weights = {}
    
    for fmt in task_config.output_formats:
        weight = task_config.output_format_ratios.get(fmt.value, 0.0)
        if weight > 0.0:
            weights[fmt] = weight
    
    if not weights:
        return random.choice(task_config.output_formats)
    
    available_formats = list(weights.keys())
    return weighted_random_choice(available_formats, weights)


def get_weighted_format_from_subset(
    task_config: TaskConfig,
    allowed_formats: list
) -> OutputFormat:
    """
    Select format from a subset of formats with proportional weights.
    
    Useful for multi-turn where different turns prefer different format types.
    
    Args:
        task_config: TaskConfig with format ratios
        allowed_formats: List of OutputFormat enums to choose from
    
    Returns:
        Selected OutputFormat from allowed_formats
    """
    if not allowed_formats:
        raise ValueError("allowed_formats cannot be empty")
    
    available = [f for f in allowed_formats if f in task_config.output_formats]
    
    if not available:
        return allowed_formats[0]
    
    if not task_config.output_format_ratios:
        return random.choice(available)
    
    weights = {}
    total_weight = 0.0
    
    for fmt in available:
        weight = task_config.output_format_ratios.get(fmt.value, 0.0)
        if weight > 0.0:
            weights[fmt] = weight
            total_weight += weight
    
    if total_weight == 0.0:
        return random.choice(available)
    
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    return weighted_random_choice(available, normalized_weights)


class QuestionCombiner:
    """
    Combines questions with 16 different styles and semantic variation.
    
    16 Combination Styles:
    
    Grammatical Styles (1-7) - Work with any question order:
    1. Sequential: "Q1? Q2? Q3?"
    2. Conjunctions: "Q1 and Q2 and Q3"
    3. Transitional: "Q1? Also, Q2. Additionally, Q3"
    4. Numbered: "1. Q1? 2. Q2? 3. Q3?"
    5. Natural flow: "Q1? I'd also like to know Q2"
    6. Directive: "Tell me Q1, explain Q2"
    7. Mixed: Random combination of above
    
    Logical/Reasoning Styles (8-16) - Require ordered questions:
    8. Causal: "Q1, which means Q2, therefore Q3"
    9. Conditional: "Q1? If so, then Q2. Otherwise, Q3"
    10. Prioritized: "Most importantly, Q1. Secondarily, Q2"
    11. Comparative: "Q1 versus Q2? Which is more appropriate?"
    12. Temporal: "Initially, Q1. Then Q2. Finally, Q3"
    13. Hierarchical: "At a high level, Q1. More specifically, Q2"
    14. Dialectical: "Q1? However, Q2. On balance, Q3"
    15. Exploratory: "Consider Q1. This raises the question: Q2"
    16. Dependency: "Q1, upon which depends Q2, which determines Q3"
    """
    
    def __init__(self, task_config: TaskConfig, mult_config: DataMultiplicationConfig):
        self.task_config = task_config
        self.mult_config = mult_config
        self.question_templates = task_config.question_templates
        self.output_fields = task_config.output_fields
        self.field_ordering = task_config.field_ordering
        
        self.questions_by_field = {}
        for field in self.output_fields:
            if field in self.question_templates:
                self.questions_by_field[field] = self.question_templates[field]
            else:
                self.questions_by_field[field] = [
                    f"What is the {field.replace('_', ' ')}?",
                    f"Tell me the {field.replace('_', ' ')}",
                    f"Provide the {field.replace('_', ' ')}"
                ]
        
        # Define style categories
        self.grammatical_styles = [
            self._combine_sequential,
            self._combine_conjunctions,
            self._combine_transitional,
            self._combine_numbered,
            self._combine_natural,
            self._combine_directive,
            self._combine_mixed
        ]
        
        self.logical_styles = [
            self._combine_causal,
            self._combine_conditional,
            self._combine_prioritized,
            self._combine_comparative,
            self._combine_temporal,
            self._combine_hierarchical,
            self._combine_dialectical,
            self._combine_exploratory,
            self._combine_dependency
        ]
        
        logger.info(f"QuestionCombiner initialized (v1.1.0)")
        logger.info(f"  Fields: {self.output_fields}")
        logger.info(f"  Total question variants: {sum(len(v) for v in self.questions_by_field.values())}")
        logger.info(f"  Grammatical styles: {len(self.grammatical_styles)}")
        logger.info(f"  Logical styles: {len(self.logical_styles)}")
        logger.info(f"  Logical style ratio: {mult_config.logical_style_ratio * 100:.0f}%")
    
    def combine_questions(
        self, 
        fields: List[str], 
        output_format: Optional[OutputFormat] = None,
        include_typo: bool = False,
        max_length: int = 8000
    ) -> Tuple[str, Dict[str, str]]:
        """
        Combine questions and return both combined text and field-to-question mapping.
        
        Returns:
            Tuple of (combined_question_text, field_to_question_dict)
            where field_to_question_dict maps field names to the actual questions selected
        """
        if len(fields) == 0:
            raise ValueError("Must specify at least one field")
        
        # Select questions for each field
        questions = []
        field_to_question = {}
        
        for field in fields:
            if field in self.questions_by_field:
                q = random.choice(self.questions_by_field[field])
                questions.append(q)
                field_to_question[field] = q
        
        if len(questions) == 0:
            raise ValueError(f"No questions for fields: {fields}")
        
        # Decide: grammatical or logical style?
        use_logical = (
            random.random() < self.mult_config.logical_style_ratio and 
            len(fields) >= 2  # Logical styles need at least 2 questions
        )
        
        if use_logical:
            # Order questions by field priority for logical styles
            ordered_fields = self._order_fields_by_priority(fields)
            ordered_questions = [field_to_question[f] for f in ordered_fields]
            style_func = random.choice(self.logical_styles)
            combined = style_func(ordered_questions)
        else:
            # Random order for grammatical styles
            random.shuffle(questions)
            style_func = random.choice(self.grammatical_styles)
            combined = style_func(questions)
        
        # Check length and retry if needed
        attempts = 0
        max_attempts = 5
        while len(combined) > max_length and attempts < max_attempts:
            if use_logical:
                style_func = random.choice(self.logical_styles)
                combined = style_func(ordered_questions)
            else:
                style_func = random.choice(self.grammatical_styles)
                combined = style_func(questions)
            attempts += 1
        
        if len(combined) > max_length:
            # Fallback: simple concatenation
            combined = " ".join(questions[:max(1, len(questions)//2)])
        
        # Add typo if requested
        if include_typo and self.mult_config.include_typos:
            combined = add_typo(combined)
        
        # Add format instruction if needed
        if output_format and output_format != OutputFormat.TEXT:
            format_q = self._create_format_question(output_format)
            if len(combined) + len(format_q) + 1 <= max_length:
                combined = f"{combined} {format_q}"
            else:
                available = max_length - len(format_q) - 10
                if available > 50:
                    combined = combined[:available].rsplit(' ', 1)[0] + "... " + format_q
        
        return combined, field_to_question
    
    def _order_fields_by_priority(self, fields: List[str]) -> List[str]:
        """
        Order fields by priority defined in field_ordering.
        
        Lower priority number = higher priority (asked first).
        
        Args:
            fields: List of field names
        
        Returns:
            Ordered list of field names
        """
        return sorted(fields, key=lambda f: self.field_ordering.get(f, 999))
    
    # ============================================================================
    # GRAMMATICAL STYLES (1-7) - Work with any question order
    # ============================================================================
    
    def _combine_sequential(self, questions: List[str]) -> str:
        """Style 1: Sequential - Simple space separation."""
        return " ".join(questions)
    
    def _combine_conjunctions(self, questions: List[str]) -> str:
        """Style 2: Conjunctions - Using 'and' to connect."""
        if len(questions) == 1:
            return questions[0]
        elif len(questions) == 2:
            return f"{questions[0]} and {questions[1].lower()}"
        return ", ".join(questions[:-1]) + f", and {questions[-1].lower()}"
    
    def _combine_transitional(self, questions: List[str]) -> str:
        """Style 3: Transitional - Using transition words."""
        if len(questions) == 1:
            return questions[0]
        result = questions[0]
        transitions = ["Also,", "Additionally,", "Furthermore,", "Moreover,"]
        for q in questions[1:]:
            trans = random.choice(transitions)
            result += f" {trans} {q.lower()}"
        return result
    
    def _combine_numbered(self, questions: List[str]) -> str:
        """Style 4: Numbered - Numbered list format."""
        if len(questions) == 1:
            return questions[0]
        return " ".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    def _combine_natural(self, questions: List[str]) -> str:
        """Style 5: Natural flow - Conversational connectors."""
        if len(questions) == 1:
            return questions[0]
        result = questions[0]
        connectors = [
            lambda q: f" I'd also like to know {q.lower()}",
            lambda q: f" Can you tell me {q.lower()}",
            lambda q: f" Please include {q.lower()}",
            lambda q: f" And {q.lower()}",
        ]
        for q in questions[1:]:
            connector = random.choice(connectors)
            result += connector(q)
        return result
    
    def _combine_directive(self, questions: List[str]) -> str:
        """Style 6: Directive - Command form."""
        if len(questions) == 1:
            return questions[0]
        directives = []
        for q in questions:
            q_clean = q.rstrip('?.')
            if q_clean.lower().startswith(('what is', 'what are')):
                q_clean = "Tell me " + q_clean[8:]
            elif q_clean.lower().startswith(('how', 'why', 'when')):
                q_clean = "Explain " + q_clean
            directives.append(q_clean)
        return ", ".join(directives) + "."
    
    def _combine_mixed(self, questions: List[str]) -> str:
        """Style 7: Mixed - Combination of styles."""
        if len(questions) <= 2:
            return self._combine_sequential(questions)
        mid = len(questions) // 2
        first_half = self._combine_conjunctions(questions[:mid])
        second_half = self._combine_transitional(questions[mid:])
        return f"{first_half} {second_half}"
    
    # ============================================================================
    # LOGICAL/REASONING STYLES (8-16) - Require ordered questions
    # ============================================================================
    
    def _combine_causal(self, questions: List[str]) -> str:
        """Style 8: Causal - Cause-and-effect chain."""
        if len(questions) == 1:
            return questions[0]
        
        connectors = ["which means", "therefore", "which indicates", "suggesting that", "implying"]
        result = questions[0]
        for i, q in enumerate(questions[1:]):
            connector = connectors[i % len(connectors)]
            result += f", {connector} {q.lower()}"
        return result
    
    def _combine_conditional(self, questions: List[str]) -> str:
        """Style 9: Conditional - If-then-else branching."""
        if len(questions) <= 1:
            return questions[0] if questions else ""
        elif len(questions) == 2:
            return f"{questions[0]} If so, then {questions[1].lower()}"
        else:
            return f"{questions[0]} If so, then {questions[1].lower()} Otherwise, {questions[2].lower()}"
    
    def _combine_prioritized(self, questions: List[str]) -> str:
        """Style 10: Prioritized - Ranked by importance."""
        if len(questions) == 1:
            return questions[0]
        
        priority_words = ["Most importantly,", "Secondarily,", "Additionally,", "Also,", "Furthermore,"]
        parts = []
        for i, q in enumerate(questions):
            prefix = priority_words[min(i, len(priority_words)-1)]
            parts.append(f"{prefix} {q.lower()}")
        return " ".join(parts)
    
    def _combine_comparative(self, questions: List[str]) -> str:
        """Style 11: Comparative - Compare options."""
        if len(questions) <= 1:
            return questions[0] if questions else ""
        elif len(questions) == 2:
            return f"{questions[0]} versus {questions[1].lower()}? Which is more appropriate?"
        else:
            # For 3+ questions, compare in pairs
            comparisons = []
            for i in range(0, len(questions)-1, 2):
                if i+1 < len(questions):
                    comparisons.append(f"{questions[i]} versus {questions[i+1].lower()}")
            if len(questions) % 2 == 1:
                comparisons.append(questions[-1])
            return " Compare: " + "; ".join(comparisons) + "?"
    
    def _combine_temporal(self, questions: List[str]) -> str:
        """Style 12: Temporal - Time-ordered sequence."""
        if len(questions) == 1:
            return questions[0]
        
        temporal_markers = ["Initially,", "Then,", "Next,", "Subsequently,", "Finally,"]
        parts = []
        for i, q in enumerate(questions):
            marker = temporal_markers[min(i, len(temporal_markers)-1)]
            parts.append(f"{marker} {q.lower()}")
        return " ".join(parts)
    
    def _combine_hierarchical(self, questions: List[str]) -> str:
        """Style 13: Hierarchical - Abstract to specific."""
        if len(questions) == 1:
            return questions[0]
        
        levels = ["At a high level,", "More specifically,", "In detail,", "Drilling deeper,", "Precisely,"]
        parts = []
        for i, q in enumerate(questions):
            level = levels[min(i, len(levels)-1)]
            parts.append(f"{level} {q.lower()}")
        return " ".join(parts)
    
    def _combine_dialectical(self, questions: List[str]) -> str:
        """Style 14: Dialectical - Thesis, antithesis, synthesis."""
        if len(questions) <= 1:
            return questions[0] if questions else ""
        elif len(questions) == 2:
            return f"{questions[0]} However, {questions[1].lower()}"
        else:
            return f"{questions[0]} However, {questions[1].lower()} On balance, {questions[2].lower()}"
    
    def _combine_exploratory(self, questions: List[str]) -> str:
        """Style 15: Exploratory - Investigation chain."""
        if len(questions) == 1:
            return questions[0]
        
        connectors = [
            "Consider",
            "This raises the question:",
            "Furthermore,",
            "We should also investigate:",
            "Additionally,"
        ]
        
        parts = []
        for i, q in enumerate(questions):
            connector = connectors[min(i, len(connectors)-1)]
            if i == 0:
                parts.append(f"{connector} {q.lower()}")
            else:
                parts.append(f"{connector} {q.lower()}")
        return " ".join(parts)
    
    def _combine_dependency(self, questions: List[str]) -> str:
        """Style 16: Dependency - Dependency chain."""
        if len(questions) == 1:
            return questions[0]
        
        connectors = ["upon which depends", "which determines", "which affects", "leading to"]
        result = questions[0]
        for i, q in enumerate(questions[1:]):
            connector = connectors[i % len(connectors)]
            result += f", {connector} {q.lower()}"
        return result
    
    def _create_format_question(self, output_format: OutputFormat) -> str:
        """Create format instruction question."""
        templates = {
            OutputFormat.JSON: [
                "Return your answer in JSON format.",
                "Provide the response as JSON.",
                "Format your answer as JSON.",
                "Give me the output in JSON.",
                "Use JSON format.",
                "Structure as JSON."
            ],
            OutputFormat.XML: [
                "Return your answer in XML format.",
                "Provide the response as XML.",
                "Format your answer as XML.",
                "Give me the output in XML.",
                "Use XML format.",
                "Structure as XML."
            ],
            OutputFormat.MARKDOWN: [
                "Return your answer in Markdown format.",
                "Provide the response as Markdown.",
                "Format your answer as Markdown.",
                "Give me the output in Markdown.",
                "Use Markdown format.",
                "Structure as Markdown."
            ]
        }
        return random.choice(templates.get(output_format, ["Return your answer."]))


class ResponseFormatter:
    """
    Formats responses using actual question text as keys.
    """
    
    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config
    
    def format_response(
        self,
        data: Dict[str, Any],
        field_to_question: Dict[str, str],
        output_format: OutputFormat = OutputFormat.TEXT
    ) -> str:
        """
        Format response using question text as keys.
        
        Args:
            data: Dictionary mapping field names to answer content
            field_to_question: Dictionary mapping field names to actual questions asked
            output_format: Desired output format
        
        Returns:
            Formatted response string
        """
        response_data = {}
        for field, question in field_to_question.items():
            response_data[question] = data.get(field, "")
        
        if output_format == OutputFormat.JSON:
            return self._format_json(response_data)
        elif output_format == OutputFormat.XML:
            return self._format_xml(response_data, field_to_question)
        elif output_format == OutputFormat.MARKDOWN:
            return self._format_markdown(response_data)
        else:
            return self._format_text(response_data)
    
    def _format_text(self, data: Dict[str, Any]) -> str:
        if len(data) == 1:
            return str(list(data.values())[0])
        parts = []
        for question, answer in data.items():
            if answer:
                parts.append(f"{question}\n{answer}")
        return "\n\n".join(parts)
    
    def _format_json(self, data: Dict[str, Any]) -> str:
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _format_xml(self, data: Dict[str, Any], field_to_question: Dict[str, str]) -> str:
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<assessment>']
        
        field_names = {q: f for f, q in field_to_question.items()}
        
        for question, answer in data.items():
            field_name = field_names.get(question, 'field')
            safe_value = str(answer).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            lines.append(f'  <{field_name} question="{question}">')
            lines.append(f'    {safe_value}')
            lines.append(f'  </{field_name}>')
        lines.append('</assessment>')
        return '\n'.join(lines)
    
    def _format_markdown(self, data: Dict[str, Any]) -> str:
        lines = ["# Assessment\n"]
        for question, answer in data.items():
            lines.append(f"## {question}\n")
            lines.append(f"{answer}\n")
        return "\n".join(lines)


class DataPrep:
    """
    Main data preparation class with semantic variation and reasoning styles.
    
    Version 1.1.0 features:
    - 16 question combination styles (7 grammatical + 9 logical reasoning)
    - Intelligent field ordering for reasoning questions
    - Optional validation split
    - 1:1 row-to-example mapping
    """
    
    CONTEXT_WINDOW = 64000
    RESPONSE_RATIO = 0.25
    BUFFER_RATIO = 0.10
    MIN_QUESTION_LENGTH = 1000
    MAX_QUESTION_LENGTH = 8000
    
    def __init__(self, task_config: TaskConfig, mult_config: DataMultiplicationConfig):
        self.task_config = task_config
        self.mult_config = mult_config
        self.question_combiner = QuestionCombiner(task_config, mult_config)
        self.response_formatter = ResponseFormatter(task_config)
        
        logger.info("DataPrep v1.1.0 initialized")
        logger.info(f"  Context window: {self.CONTEXT_WINDOW} chars")
        logger.info(f"  Question length: {self.MIN_QUESTION_LENGTH}-{self.MAX_QUESTION_LENGTH} chars")
        logger.info(f"  Single-turn ratio: {mult_config.single_turn_ratio * 100:.0f}%")
        logger.info(f"  Multi-turn ratio: {(1 - mult_config.single_turn_ratio) * 100:.0f}%")
        logger.info(f"  Output fields: {len(task_config.output_fields)}")
        logger.info(f"  Max turns per conversation: {mult_config.max_multi_turns + 1}")
        logger.info(f"  Typo ratio: {mult_config.typo_ratio * 100:.1f}%")
        logger.info(f"  Logical style ratio: {mult_config.logical_style_ratio * 100:.0f}%")
        logger.info(f"  Validation split: {mult_config.validation_split * 100:.1f}%")
        logger.info(f"  Seed: 42")
    
    def prepare_data(
        self, 
        data: pd.DataFrame
    ) -> Tuple[List[ConversationExample], List[ConversationExample]]:
        """
        Prepare training and optional validation data.
        
        Args:
            data: Input DataFrame with clinical notes and output fields
            
        Returns:
            Tuple of (train_examples, val_examples)
            val_examples will be empty list if validation_split = 0.0
        """
        logger.info(f"Preparing {len(data)} examples with semantic variation (v1.1.0)")
        
        # Split data if validation requested
        if self.mult_config.validation_split > 0:
            train_data = data.sample(frac=1-self.mult_config.validation_split, random_state=42)
            val_data = data.drop(train_data.index)
            logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")
        else:
            train_data = data
            val_data = pd.DataFrame()
            logger.info("No validation split (validation_split = 0.0)")
        
        # Generate training examples
        train_examples = []
        for idx, row in train_data.iterrows():
            if random.random() < self.mult_config.single_turn_ratio:
                example = self._create_single_turn_example(row)
            else:
                example = self._create_multi_turn_example(row)
            train_examples.append(example)
        
        # Generate validation examples if needed
        val_examples = []
        if not val_data.empty:
            for idx, row in val_data.iterrows():
                if random.random() < self.mult_config.single_turn_ratio:
                    example = self._create_single_turn_example(row)
                else:
                    example = self._create_multi_turn_example(row)
                val_examples.append(example)
        
        # Statistics
        train_single = sum(1 for ex in train_examples if ex.metadata['type'] == 'single_turn')
        train_multi = len(train_examples) - train_single
        val_single = sum(1 for ex in val_examples if ex.metadata['type'] == 'single_turn')
        val_multi = len(val_examples) - val_single
        
        logger.info(f"Generated {len(train_examples)} train examples: "
                   f"{train_single} single-turn, {train_multi} multi-turn")
        if val_examples:
            logger.info(f"Generated {len(val_examples)} val examples: "
                       f"{val_single} single-turn, {val_multi} multi-turn")
        
        return train_examples, val_examples
    
    def _calculate_max_question_length(self, note_length: int) -> int:
        """Calculate maximum question length based on note length."""
        remaining_context = self.CONTEXT_WINDOW - note_length
        
        if remaining_context < self.CONTEXT_WINDOW * 0.2:
            max_q_len = max(500, int(remaining_context * 0.3))
            logger.warning(f"Note length {note_length} chars uses >80% of context. "
                         f"Question limited to {max_q_len} chars.")
            return max(self.MIN_QUESTION_LENGTH // 2, min(max_q_len, self.MIN_QUESTION_LENGTH))
        
        response_budget = int(remaining_context * self.RESPONSE_RATIO)
        buffer_budget = int(self.CONTEXT_WINDOW * self.BUFFER_RATIO)
        
        max_q_len = remaining_context - response_budget - buffer_budget
        max_q_len = max(self.MIN_QUESTION_LENGTH, min(max_q_len, self.MAX_QUESTION_LENGTH))
        
        logger.debug(f"Context calculation: note={note_length}, remaining={remaining_context}, "
                    f"response_budget={response_budget}, buffer={buffer_budget}, "
                    f"max_question={max_q_len}")
        
        return max_q_len
    
    def _create_single_turn_example(self, row: pd.Series) -> ConversationExample:
        """Create single-turn conversation example."""
        clinical_note = preprocess_clinical_text(row[self.task_config.input_field])
        output_data = {field: row.get(field, "") for field in self.task_config.output_fields}
        
        note_length = len(clinical_note)
        max_q_len = self._calculate_max_question_length(note_length)
        
        num_fields = len(self.task_config.output_fields)
        num_to_ask = random.randint(1, num_fields)
        requested_fields = random.sample(self.task_config.output_fields, num_to_ask)
        
        output_format = select_output_format(self.task_config)
        
        include_typo = (
            self.mult_config.include_typos and 
            random.random() < self.mult_config.typo_ratio
        )
        
        question, field_to_question = self.question_combiner.combine_questions(
            requested_fields,
            output_format,
            include_typo,
            max_length=max_q_len
        )
        
        response = self.response_formatter.format_response(
            output_data,
            field_to_question,
            output_format
        )
        
        conversation = [
            {"role": "user", "content": f"{question}\n\nCLINICAL NOTE:\n{clinical_note}"},
            {"role": "assistant", "content": response}
        ]
        
        return ConversationExample(
            conversation=conversation,
            metadata={
                "type": "single_turn",
                "num_fields_asked": num_to_ask,
                "total_fields": num_fields,
                "fields": requested_fields,
                "questions_used": list(field_to_question.values()),
                "format": output_format.value,
                "has_typo": include_typo,
                "note_length": note_length,
                "max_question_length": max_q_len,
                "context_utilization": round((note_length + max_q_len) / self.CONTEXT_WINDOW * 100, 1)
            }
        )
    
    def _create_multi_turn_example(self, row: pd.Series) -> ConversationExample:
        """Create multi-turn conversation example."""
        clinical_note = preprocess_clinical_text(row[self.task_config.input_field])
        output_data = {field: row.get(field, "") for field in self.task_config.output_fields}
        
        conversation = []
        available_fields = self.task_config.output_fields.copy()
        random.shuffle(available_fields)
        
        num_fields = len(available_fields)
        note_length = len(clinical_note)
        max_q_len = self._calculate_max_question_length(note_length)
        
        logger.debug(f"Creating multi-turn: note_length={note_length}, "
                    f"max_question_length={max_q_len}, fields={num_fields}")
        
        # Turn 1: First questions
        num_first = min(2, len(available_fields))
        first_fields = available_fields[:num_first]
        
        first_q, first_field_to_q = self.question_combiner.combine_questions(
            first_fields, 
            OutputFormat.TEXT, 
            include_typo=False,
            max_length=max_q_len
        )
        first_r = self.response_formatter.format_response(
            output_data, 
            first_field_to_q, 
            OutputFormat.TEXT
        )
        
        conversation.extend([
            {"role": "user", "content": f"{first_q}\n\nCLINICAL NOTE:\n{clinical_note}"},
            {"role": "assistant", "content": first_r}
        ])
        
        remaining_fields = available_fields[num_first:]
        
        # Turn 2: Follow-up (if fields remain)
        if len(remaining_fields) >= 2 and self.mult_config.include_followup_questions:
            num_second = min(2, len(remaining_fields))
            second_fields = remaining_fields[:num_second]
            
            json_xml_formats = [f for f in self.task_config.output_formats 
                              if f in [OutputFormat.JSON, OutputFormat.XML]]
            second_format = get_weighted_format_from_subset(self.task_config, json_xml_formats) if json_xml_formats else OutputFormat.TEXT
            
            second_q, second_field_to_q = self.question_combiner.combine_questions(
                second_fields,
                second_format,
                include_typo=False,
                max_length=max_q_len
            )
            second_r = self.response_formatter.format_response(
                output_data,
                second_field_to_q,
                second_format
            )
            
            conversation.extend([
                {"role": "user", "content": second_q},
                {"role": "assistant", "content": second_r}
            ])
            
            remaining_fields = remaining_fields[num_second:]
        
        # Additional turns (if fields remain)
        if remaining_fields and self.mult_config.include_followup_questions:
            max_additional_turns = min(
                self.mult_config.max_multi_turns - 1,
                len(remaining_fields)
            )
            
            for i in range(max_additional_turns):
                if i < len(remaining_fields):
                    field = remaining_fields[i]
                    
                    text_md_formats = [f for f in self.task_config.output_formats 
                                     if f in [OutputFormat.TEXT, OutputFormat.MARKDOWN]]
                    turn_format = get_weighted_format_from_subset(self.task_config, text_md_formats) if text_md_formats else OutputFormat.TEXT
                    
                    include_typo = (
                        self.mult_config.include_typos and 
                        random.random() < self.mult_config.typo_ratio
                    )
                    
                    followup_q, followup_field_to_q = self.question_combiner.combine_questions(
                        [field],
                        turn_format,
                        include_typo,
                        max_length=max_q_len
                    )
                    followup_r = self.response_formatter.format_response(
                        output_data,
                        followup_field_to_q,
                        turn_format
                    )
                    
                    conversation.extend([
                        {"role": "user", "content": followup_q},
                        {"role": "assistant", "content": followup_r}
                    ])
        
        num_turns = len(conversation) // 2
        formats_used = []
        if num_turns >= 1:
            formats_used.append("TEXT")
        if num_turns >= 2 and 'second_format' in locals():
            formats_used.append(second_format.value)
        if num_turns >= 3 and 'turn_format' in locals():
            formats_used.extend([turn_format.value for _ in range(num_turns - 2)])
        
        return ConversationExample(
            conversation=conversation,
            metadata={
                "type": "multi_turn_conversation",
                "num_turns": num_turns,
                "num_fields_covered": num_fields,
                "formats_used": formats_used,
                "note_length": note_length,
                "max_question_length": max_q_len,
                "context_utilization": round((note_length + max_q_len) / self.CONTEXT_WINDOW * 100, 1)
            }
        )
    
    def prepare_dataset(
        self, 
        examples: List[ConversationExample], 
        tokenizer: Any,
        num_proc: int = 4 
    ) -> Dataset:
        """
        Prepare dataset for training by tokenizing conversations.
        
        Args:
            examples: List of ConversationExample objects
            tokenizer: Tokenizer to use
            num_proc: Number of parallel processes
            
        Returns:
            Tokenized Dataset ready for training
        """
        logger.info(f"Preparing dataset from {len(examples)} examples")
        
        conv_list = [ex.conversation for ex in examples]
        dataset = Dataset.from_dict({'conversations': conv_list})
        
        def formatting_func(examples):
            texts = []
            for convo in examples['conversations']:
                try:
                    text = tokenizer.apply_chat_template(
                        convo,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Formatting failed: {e}")
                    texts.append("")
            return {'text': texts}
        
        actual_num_proc = min(num_proc, len(dataset))
        
        dataset = dataset.map(
            formatting_func, 
            batched=True, 
            num_proc=actual_num_proc,  
            remove_columns=['conversations']
        )
        
        dataset = dataset.filter(
            lambda x: isinstance(x.get("text", ""), str) and len(x["text"].strip()) >= 50,
            num_proc=actual_num_proc
        )
        
        single_turn_examples = [ex for ex in examples if ex.metadata['type'] == 'single_turn']
        multi_turn_examples = [ex for ex in examples if ex.metadata['type'] == 'multi_turn_conversation']
        
        logger.info(f"Final dataset: {len(dataset)} examples")
        logger.info(f"  Single-turn: {len(single_turn_examples)} ({len(single_turn_examples)/len(examples)*100:.1f}%)")
        logger.info(f"  Multi-turn: {len(multi_turn_examples)} ({len(multi_turn_examples)/len(examples)*100:.1f}%)")
        
        if single_turn_examples:
            avg_fields_single = sum(ex.metadata.get('num_fields_asked', 0) for ex in single_turn_examples) / len(single_turn_examples)
            logger.info(f"  Single-turn avg fields: {avg_fields_single:.1f}")
        
        if multi_turn_examples:
            avg_turns = sum(ex.metadata.get('num_turns', 0) for ex in multi_turn_examples) / len(multi_turn_examples)
            logger.info(f"  Multi-turn avg turns: {avg_turns:.1f}")
        
        avg_note_length = sum(ex.metadata.get('note_length', 0) for ex in examples) // len(examples)
        avg_context_util = sum(ex.metadata.get('context_utilization', 0) for ex in examples) / len(examples)
        
        logger.info(f"  Average note length: {avg_note_length} chars")
        logger.info(f"  Average context utilization: {avg_context_util:.1f}%")
        
        return dataset


# Backward compatibility alias
DataMultiplier = DataPrep