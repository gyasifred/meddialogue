#!/usr/bin/env python3
"""
Simple test script to verify the compound question fix.
This tests that answers appear in the same order as questions are asked.
"""
import sys
import random
random.seed(42)

# Add meddialogue to path
sys.path.insert(0, '/home/user/meddialogue')

from meddialogue.config import TaskConfig, ConversationConfig, OutputFormat
from meddialogue.data_prep import QuestionCombiner, ResponseFormatter

# Create test configuration
task_config = TaskConfig(
    task_name="test_task",
    input_field="note",
    output_fields=["field_a", "field_b", "field_c"],
    question_templates={
        "field_a": ["What is field A?", "Tell me field A"],
        "field_b": ["What is field B?", "Tell me field B"],
        "field_c": ["What is field C?", "Tell me field C"]
    }
)

conversation_config = ConversationConfig(
    single_turn_ratio=1.0,
    logical_style_ratio=0.0  # Use only grammatical styles to test shuffling
)

# Create combiner and formatter
combiner = QuestionCombiner(task_config, conversation_config)
formatter = ResponseFormatter(task_config)

# Test data
test_data = {
    "field_a": "Answer A",
    "field_b": "Answer B",
    "field_c": "Answer C"
}

print("Testing compound question order fix...")
print("=" * 80)

# Run 5 tests with different random combinations to see shuffling in action
for i in range(5):
    print(f"\nTest {i+1}:")
    print("-" * 80)

    # Combine questions
    combined_question, ordered_pairs = combiner.combine_questions(
        fields=["field_a", "field_b", "field_c"],
        output_format=OutputFormat.TEXT,
        include_typo=False,
        max_length=8000
    )

    print(f"Question: {combined_question}")
    print(f"\nOrdered pairs (field, question):")
    for j, (field, question) in enumerate(ordered_pairs, 1):
        print(f"  {j}. {field}: {question}")

    # Format response
    response = formatter.format_response(
        data=test_data,
        ordered_field_question_pairs=ordered_pairs,
        output_format=OutputFormat.TEXT
    )

    print(f"\nResponse:\n{response}")

    # Verify order matches
    response_lines = [line.strip() for line in response.split('\n') if line.strip()]
    questions_in_response = [response_lines[i] for i in range(0, len(response_lines), 2)]

    expected_questions = [q for _, q in ordered_pairs]

    print(f"\nVerification:")
    if questions_in_response == expected_questions:
        print("✓ PASS: Answer order matches question order!")
    else:
        print("✗ FAIL: Answer order does NOT match question order!")
        print(f"  Expected: {expected_questions}")
        print(f"  Got: {questions_in_response}")

print("\n" + "=" * 80)
print("All tests complete! The fix ensures answers are in the same order as questions.")
