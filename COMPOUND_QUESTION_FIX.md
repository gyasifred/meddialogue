# Compound Question Order Fix

## Issue Summary

**Problem**: When asking compound questions with conjunctions (e.g., "What is X? **and** what is Y? **and** what is Z?"), the SDK only answered the last part or provided answers in the wrong order.

**Root Cause**: Mismatch between question order and answer order during training data preparation.

**Date Fixed**: 2025-11-06

---

## Detailed Analysis

### The Bug

In `/meddialogue/data_prep.py`, the `QuestionCombiner.combine_questions()` method had a critical flaw:

1. **Lines 220-224**: Questions were selected for each field and stored in order:
   ```python
   for field in fields:  # e.g., [field_a, field_b, field_c]
       q = random.choice(self.questions_by_field[field])
       questions.append(q)  # [q_a, q_b, q_c]
       field_to_question[field] = q  # {field_a: q_a, field_b: q_b, ...}
   ```

2. **Line 243**: For grammatical styles, questions were **shuffled**:
   ```python
   random.shuffle(questions)  # Now: [q_c, q_a, q_b]
   combined = style_func(questions)  # "What is C? and what is A? and what is B?"
   ```

3. **Line 277**: Returned shuffled question but original-order mapping:
   ```python
   return combined, field_to_question  # ❌ field_to_question still in original order!
   ```

4. **Lines 541-561**: ResponseFormatter used the original field order:
   ```python
   for field, question in field_to_question.items():  # Original order!
       response_data[question] = data.get(field, "")
   ```

### Training Data Impact

During training, the model saw:

**User Question** (shuffled order):
```
What is diagnosis? and what is assessment? and what is treatment?
```

**Assistant Response** (original order):
```
What is assessment?
Answer to assessment

What is diagnosis?
Answer to diagnosis

What is treatment?
Answer to treatment
```

**Result**: The model learned to **ignore the question order** and always answer in field order, causing it to appear as if it only answered the "last" part or provided answers out of order.

---

## The Fix

### Changes Made

1. **Changed Return Type** (`data_prep.py:205`):
   - **Before**: `Tuple[str, Dict[str, str]]` (question text + unordered dict)
   - **After**: `Tuple[str, List[Tuple[str, str]]]` (question text + ordered pairs)

2. **Track Question Order** (`data_prep.py:236-258`):
   ```python
   # Track the actual order of questions as they appear in combined text
   ordered_field_question_pairs = []

   if use_logical:
       # For logical styles: use priority order
       ordered_fields = self._order_fields_by_priority(fields)
       ordered_field_question_pairs = [(f, field_to_question[f]) for f in ordered_fields]
   else:
       # For grammatical styles: shuffle pairs TOGETHER
       field_question_pairs = [(field, field_to_question[field]) for field in fields]
       random.shuffle(field_question_pairs)  # Shuffle pairs, not just questions
       ordered_field_question_pairs = field_question_pairs
   ```

3. **Preserve Order in Response** (`data_prep.py:543-573`):
   ```python
   def format_response(
       self,
       data: Dict[str, Any],
       ordered_field_question_pairs: List[Tuple[str, str]],  # ✓ Ordered!
       output_format: OutputFormat = OutputFormat.TEXT
   ) -> str:
       # Build response in same order as questions
       response_data = []
       for field, question in ordered_field_question_pairs:
           answer = data.get(field, "")
           response_data.append((question, answer))
       # ... format response preserving order
   ```

### Files Modified

- `/meddialogue/data_prep.py`:
  - `QuestionCombiner.combine_questions()` (lines 199-296)
  - `ResponseFormatter.format_response()` (lines 543-609)
  - `DataPrep._create_single_turn_example()` (lines 747-779)
  - `DataPrep._create_multi_turn_example()` (lines 801-883)

---

## Example: Before vs After

### Before Fix

**Training Data Generated**:
```
User: "What is diagnosis? and what is assessment? and what is treatment?"