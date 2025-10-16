# Logits and Logit Processors: The Foundation of Constrained Generation

## Executive Summary

Logits and logit processors are the fundamental mechanisms enabling constrained generation in modern LLMs. They provide the low-level control that powers libraries like vLLM and Guidance, allowing precise manipulation of model outputs without retraining or complex prompting. By operating at the token probability level, logit processors can enforce arbitrary constraints—from simple word choices to complex grammars—with mathematical guarantees and minimal performance overhead.

## What Are Logits?

### Mathematical Foundation

A **logit** is the raw, unnormalized output score from a neural network's final layer before applying softmax activation. In the context of LLMs:

```
Model Architecture:
Input → Embeddings → Transformer Layers → Linear Projection → Logits → Softmax → Probabilities
                                                               ↑
                                                    [vocab_size] dimensional vector
```

For a vocabulary of size V, the model outputs a V-dimensional vector of logits at each generation step:

```python
# Example: GPT-2 with 50,257 token vocabulary
logits = model.forward(input_ids)  # Shape: [batch_size, sequence_length, 50257]
```

### From Logits to Probabilities

The transformation from logits to probabilities uses the softmax function:

```
P(token_i) = exp(logit_i) / Σ_j exp(logit_j)
```

In code:
```python
import torch
import torch.nn.functional as F

def logits_to_probabilities(logits):
    """Convert raw logits to probability distribution."""
    # logits shape: [vocab_size]
    probabilities = F.softmax(logits, dim=-1)
    return probabilities

# Example
logits = torch.tensor([2.0, 1.0, 0.5, -1.0])  # Raw scores
probs = logits_to_probabilities(logits)
# Result: tensor([0.5761, 0.2119, 0.1285, 0.0286])
```

### Why Logits Matter

1. **Numerical Stability**: Working with logits avoids numerical issues from exponentials
2. **Efficient Computation**: Many operations are simpler in logit space
3. **Fine Control**: Direct manipulation before probability conversion
4. **Gradient Flow**: Better for backpropagation during training

## What Are Logit Processors?

### Core Concept

A **logit processor** is a function that modifies the logits vector before token sampling. It acts as a filter between the model's raw predictions and the final token selection:

```python
class LogitProcessor:
    """Abstract base class for all logit processors."""

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Previously generated tokens [batch_size, sequence_length]
            scores: Logits for next token [batch_size, vocab_size]

        Returns:
            Modified logits [batch_size, vocab_size]
        """
        raise NotImplementedError
```

### The Processing Pipeline

```
Raw Logits → Processor 1 → Processor 2 → ... → Processor N → Final Logits → Sampling
```

Each processor in the chain can:
- **Mask tokens**: Set certain logits to -∞
- **Bias tokens**: Add/subtract values from specific logits
- **Transform distributions**: Apply mathematical operations
- **Enforce constraints**: Implement business logic

## Types of Logit Processors

### 1. Temperature Scaling

Controls randomness in generation:

```python
class TemperatureLogitProcessor(LogitProcessor):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, input_ids, scores):
        if self.temperature != 1.0:
            scores = scores / self.temperature
        return scores

# temperature < 1.0: More focused (sharper distribution)
# temperature > 1.0: More random (flatter distribution)
# temperature = 0: Becomes greedy decoding (argmax)
```

### 2. Top-k Filtering

Keeps only the k most likely tokens:

```python
class TopKLogitProcessor(LogitProcessor):
    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, input_ids, scores):
        top_k = min(self.top_k, scores.size(-1))
        # Find threshold: kth largest value
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores[indices_to_remove] = float('-inf')
        return scores
```

### 3. Top-p (Nucleus) Sampling

Keeps smallest set of tokens with cumulative probability ≥ p:

```python
class TopPLogitProcessor(LogitProcessor):
    def __init__(self, top_p: float):
        self.top_p = top_p

    def __call__(self, input_ids, scores):
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        scores[indices_to_remove] = float('-inf')
        return scores
```

### 4. Repetition Penalty

Discourages repeating previously generated tokens:

```python
class RepetitionPenaltyLogitProcessor(LogitProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        for i in range(scores.shape[0]):
            for previous_token in set(input_ids[i].tolist()):
                # Reduce probability of tokens that have appeared
                if scores[i, previous_token] < 0:
                    scores[i, previous_token] *= self.penalty
                else:
                    scores[i, previous_token] /= self.penalty
        return scores
```

### 5. Constraint-Based Processors

#### Regex Constraint

```python
class RegexLogitProcessor(LogitProcessor):
    def __init__(self, pattern: str, tokenizer):
        self.pattern = re.compile(pattern)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        current_text = self.tokenizer.decode(input_ids[0])

        # Check each token in vocabulary
        for token_id in range(scores.shape[-1]):
            potential_text = current_text + self.tokenizer.decode([token_id])

            # If adding this token would violate the regex
            if not self.pattern.match(potential_text):
                scores[:, token_id] = float('-inf')

        return scores
```

#### JSON Schema Enforcement

```python
class JSONSchemaLogitProcessor(LogitProcessor):
    def __init__(self, schema: dict, tokenizer):
        self.schema = schema
        self.tokenizer = tokenizer
        self.validator = JSONValidator(schema)

    def __call__(self, input_ids, scores):
        current_text = self.tokenizer.decode(input_ids[0])

        for token_id in range(scores.shape[-1]):
            potential_text = current_text + self.tokenizer.decode([token_id])

            # Check if this would create invalid JSON
            if not self.validator.is_valid_prefix(potential_text):
                scores[:, token_id] = float('-inf')

        return scores
```

#### Multiple Choice Constraint

```python
class ChoiceLogitProcessor(LogitProcessor):
    def __init__(self, choices: List[str], tokenizer):
        self.choices = choices
        self.tokenizer = tokenizer
        # Pre-compute token IDs for each choice
        self.choice_tokens = [
            tokenizer.encode(choice, add_special_tokens=False)
            for choice in choices
        ]

    def __call__(self, input_ids, scores):
        # Mask all tokens except those that lead to valid choices
        mask = torch.full_like(scores, float('-inf'))

        for choice_tokens in self.choice_tokens:
            # Check if we can continue toward this choice
            position = len(input_ids[0]) % len(choice_tokens)
            if position < len(choice_tokens):
                valid_token = choice_tokens[position]
                mask[:, valid_token] = 0

        return scores + mask
```

## Logit Processing in vLLM

### Architecture

vLLM implements guided decoding through modular logit processors:

```python
# vLLM's guided decoding structure
class GuidedDecodingLogitProcessor:
    def __init__(self, guided_params):
        self.processors = []

        if guided_params.regex:
            self.processors.append(RegexLogitProcessor(guided_params.regex))
        if guided_params.json_schema:
            self.processors.append(JSONLogitProcessor(guided_params.json_schema))
        if guided_params.choice:
            self.processors.append(ChoiceLogitProcessor(guided_params.choice))

    def __call__(self, input_ids, scores):
        for processor in self.processors:
            scores = processor(input_ids, scores)
        return scores
```

### Backend Implementations

#### Outlines Backend

```python
# Simplified Outlines integration in vLLM
class OutlinesLogitProcessor:
    def __init__(self, regex_string, tokenizer):
        from outlines import generate

        # Build FSM from regex
        self.fsm = generate.fsm(regex_string, tokenizer)
        self.state = 0

    def __call__(self, input_ids, scores):
        # Get valid next tokens from FSM
        valid_tokens = self.fsm.get_valid_tokens(self.state)

        # Mask invalid tokens
        mask = torch.full_like(scores, float('-inf'))
        mask[:, valid_tokens] = 0

        return scores + mask
```

#### XGrammar Backend

```python
# XGrammar integration (C++ accelerated)
class XGrammarLogitProcessor:
    def __init__(self, grammar, tokenizer):
        import xgrammar

        # Compile grammar to efficient representation
        self.matcher = xgrammar.GrammarMatcher(grammar, tokenizer)

    def __call__(self, input_ids, scores):
        # Ultra-fast mask computation (~50μs)
        mask = self.matcher.compute_mask(input_ids)
        return scores * mask  # Efficient masking
```

### Performance Optimizations

1. **Caching**: Reuse computed masks for common prefixes
2. **Batch Processing**: Apply constraints across multiple sequences
3. **Async Compilation**: Prepare constraints while model runs
4. **Token Prefetching**: Predict likely tokens and precompute masks

## Logit Processing in Guidance

### The llguidance Engine

Guidance uses a Rust-based engine for ultra-fast constraint processing:

```rust
// Simplified llguidance core
pub struct Constraint {
    parser: Parser,
    tokenizer: TokenTrie,
}

impl Constraint {
    pub fn compute_mask(&mut self, tokens: &[TokenId]) -> TokenMask {
        // Parse current state
        let state = self.parser.parse(tokens);

        // Traverse token trie to find valid continuations
        let mut mask = TokenMask::new_empty();
        self.traverse_trie(state, &mut mask);

        mask
    }
}
```

### Token Masking Pipeline

```python
# Guidance's Python interface
from guidance import gen, select

# This translates to logit processing
with assistant():
    # Simple constraint
    lm += gen(regex=r"\d{3}-\d{4}")  # Phone number

    # Behind the scenes:
    # 1. Parse regex into grammar
    # 2. Track parser state
    # 3. Compute valid tokens at each step
    # 4. Apply mask to logits
```

### Advanced Features

#### Token Healing

```python
class TokenHealingProcessor(LogitProcessor):
    """Fix tokenization boundaries for better generation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        # Check if last token was partial
        last_token = input_ids[0, -1]
        decoded = self.tokenizer.decode([last_token])

        # If it's a partial token, boost completion tokens
        if decoded.startswith(" ") or decoded.endswith(" "):
            # Boost tokens that complete the word
            completion_tokens = self.find_completions(last_token)
            scores[:, completion_tokens] += 2.0

        return scores
```

#### Stateful Constraints

```python
class StatefulGrammarProcessor(LogitProcessor):
    """Maintains grammar state across generation."""

    def __init__(self, grammar):
        self.grammar = grammar
        self.state_stack = [StartState()]

    def __call__(self, input_ids, scores):
        current_state = self.state_stack[-1]

        # Get valid transitions from current state
        valid_tokens = self.grammar.get_valid_tokens(current_state)

        # Mask invalid tokens
        mask = torch.full_like(scores, float('-inf'))
        for token in valid_tokens:
            mask[:, token] = 0

        return scores + mask

    def update_state(self, chosen_token):
        """Update grammar state after token selection."""
        new_state = self.grammar.transition(self.state_stack[-1], chosen_token)
        self.state_stack.append(new_state)
```

## Implementation Examples

### 1. Building a Custom Logit Processor

```python
class CustomBusinessLogicProcessor(LogitProcessor):
    """Enforce business-specific constraints."""

    def __init__(self, config):
        self.banned_words = config.get('banned_words', [])
        self.required_format = config.get('format')
        self.max_length = config.get('max_length', 100)

    def __call__(self, input_ids, scores):
        # Ban specific tokens
        for word in self.banned_words:
            token_ids = tokenizer.encode(word)
            scores[:, token_ids] = float('-inf')

        # Enforce length limit
        if len(input_ids[0]) >= self.max_length:
            # Only allow EOS token
            eos_token = tokenizer.eos_token_id
            mask = torch.full_like(scores, float('-inf'))
            mask[:, eos_token] = 0
            return scores + mask

        return scores
```

### 2. Combining Multiple Processors

```python
class ProcessorPipeline:
    """Chain multiple logit processors."""

    def __init__(self, processors):
        self.processors = processors

    def __call__(self, input_ids, scores):
        for processor in self.processors:
            scores = processor(input_ids, scores)
        return scores

# Usage
pipeline = ProcessorPipeline([
    TemperatureLogitProcessor(0.7),
    TopKLogitProcessor(50),
    RepetitionPenaltyLogitProcessor(1.2),
    JSONSchemaLogitProcessor(schema, tokenizer),
])
```

### 3. Dynamic Constraint Selection

```python
class DynamicConstraintProcessor(LogitProcessor):
    """Switch constraints based on generation progress."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stages = {
            'greeting': ChoiceLogitProcessor(['Hello', 'Hi', 'Greetings']),
            'body': RegexLogitProcessor(r'[A-Za-z\s,.!?]+'),
            'closing': ChoiceLogitProcessor(['Regards', 'Sincerely', 'Best']),
        }

    def detect_stage(self, text):
        if len(text) < 10:
            return 'greeting'
        elif 'regards' in text.lower() or len(text) > 200:
            return 'closing'
        else:
            return 'body'

    def __call__(self, input_ids, scores):
        text = self.tokenizer.decode(input_ids[0])
        stage = self.detect_stage(text)
        processor = self.stages[stage]
        return processor(input_ids, scores)
```

### 4. Efficient Batch Processing

```python
class BatchedLogitProcessor(LogitProcessor):
    """Process multiple sequences with different constraints."""

    def __init__(self, constraints_per_sequence):
        self.constraints = constraints_per_sequence

    def __call__(self, input_ids, scores):
        batch_size = scores.shape[0]

        for i in range(batch_size):
            # Apply sequence-specific constraints
            constraint = self.constraints[i]
            scores[i] = constraint(input_ids[i:i+1], scores[i:i+1])

        return scores
```

## Performance Considerations

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Temperature scaling | O(V) | O(1) |
| Top-k filtering | O(V log k) | O(k) |
| Top-p filtering | O(V log V) | O(V) |
| Regex matching | O(V × L) | O(S) |
| Grammar parsing | O(V × G) | O(G × S) |
| Token masking | O(V) | O(V) |

Where:
- V = vocabulary size
- L = average token length
- S = state space size
- G = grammar complexity
- k = top-k value

### Optimization Strategies

#### 1. Preprocessing

```python
class OptimizedRegexProcessor(LogitProcessor):
    def __init__(self, pattern, tokenizer):
        self.pattern = pattern
        self.tokenizer = tokenizer

        # Precompute valid token sets for common prefixes
        self.cache = {}
        self.precompute_common_prefixes()

    def precompute_common_prefixes(self):
        # Cache valid tokens for frequent prefixes
        common_prefixes = ["", "The ", "A ", "I "]
        for prefix in common_prefixes:
            self.cache[prefix] = self.compute_valid_tokens(prefix)

    def compute_valid_tokens(self, prefix):
        valid = []
        for token_id in range(self.tokenizer.vocab_size):
            token = self.tokenizer.decode([token_id])
            if self.pattern.match(prefix + token):
                valid.append(token_id)
        return valid
```

#### 2. Vectorization

```python
def vectorized_mask_application(scores, valid_token_ids):
    """Apply mask using vectorized operations."""
    # Create mask tensor
    mask = torch.full_like(scores, float('-inf'))

    # Vectorized update
    mask[:, valid_token_ids] = 0

    # Single operation
    return scores + mask
```

#### 3. GPU Acceleration

```python
class GPUAcceleratedProcessor(LogitProcessor):
    def __init__(self, constraints):
        # Move constraint data to GPU
        self.constraint_matrix = torch.tensor(
            constraints,
            device='cuda'
        )

    def __call__(self, input_ids, scores):
        # Perform computation on GPU
        mask = self.compute_mask_gpu(input_ids)
        return scores * mask  # Element-wise on GPU
```

## Real-World Applications

### 1. Structured API Responses

```python
class APIResponseProcessor(LogitProcessor):
    """Ensure API responses follow OpenAPI spec."""

    def __init__(self, openapi_schema):
        self.schema = openapi_schema
        self.validator = OpenAPIValidator(openapi_schema)

    def __call__(self, input_ids, scores):
        current_json = self.tokenizer.decode(input_ids[0])

        # Validate against OpenAPI schema
        valid_tokens = self.validator.get_valid_continuations(current_json)

        mask = torch.full_like(scores, float('-inf'))
        mask[:, valid_tokens] = 0

        return scores + mask
```

### 2. Code Generation

```python
class SyntaxValidProcessor(LogitProcessor):
    """Ensure generated code is syntactically valid."""

    def __init__(self, language='python'):
        self.language = language
        self.parser = CodeParser(language)

    def __call__(self, input_ids, scores):
        code = self.tokenizer.decode(input_ids[0])

        # Get tokens that maintain valid syntax
        valid_tokens = self.parser.get_valid_tokens(code)

        # Boost valid tokens, penalize invalid
        for token_id in range(scores.shape[-1]):
            if token_id not in valid_tokens:
                scores[:, token_id] -= 10.0

        return scores
```

### 3. Safety Filtering

```python
class SafetyProcessor(LogitProcessor):
    """Filter unsafe or inappropriate content."""

    def __init__(self, safety_model):
        self.safety_model = safety_model
        self.threshold = 0.9

    def __call__(self, input_ids, scores):
        # Check each possible continuation
        for token_id in range(scores.shape[-1]):
            potential_text = self.get_text_with_token(input_ids, token_id)
            safety_score = self.safety_model.score(potential_text)

            if safety_score < self.threshold:
                scores[:, token_id] = float('-inf')

        return scores
```

## Best Practices

### 1. Processor Design

- **Keep it simple**: Each processor should have a single responsibility
- **Make it composable**: Design processors to work in pipelines
- **Cache aggressively**: Precompute and cache when possible
- **Handle edge cases**: Always check for empty/invalid inputs

### 2. Performance

```python
# Good: Vectorized operations
mask = torch.zeros_like(scores)
mask[:, invalid_tokens] = float('-inf')
scores = scores + mask

# Bad: Loop-based operations
for token in invalid_tokens:
    scores[:, token] = float('-inf')
```

### 3. Debugging

```python
class DebugLogitProcessor(LogitProcessor):
    """Wrapper for debugging other processors."""

    def __init__(self, processor, name):
        self.processor = processor
        self.name = name

    def __call__(self, input_ids, scores):
        # Log before
        top_5_before = torch.topk(scores, 5)
        print(f"{self.name} - Before: {top_5_before}")

        # Apply processor
        scores = self.processor(input_ids, scores)

        # Log after
        top_5_after = torch.topk(scores, 5)
        print(f"{self.name} - After: {top_5_after}")

        return scores
```

### 4. Testing

```python
def test_logit_processor(processor, test_cases):
    """Test framework for logit processors."""
    for input_ids, expected_valid_tokens in test_cases:
        # Create dummy scores
        scores = torch.randn(1, vocab_size)

        # Apply processor
        processed = processor(input_ids, scores)

        # Check only valid tokens have non-inf scores
        non_inf_tokens = (processed > float('-inf')).nonzero()
        assert set(non_inf_tokens) == set(expected_valid_tokens)
```

## Advanced Topics

### 1. Learned Logit Processors

```python
class LearnedConstraintProcessor(nn.Module, LogitProcessor):
    """Neural network that learns to apply constraints."""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.constraint_encoder = nn.LSTM(hidden_dim, hidden_dim)
        self.mask_predictor = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, scores, constraint_embedding):
        # Encode constraint
        _, (h, _) = self.constraint_encoder(constraint_embedding)

        # Predict mask
        mask_logits = self.mask_predictor(h.squeeze())
        mask = torch.sigmoid(mask_logits)

        # Apply learned mask
        return scores * mask
```

### 2. Probabilistic Constraints

```python
class ProbabilisticProcessor(LogitProcessor):
    """Apply constraints probabilistically."""

    def __init__(self, constraint, strength=0.9):
        self.constraint = constraint
        self.strength = strength

    def __call__(self, input_ids, scores):
        # Get hard constraint mask
        hard_mask = self.constraint(input_ids, scores)

        # Apply probabilistically
        soft_mask = torch.where(
            hard_mask == float('-inf'),
            scores - (scores.max() * self.strength),
            scores
        )

        return soft_mask
```

### 3. Adaptive Constraints

```python
class AdaptiveProcessor(LogitProcessor):
    """Adapt constraints based on generation quality."""

    def __init__(self, base_constraint):
        self.base_constraint = base_constraint
        self.history = []
        self.strength = 1.0

    def __call__(self, input_ids, scores):
        # Apply constraint with adaptive strength
        constraint_scores = self.base_constraint(input_ids, scores)

        # Blend with original scores based on strength
        blended = (self.strength * constraint_scores +
                  (1 - self.strength) * scores)

        # Adapt strength based on perplexity
        self.adapt_strength(scores)

        return blended

    def adapt_strength(self, scores):
        # High entropy → relax constraints
        entropy = -(F.softmax(scores) * F.log_softmax(scores)).sum()
        if entropy > threshold:
            self.strength *= 0.95
        else:
            self.strength = min(1.0, self.strength * 1.05)
```

## Future Directions

### Emerging Techniques

1. **Speculative Decoding**: Use logit processors to verify speculated tokens
2. **Watermarking**: Embed invisible signatures through logit manipulation
3. **Differential Privacy**: Add noise to logits for privacy preservation
4. **Quantum Constraints**: Leverage quantum computing for constraint solving

### Research Frontiers

- **Neural Constraint Compilation**: Learn optimal constraint representations
- **Cross-Model Transfer**: Share processors across different models
- **Continuous Constraints**: Extend beyond discrete token constraints
- **Compositional Reasoning**: Build complex constraints from primitives

## Conclusion

Logits and logit processors represent the fundamental control mechanism for modern LLM generation. They provide:

1. **Precise Control**: Mathematical guarantees on output structure
2. **Efficiency**: Minimal overhead compared to retraining or prompting
3. **Flexibility**: Arbitrary constraints through modular design
4. **Reliability**: Deterministic enforcement of requirements

As LLMs become critical infrastructure, mastery of logit processing becomes essential for building robust, controlled AI systems. The techniques covered here—from basic temperature scaling to complex grammar enforcement—form the foundation of constrained generation in production systems.

Whether using vLLM's high-throughput serving with XGrammar backend or Guidance's ultra-fast llguidance engine, understanding logits and their manipulation unlocks the full potential of structured LLM generation.

## Key Takeaways

✅ **Logits are raw scores** before probability conversion, giving us a control point

✅ **Logit processors modify these scores** to enforce constraints

✅ **Simple masks (setting to -∞)** can enforce complex requirements

✅ **Performance matters**: 50μs per token for production systems

✅ **Composability is key**: Chain simple processors for complex behavior

✅ **Both vLLM and Guidance** leverage logit processing, with different optimizations

The future of LLM deployment lies not in hoping models follow instructions, but in mathematically guaranteeing they do—and logit processors are the mechanism that makes this possible.