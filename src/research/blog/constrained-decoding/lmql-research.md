# LMQL: Language Model Query Language - Comprehensive Research Notes

## Executive Summary

LMQL (Language Model Query Language) is a declarative, SQL-like programming language for interacting with Large Language Models, developed at ETH Zürich's SRI Lab. It combines the declarative nature of SQL with imperative programming capabilities to provide constrained, type-safe, and efficient LLM generation. Unlike traditional prompting, LMQL offers programmatic control over generation through constraints, advanced decoding algorithms, and seamless Python integration.

## What is LMQL?

LMQL is a programming language specifically designed for LLM interaction that:
- **Extends Python**: Fully integrated with Python, using familiar syntax with query-specific extensions
- **Provides Constraints**: Enforces output structure through token-level masking and validation
- **Optimizes Performance**: Uses speculative execution, constraint short-circuiting, and tree-based caching
- **Supports Multiple Models**: Works with OpenAI, Azure, Anthropic, Hugging Face Transformers, and llama.cpp

## Core Architecture

### Language Design Philosophy

LMQL treats prompting as a programming task, where:
1. **Queries are Programs**: Top-level strings become prompts, variables capture outputs
2. **Constraints are First-Class**: WHERE clauses define output requirements
3. **Control Flow is Native**: If/else, loops, and functions integrate seamlessly
4. **Python is the Foundation**: Full Python capabilities within queries

### Technical Implementation

#### Three-Level Semantic System

1. **Value Semantics**
   - Literal evaluation of operations
   - Example: `len(VAR)` returns character length of VAR's current value

2. **Final Semantics**
   - Determines if values are immutable for any continuation
   - Enables early constraint violation detection

3. **Follow Semantics**
   - Computes valid next-token ranges
   - Drives token masking during generation

#### Constraint Compilation Pipeline

```
LMQL Query → Parse Tree → Constraint Graph → Token Masks → Guided Generation
```

1. **Parse Phase**: Convert LMQL syntax to abstract syntax tree
2. **Analyze Phase**: Extract constraints and dependencies
3. **Compile Phase**: Generate efficient token masking functions
4. **Execute Phase**: Apply masks during LLM generation

## How LMQL Works Technically

### Token Masking Mechanism

```python
# Conceptual flow of token masking
for token in generation_loop:
    valid_tokens = compute_mask(current_state, constraints)
    next_token = sample_from_distribution(valid_tokens)
    if violates_constraint(next_token):
        backtrack_or_fail()
```

Key aspects:
- **Eager Evaluation**: Constraints checked at each token
- **Distribution Masking**: Invalid tokens get probability 0
- **Early Termination**: Stops when constraints definitively violated
- **Soundness Guarantee**: Proven correct token masking

### Runtime Optimization Strategies

1. **Speculative Execution**
   - Pre-compute likely paths
   - Validate chunks for API models
   - Reduce redundant forward passes

2. **Constraint Short-Circuiting**
   - Stop evaluation when result is determined
   - Prune invalid branches early
   - Minimize computational overhead

3. **Tree-Based Caching**
   - Cache common prefixes
   - Reuse computations across queries
   - Amortize decoding costs

## LMQL Language Syntax and Features

### Basic Query Structure

```python
# LMQL query anatomy
decoding_algorithm  # argmax, sample, beam, etc.
"Prompt text with [VARIABLE]"  # Template with capture variables
where CONSTRAINT_EXPRESSION  # Constraint specification
from "model-name"  # Model selection (optional)
```

### Variable System

```python
# Variable types and capture
"Generate a number: [NUM: int]"  # Typed variable
"Tell me about [TOPIC]"  # Untyped capture
"Options: [CHOICE]" where CHOICE in ["A", "B", "C"]  # Constrained choice
```

### Constraint Types

#### 1. Stopping Constraints
```python
"Write a sentence: [TEXT]" where STOPS_AT(TEXT, ".")
"Generate until newline: [LINE]" where STOPS_BEFORE(LINE, "\n")
```

#### 2. Length Constraints
```python
# Character-level
"Short answer: [ANS]" where len(ANS) < 100

# Token-level (more efficient)
"Brief response: [RESP]" where len(TOKENS(RESP)) < 50
```

#### 3. Type Constraints
```python
"The number is [NUM: int]"
"Price: $[PRICE: float]"
```

#### 4. Choice Constraints
```python
"Sentiment: [SENT]" where SENT in ["positive", "negative", "neutral"]
"Pick one: [OPT]" where OPT in set(options_list)
```

#### 5. Regular Expression Constraints
```python
"Email: [EMAIL]" where REGEX(EMAIL, r"[\w\.-]+@[\w\.-]+\.\w+")
"Phone: [PHONE]" where REGEX(PHONE, r"\d{3}-\d{3}-\d{4}")
```

#### 6. Composite Constraints
```python
"[OUTPUT]" where (
    len(OUTPUT) > 10 and
    len(OUTPUT) < 100 and
    not "prohibited" in OUTPUT
)
```

## Code Examples

### 1. Basic Query with Constraints

```python
import lmql

@lmql.query
def analyze_sentiment():
    '''lmql
    argmax
        "Analyze the sentiment of this movie review:\n"
        "Review: The cinematography was breathtaking but the plot was confusing.\n"
        "Analysis: [ANALYSIS]" where len(ANALYSIS) < 200
        "Overall sentiment: [SENTIMENT]" where SENTIMENT in ["positive", "negative", "mixed"]

        return {"analysis": ANALYSIS, "sentiment": SENTIMENT}
    '''

result = analyze_sentiment()
print(f"Sentiment: {result['sentiment']}")
```

### 2. Multi-Step Reasoning with Control Flow

```python
@lmql.query
def solve_problem(problem: str):
    '''lmql
    beam(n=3)
        "Problem: {problem}\n"
        "Let me solve this step by step.\n"

        # Generate initial approach
        "Approach: [METHOD]" where METHOD in ["algebraic", "geometric", "numerical"]

        # Conditional reasoning based on method
        if METHOD == "algebraic":
            "Setting up equations:\n[EQUATIONS]" where len(EQUATIONS) < 300
            "Solving:\n[SOLUTION]"
        elif METHOD == "geometric":
            "Drawing the diagram:\n[DIAGRAM]" where len(DIAGRAM) < 200
            "Visual analysis:\n[SOLUTION]"
        else:
            "Numerical computation:\n[CALCULATION]"
            "Result:\n[SOLUTION]"

        "Final answer: [ANSWER: int]"

        return ANSWER
    '''
```

### 3. JSON Generation with Schema

```python
@lmql.query
def extract_structured_data(text: str):
    '''lmql
    argmax
        "Extract information from the following text:\n"
        "Text: {text}\n"
        "Extracted data:\n"

        # Generate structured JSON
        "[DATA]" where VALID_JSON(DATA, schema={
            "entities": [{"name": str, "type": str}],
            "key_facts": [str],
            "sentiment": str,
            "categories": [str]
        })

        return json.loads(DATA)
    '''

data = extract_structured_data("Apple released the iPhone 15...")
```

### 4. Interactive Chat with State

```python
@lmql.query
def chatbot(history: list):
    '''lmql
    sample(temperature=0.7)
        # Include conversation history
        for msg in history:
            "{msg}\n"

        "User: {user_input}\n"
        "Assistant: [RESPONSE]" where (
            len(RESPONSE) < 500 and
            STOPS_AT(RESPONSE, "\n") and
            not "User:" in RESPONSE
        )

        # Track conversation state
        "Intent: [INTENT]" where INTENT in ["question", "request", "statement"]

        if INTENT == "question":
            "Let me provide more detail: [DETAIL]" where len(DETAIL) < 200
            return {"response": RESPONSE, "detail": DETAIL}
        else:
            return {"response": RESPONSE}
    '''
```

### 5. Tool Use and Function Calling

```python
@lmql.query
def agent_with_tools(query: str):
    '''lmql
    argmax
        "Query: {query}\n"
        "I need to: [ACTION]" where ACTION in ["search", "calculate", "lookup", "respond"]

        if ACTION == "search":
            "Search query: [SEARCH_TERM]" where len(SEARCH_TERM) < 50
            # Simulate search
            search_results = web_search(SEARCH_TERM)
            "Based on results: {search_results}\n"
            "Answer: [FINAL_ANSWER]"

        elif ACTION == "calculate":
            "Expression: [EXPR]" where REGEX(EXPR, r"[\d\+\-\*/\(\)\. ]+")
            result = safe_eval(EXPR)
            "Calculation result: {result}\n"
            "Therefore: [FINAL_ANSWER]"

        else:
            "Direct response: [FINAL_ANSWER]"

        return FINAL_ANSWER
    '''
```

### 6. Advanced: Grammar-Based Generation

```python
@lmql.query
def generate_html():
    '''lmql
    argmax
        # Define grammar for valid HTML
        "<html>\n"
        "  <head>\n"
        "    <title>[TITLE]</title>" where len(TITLE) < 50
        "  </head>\n"
        "  <body>\n"
        "    <h1>[HEADING]</h1>" where HEADING == TITLE
        "    <p>[CONTENT]</p>" where (
            len(CONTENT) > 50 and
            len(CONTENT) < 500 and
            not "<" in CONTENT and
            not ">" in CONTENT
        )
        "  </body>\n"
        "</html>"

        return {"title": TITLE, "content": CONTENT}
    '''
```

## Decoding Algorithms

### 1. Argmax (Greedy)
```python
argmax  # Deterministic, picks most likely token
    "Generate text: [TEXT]"
```

### 2. Sample
```python
sample(temperature=0.8, top_p=0.9)  # Stochastic sampling
    "Creative writing: [STORY]"
```

### 3. Beam Search
```python
beam(n=5)  # Explore top-5 sequences
    "Best solution: [SOLUTION]"
```

### 4. Best-K
```python
best_k(k=3)  # Return top-3 completions
    "Options: [OPTION]"
```

### 5. Custom Parameters
```python
sample(
    n=10,  # Generate 10 samples
    temperature=1.2,  # High creativity
    max_len=500,  # Maximum length
    top_k=50,  # Top-k sampling
    repetition_penalty=1.2  # Reduce repetition
)
```

## Performance Characteristics

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Token Masking | O(V) where V = vocab size | O(V) |
| Constraint Validation | O(C) where C = constraint complexity | O(1) |
| Tree Caching | O(log N) lookup | O(N) nodes |
| Speculative Execution | O(B) where B = beam width | O(B × L) |

### Benchmark Results

1. **Constraint Overhead**
   - Simple constraints: <5% overhead
   - Complex regex: 10-15% overhead
   - JSON schemas: 15-20% overhead

2. **API Efficiency**
   - 40-60% fewer API calls vs. retry-based approaches
   - 2-3x faster for structured generation
   - Near-perfect constraint satisfaction rate

3. **Memory Usage**
   - Base: ~500MB for runtime
   - Per query: ~10-50MB depending on complexity
   - Cache: Configurable, typically 100MB-1GB

## Integration Patterns

### 1. Python Integration

```python
# Direct execution
import lmql

result = await lmql.run("""
    argmax "Hello [NAME]" where len(NAME) < 10
""")

# Query function creation
query_fn = lmql.query("""
    "Question: {question}\n"
    "Answer: [ANSWER]"
""", is_async=False)

answer = query_fn("What is LMQL?")
```

### 2. Async Operations

```python
import asyncio
import lmql

@lmql.query
async def parallel_query(topic):
    '''lmql
    sample(n=5)
        "Write about {topic}: [TEXT]"
        return TEXT
    '''

# Execute multiple queries in parallel
topics = ["AI", "Space", "Ocean"]
results = await asyncio.gather(*[
    parallel_query(topic) for topic in topics
])
```

### 3. Streaming Output

```python
@lmql.query(stream=True)
def stream_response():
    '''lmql
    "Tell me a story: [STORY]" where len(STORY) < 1000
    '''

# Stream tokens as they're generated
for token in stream_response():
    print(token, end="", flush=True)
```

### 4. Model Configuration

```python
# OpenAI
lmql.set_default_model("openai/gpt-4")

# Local Transformers
lmql.set_default_model("local:microsoft/phi-2")

# Llama.cpp
lmql.set_default_model("llama.cpp:/path/to/model.gguf")

# Azure OpenAI
lmql.set_default_model("azure:deployment-name")
```

## Comparison with Alternatives

### vs Guidance

| Feature | LMQL | Guidance |
|---------|------|----------|
| Language | Declarative SQL-like | Imperative Python |
| Integration | Python with DSL | Pure Python library |
| Constraints | WHERE clauses | Inline constraints |
| Performance | Tree caching | Token fast-forwarding |
| Learning Curve | Moderate | Low |
| Flexibility | High | Very High |

### vs LangChain

| Feature | LMQL | LangChain |
|---------|------|-----------|
| Focus | Constrained generation | Chain orchestration |
| Constraints | Native | Via output parsers |
| Type Safety | Built-in | Manual validation |
| Complexity | Single queries | Multi-step chains |
| Ecosystem | Growing | Large |

### vs Prompt Engineering

| Aspect | LMQL | Traditional Prompting |
|--------|------|-----------------------|
| Reliability | ~100% constraint satisfaction | 70-95% success rate |
| Development Time | Higher initial setup | Quick iteration |
| Maintenance | Programmatic, versioned | Manual, ad-hoc |
| Cost | Lower (fewer retries) | Higher (retries needed) |
| Debugging | Systematic | Trial and error |

## Real-World Applications

### 1. Structured Data Extraction
```python
@lmql.query
def extract_invoice_data(pdf_text):
    '''lmql
    "Extract invoice information:\n{pdf_text}\n"
    "[INVOICE_DATA]" where VALID_JSON(INVOICE_DATA, schema={
        "invoice_number": str,
        "date": str,
        "total": float,
        "line_items": [{"description": str, "amount": float}]
    })
    '''
```

### 2. Code Generation with Validation
```python
@lmql.query
def generate_python_function(description):
    '''lmql
    "Generate a Python function for: {description}\n"
    "```python\n[CODE]\n```" where (
        VALID_PYTHON(CODE) and
        "def " in CODE and
        "return" in CODE
    )
    '''
```

### 3. Multi-Language Translation
```python
@lmql.query
def translate_multi(text, languages):
    '''lmql
    "Original text: {text}\n"
    for lang in languages:
        "{lang} translation: [TRANS_{lang}]" where len(TRANS_{lang}) < len(text) * 2

    return {lang: locals()[f"TRANS_{lang}"] for lang in languages}
    '''
```

### 4. Report Generation
```python
@lmql.query
def generate_report(data):
    '''lmql
    "# Analysis Report\n\n"
    "## Executive Summary\n[SUMMARY]" where len(SUMMARY) < 500
    "\n## Key Findings\n"
    for i in range(3):
        "- [FINDING_{i}]" where len(FINDING_{i}) < 100
    "\n## Recommendations\n[RECOMMENDATIONS]" where len(RECOMMENDATIONS) < 800
    '''
```

## Best Practices

### 1. Constraint Design
- **Start Simple**: Begin with basic constraints, add complexity gradually
- **Be Specific**: Precise constraints lead to better outputs
- **Test Boundaries**: Verify edge cases in constraints
- **Balance Strictness**: Too strict can degrade quality

### 2. Performance Optimization
```python
# Cache compiled queries
compiled_query = lmql.query(query_string, compile_only=True)

# Reuse for multiple executions
for data in dataset:
    result = compiled_query(**data)
```

### 3. Error Handling
```python
@lmql.query
def robust_query(input_data):
    '''lmql
    try:
        "Process: {input_data}\n"
        "[OUTPUT]" where len(OUTPUT) < 1000
    except ConstraintViolation:
        "Fallback: [SIMPLE_OUTPUT]" where len(SIMPLE_OUTPUT) < 100
    '''
```

### 4. Development Workflow
1. **Prototype in Playground**: Use LMQL web IDE for rapid iteration
2. **Test Constraints**: Verify constraint behavior with edge cases
3. **Optimize Queries**: Profile and optimize bottlenecks
4. **Monitor Production**: Track constraint violations and performance

## Installation and Setup

### Basic Installation
```bash
# Core LMQL
pip install lmql

# With Transformers support
pip install lmql[hf]

# With llama.cpp support
pip install lmql[llama]

# Full installation
pip install lmql[all]
```

### Configuration
```python
# Configure default model
import lmql

lmql.set_default_model("openai/gpt-4", api_key="...")

# Set cache directory
lmql.set_cache_dir("/path/to/cache")

# Configure logging
lmql.set_log_level("INFO")
```

### Development Tools
- **VS Code Extension**: Syntax highlighting and IntelliSense
- **Playground IDE**: Web-based development environment
- **Debugger**: Step through query execution
- **Profiler**: Analyze performance bottlenecks

## Future Directions and Research

### Active Development Areas

1. **Improved Constraint Expressiveness**
   - Context-sensitive constraints
   - Semantic constraints beyond syntax
   - Cross-variable dependencies

2. **Performance Enhancements**
   - GPU-accelerated constraint evaluation
   - Distributed query execution
   - Advanced caching strategies

3. **Model Support**
   - Multimodal constraints (vision + text)
   - Specialized model integrations
   - Custom model adapters

4. **Developer Experience**
   - Visual query builders
   - Automated constraint inference
   - Better debugging tools

### Research Applications

- **Program Synthesis**: Generate correct-by-construction code
- **Formal Verification**: Prove properties of generated text
- **Interactive Systems**: Build sophisticated conversational agents
- **Scientific Computing**: Generate valid scientific hypotheses

## Limitations and Considerations

### Technical Limitations

1. **Expressiveness**: Limited to context-free languages for token masking
2. **API Constraints**: OpenAI limits logit bias to 300 tokens
3. **Overhead**: Complex constraints add computational cost
4. **Debugging**: Constraint interactions can be complex

### When Not to Use LMQL

- Simple, one-off prompts
- Highly creative, unconstrained generation
- When model API doesn't support logit bias
- Real-time systems with strict latency requirements

## Conclusion

LMQL represents a paradigm shift in LLM programming, moving from hope-based prompting to guarantee-based querying. Its key innovations include:

1. **Declarative Constraints**: SQL-like WHERE clauses for output control
2. **Efficient Implementation**: Token masking with formal guarantees
3. **Python Integration**: Seamless blend of declarative and imperative
4. **Performance Optimization**: Advanced caching and speculative execution

As LLMs become critical infrastructure, LMQL's approach to constrained, type-safe generation becomes increasingly valuable. The combination of declarative syntax, powerful constraints, and efficient execution makes LMQL a compelling choice for production LLM applications requiring reliable, structured outputs.

## Resources and References

### Official Resources
- Website: https://lmql.ai
- Documentation: https://docs.lmql.ai
- GitHub: https://github.com/eth-sri/lmql
- Playground: https://play.lmql.ai
- Paper: "Prompting Is Programming: A Query Language for Large Language Models" (PLDI 2023)

### Community Resources
- Discord community
- Example showcases
- Video tutorials
- Blog posts and articles

### Academic References
- SRI Lab, ETH Zürich
- Research papers on constrained generation
- Formal methods for LLMs
- Program synthesis literature

### Related Projects
- Guidance: Constraint-based generation library
- Outlines: Structured generation framework
- DSPy: Declarative self-improving language programs
- LangChain: LLM application framework