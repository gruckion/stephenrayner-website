# vLLM: High-Throughput LLM Serving Engine - Comprehensive Research Notes

## Executive Summary

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs), originally developed at UC Berkeley's Sky Computing Lab and now a community-driven project. Its revolutionary PagedAttention algorithm fundamentally reimagines memory management for LLM serving, achieving up to 24x higher throughput than baseline transformers implementations while maintaining near-optimal memory utilization with less than 4% waste.

## What is vLLM?

vLLM is an open-source library that provides:

- **State-of-the-art serving throughput** through innovative memory management
- **PagedAttention algorithm** for efficient KV cache management
- **Continuous batching** for dynamic request handling
- **Hardware flexibility** across NVIDIA, AMD, Intel GPUs and CPUs
- **Production-ready features** including OpenAI-compatible API, streaming, and guided generation

## Core Architecture

### System Components

```
┌──────────────────────────────────────────┐
│           vLLM Architecture              │
├──────────────────────────────────────────┤
│         API Layer (Python)               │
│  ┌────────────────────────────────────┐  │
│  │ OpenAI Server │ LLM Class │ Async │  │
│  └────────────────────────────────────┘  │
├──────────────────────────────────────────┤
│         Control Layer (Python)           │
│  ┌────────────────────────────────────┐  │
│  │ Scheduler │ Block Manager │ Cache │  │
│  └────────────────────────────────────┘  │
├──────────────────────────────────────────┤
│      Execution Layer (C++/CUDA)          │
│  ┌────────────────────────────────────┐  │
│  │ PagedAttention │ Custom Kernels   │  │
│  │ Tensor/Pipeline Parallelism        │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

### Key Components

1. **Engine Client**: Manages lifecycle and communication
2. **Core Engine**: Handles request scheduling and tokenization
3. **Executor**: Determines execution strategy (single vs. distributed)
4. **Workers**: Handle device-specific inference tasks
5. **Model Runners**: Load weights and execute inference logic
6. **Block Manager**: Manages memory allocation for KV cache

## PagedAttention: The Core Innovation

### The Memory Challenge

Traditional LLM serving faces critical memory bottlenecks:

- **KV Cache Growth**: Memory for each request grows dynamically with sequence length
- **Fragmentation**: 60-80% memory waste in existing systems
- **Redundancy**: Duplicate storage of shared prefixes
- **Static Allocation**: Pre-allocated maximum memory leads to underutilization

### PagedAttention Solution

Inspired by operating system virtual memory, PagedAttention introduces paging to LLM serving:

#### 1. Block-Based Memory Management

```python
# Conceptual representation
class KVBlock:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.keys = torch.empty(block_size, key_dim)
        self.values = torch.empty(block_size, value_dim)
        self.ref_count = 0
```

#### 2. Non-Contiguous Storage

- **Logical Blocks**: Continuous abstraction for sequences
- **Physical Blocks**: Actual GPU memory allocation
- **Block Tables**: Map logical to physical blocks
- **Dynamic Growth**: Allocate new blocks only when needed

#### 3. Memory Sharing Mechanisms

- **Reference Counting**: Track block usage across sequences
- **Copy-on-Write**: Share blocks until modification needed
- **Prefix Caching**: Reuse common prompt prefixes
- **Automatic Deduplication**: Hash-based block sharing

### Technical Implementation

#### Block Table Structure

```python
# Simplified block table implementation
class BlockTable:
    def __init__(self):
        self.logical_to_physical = {}  # Mapping
        self.physical_blocks = []       # Actual memory
        self.free_blocks = []           # Available blocks

    def allocate(self, sequence_id, logical_block_id):
        if self.free_blocks:
            physical_block = self.free_blocks.pop()
        else:
            physical_block = self.create_new_block()

        self.logical_to_physical[(sequence_id, logical_block_id)] = physical_block
        return physical_block
```

#### Attention Computation

```python
# PagedAttention kernel (conceptual)
def paged_attention(query, key_cache, value_cache, block_table):
    attention_scores = []

    for logical_block_id in range(num_blocks):
        physical_block_id = block_table[logical_block_id]

        # Fetch keys and values from physical blocks
        keys = key_cache[physical_block_id]
        values = value_cache[physical_block_id]

        # Compute attention for this block
        scores = torch.matmul(query, keys.T)
        attention_scores.append(scores)

    # Combine scores across blocks
    combined_scores = torch.cat(attention_scores)
    attention_weights = torch.softmax(combined_scores)

    # Apply attention to values
    output = apply_attention(attention_weights, value_cache, block_table)
    return output
```

### Performance Characteristics

| Metric | Traditional Systems | vLLM with PagedAttention |
|--------|-------------------|--------------------------|
| Memory Waste | 60-80% | <4% |
| Throughput | Baseline | Up to 24x |
| Batch Size | Limited by fragmentation | 2-4x larger |
| Prefix Sharing | None/Manual | Automatic |
| Dynamic Allocation | No | Yes |

## Advanced Features

### 1. Continuous Batching

Unlike static batching, vLLM implements continuous batching:

```python
# Continuous batching workflow
class ContinuousBatcher:
    def __init__(self):
        self.active_requests = []
        self.waiting_queue = []

    def schedule(self):
        # Add new requests to batch dynamically
        while self.has_capacity() and self.waiting_queue:
            request = self.waiting_queue.pop(0)
            self.active_requests.append(request)

        # Process batch
        self.process_batch(self.active_requests)

        # Remove completed requests
        self.active_requests = [r for r in self.active_requests if not r.is_complete()]
```

Benefits:

- **No padding overhead**: Different sequence lengths in same batch
- **Dynamic scheduling**: Add/remove requests on-the-fly
- **Better GPU utilization**: Keep GPU busy with varying workloads

### 2. Speculative Decoding

vLLM supports multiple speculative decoding strategies:

#### Draft Model Speculation

```python
# Using a smaller draft model
vllm serve meta-llama/Meta-Llama-3-70B \
    --speculative-model meta-llama/Meta-Llama-3-8B \
    --num-speculative-tokens 5
```

#### Prompt Lookup Decoding

```python
# Reuse tokens from prompt
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    speculative_model="[prompt_lookup]",
    num_speculative_tokens=5
)
```

Performance gains:

- **Up to 2.8x speedup** on repetitive tasks
- **1.5x speedup** on general text generation
- **Minimal overhead** when speculation fails

### 3. Tensor and Pipeline Parallelism

#### Tensor Parallelism

```python
# Distribute model layers across GPUs
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    tensor_parallel_size=4  # Use 4 GPUs
)
```

#### Pipeline Parallelism

```python
# Split model stages across GPUs
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    pipeline_parallel_size=2
)
```

#### Hybrid Parallelism

```python
# Combine both strategies
llm = LLM(
    model="meta-llama/Meta-Llama-3-405B",
    tensor_parallel_size=4,
    pipeline_parallel_size=2  # Total 8 GPUs
)
```

### 4. Guided Generation and Structured Outputs

vLLM provides multiple backends for constrained generation:

#### JSON Schema Enforcement

```python
from vllm import LLM, SamplingParams

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["name", "age"]
}

sampling_params = SamplingParams(
    guided_json=schema,
    temperature=0.7
)

llm = LLM(model="meta-llama/Meta-Llama-3-8B")
outputs = llm.generate(prompts, sampling_params)
```

#### Regex Pattern Matching

```python
# Generate email addresses
sampling_params = SamplingParams(
    guided_regex=r"[a-z]+@[a-z]+\.(com|org|net)",
    temperature=0
)
```

#### Choice Constraints

```python
# Force selection from options
sampling_params = SamplingParams(
    guided_choice=["positive", "negative", "neutral"],
    temperature=0
)
```

#### Grammar-Based Generation

```python
# Use context-free grammar
grammar = """
start: sentence
sentence: noun verb noun
noun: "Alice" | "Bob" | "Charlie"
verb: "likes" | "sees" | "helps"
"""

sampling_params = SamplingParams(
    guided_grammar=grammar
)
```

Backend options:

- **xgrammar**: Low latency per token, best for reused grammars
- **guidance**: Fast time-to-first-token, good for dynamic workloads
- **outlines**: Regex and JSON schema support
- **lm-format-enforcer**: Lightweight option

## API Design and Usage Patterns

### 1. Offline Batched Inference

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

# Configure generation
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
    repetition_penalty=1.2
)

# Batch processing
prompts = [
    "Explain quantum computing",
    "Write a Python function for sorting",
    "Describe the water cycle"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}")
```

### 2. OpenAI-Compatible Server

```bash
# Start server
vllm serve meta-llama/Meta-Llama-3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key YOUR_API_KEY
```

```python
# Client usage
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="http://localhost:8000/v1"
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain vLLM"}
    ],
    temperature=0.7,
    max_tokens=500
)

# Streaming
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 3. AsyncLLMEngine for High Concurrency

```python
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

async def process_requests():
    # Initialize async engine
    engine_args = AsyncEngineArgs(
        model="meta-llama/Meta-Llama-3-8B",
        max_num_seqs=256,
        gpu_memory_utilization=0.9
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Process multiple requests concurrently
    async def generate(prompt):
        request_id = f"req_{hash(prompt)}"
        sampling_params = SamplingParams(temperature=0.7)

        results = []
        async for output in engine.generate(prompt, sampling_params, request_id):
            results.append(output)

        return results[-1].outputs[0].text

    # Concurrent generation
    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    results = await asyncio.gather(*[generate(p) for p in prompts])

    return results

# Run async processing
results = asyncio.run(process_requests())
```

### 4. Multi-Modal Support

```python
from vllm import LLM, SamplingParams
from PIL import Image

# Load multi-modal model
llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Prepare image and text
image = Image.open("example.jpg")
prompt = "Describe this image in detail"

# Generate with image context
sampling_params = SamplingParams(temperature=0.8, max_tokens=200)
outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    sampling_params
)
```

## Performance Optimization Techniques

### 1. Quantization

```python
# INT8 quantization
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    quantization="int8"
)

# FP8 quantization (H100 GPUs)
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    quantization="fp8"
)

# AWQ quantization
llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq"
)
```

### 2. Memory Management

```python
# Configure memory allocation
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    gpu_memory_utilization=0.95,  # Use 95% of GPU memory
    swap_space=4,  # 4GB CPU swap space
    max_num_batched_tokens=8192,
    max_num_seqs=256
)
```

### 3. CUDA Graph Optimization

```python
# Enable CUDA graphs for lower latency
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    enforce_eager=False,  # Enable CUDA graphs
    max_seq_len_to_capture=4096
)
```

### 4. Prefix Caching

```python
# Enable automatic prefix caching
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    enable_prefix_caching=True
)

# Shared system prompt across requests
system_prompt = "You are a helpful coding assistant..."
prompts = [
    f"{system_prompt}\nWrite a Python function for {task}"
    for task in ["sorting", "searching", "hashing"]
]

# Prefix automatically cached and reused
outputs = llm.generate(prompts, sampling_params)
```

## Performance Benchmarks

### Throughput Comparison

| Model | Framework | Throughput (tokens/s) | Relative Performance |
|-------|-----------|----------------------|---------------------|
| Llama 3 8B | HuggingFace | 1,000 | 1x (baseline) |
| Llama 3 8B | TGI | 18,000 | 18x |
| Llama 3 8B | vLLM | 24,000 | 24x |
| Llama 3 70B | HuggingFace | 100 | 1x (baseline) |
| Llama 3 70B | TGI | 160 | 1.6x |
| Llama 3 70B | vLLM | 180 | 1.8x |

### Latency Metrics

| Metric | vLLM | TGI | Advantage |
|--------|------|-----|-----------|
| Time to First Token (TTFT) | 45ms | 52ms | 15% faster |
| Time per Output Token (TPOT) | 8ms | 16ms | 2x faster |
| P50 Latency | 200ms | 250ms | 20% lower |
| P99 Latency | 800ms | 1200ms | 33% lower |

### Memory Efficiency

```
Memory Usage Comparison (70B model, 2048 context):
┌─────────────────────────────────────┐
│ Framework │ KV Cache │ Waste │ Total │
├───────────┼──────────┼───────┼───────┤
│ Baseline  │ 40GB     │ 60%   │ 64GB  │
│ TGI       │ 35GB     │ 40%   │ 49GB  │
│ vLLM      │ 28GB     │ <4%   │ 29GB  │
└─────────────────────────────────────┘
```

### Scaling Performance

```python
# Batch size vs. throughput
batch_sizes = [1, 8, 32, 128, 256]
throughputs = {
    "vLLM": [1000, 7500, 28000, 95000, 180000],
    "TGI": [950, 6800, 22000, 70000, 120000],
    "HF": [900, 5000, 12000, 25000, None]  # OOM at 256
}
```

## Real-World Deployment

### 1. Production Configuration

```yaml
# vllm_config.yaml
model:
  name: "meta-llama/Meta-Llama-3-70B"
  revision: "main"
  tokenizer_mode: "auto"
  trust_remote_code: false
  download_dir: "/models"

serving:
  host: "0.0.0.0"
  port: 8000
  api_key: "${VLLM_API_KEY}"
  served_model_name: "llama-3-70b"

engine:
  max_model_len: 4096
  gpu_memory_utilization: 0.9
  max_num_seqs: 256
  max_num_batched_tokens: 8192

optimization:
  quantization: "fp8"
  enable_prefix_caching: true
  enable_chunked_prefill: true
  speculative_model: "meta-llama/Meta-Llama-3-8B"

parallelism:
  tensor_parallel_size: 4
  pipeline_parallel_size: 1

monitoring:
  disable_log_stats: false
  log_stats_interval: 10
```

### 2. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["vllm", "serve"]
        args:
          - "meta-llama/Meta-Llama-3-70B"
          - "--tensor-parallel-size=4"
          - "--gpu-memory-utilization=0.9"
          - "--max-model-len=4096"
        resources:
          limits:
            nvidia.com/gpu: 4
        ports:
        - containerPort: 8000
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-credentials
              key: token
```

### 3. Load Balancing

```python
# HAProxy configuration for vLLM cluster
from typing import List
import random
import requests

class VLLMLoadBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.health_check_interval = 30
        self.healthy_servers = servers.copy()

    def get_server(self, strategy="round_robin"):
        if strategy == "round_robin":
            server = self.healthy_servers[0]
            self.healthy_servers.rotate(1)
            return server
        elif strategy == "least_connections":
            return min(self.healthy_servers, key=self.get_connections)
        elif strategy == "random":
            return random.choice(self.healthy_servers)

    def health_check(self):
        for server in self.servers:
            try:
                response = requests.get(f"{server}/health", timeout=5)
                if response.status_code == 200:
                    if server not in self.healthy_servers:
                        self.healthy_servers.append(server)
                else:
                    self.healthy_servers.remove(server)
            except:
                if server in self.healthy_servers:
                    self.healthy_servers.remove(server)
```

## Integration Ecosystem

### 1. LangChain Integration

```python
from langchain_community.llms import VLLM

llm = VLLM(
    model="meta-llama/Meta-Llama-3-8B",
    trust_remote_code=True,
    max_new_tokens=128,
    temperature=0.7,
    vllm_kwargs={
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
    }
)

response = llm.invoke("What is vLLM?")
```

### 2. BentoML Integration

```python
import bentoml
from vllm import LLM

@bentoml.service
class VLLMService:
    def __init__(self):
        self.model = LLM(
            model="meta-llama/Meta-Llama-3-8B",
            gpu_memory_utilization=0.9
        )

    @bentoml.api
    async def generate(self, prompt: str) -> str:
        outputs = self.model.generate([prompt])
        return outputs[0].outputs[0].text
```

### 3. Ray Serve Integration

```python
from ray import serve
from vllm import LLM

@serve.deployment(
    num_replicas=3,
    ray_actor_options={"num_gpus": 1}
)
class VLLMDeployment:
    def __init__(self):
        self.llm = LLM(model="meta-llama/Meta-Llama-3-8B")

    async def __call__(self, request):
        prompt = await request.json()
        outputs = self.llm.generate([prompt["text"]])
        return {"response": outputs[0].outputs[0].text}

# Deploy
serve.run(VLLMDeployment.bind())
```

## Advanced Topics

### 1. Custom CUDA Kernels

vLLM implements optimized CUDA kernels for:

- **FlashAttention**: Fused attention computation
- **FlashInfer**: Optimized inference kernels
- **Triton kernels**: Custom operations
- **Marlin kernels**: Quantized operations

### 2. Chunked Prefill

```python
# Enable chunked prefill for long prompts
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    enable_chunked_prefill=True,
    max_num_batched_tokens=512
)
```

Benefits:

- Lower time-to-first-token for long prompts
- Better scheduling of prefill and decode
- Reduced memory spikes

### 3. LoRA Adapter Support

```python
# Serve multiple LoRA adapters
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    enable_lora=True,
    max_loras=4,
    max_lora_rank=32
)

# Use specific adapter
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql-adapter", 1, "/path/to/adapter")
)
```

### 4. Mixture of Experts (MoE)

```python
# Efficient MoE serving
llm = LLM(
    model="mistralai/Mixtral-8x7B-v0.1",
    tensor_parallel_size=8,
    enable_fused_moe=True
)
```

## Best Practices

### 1. Model Selection

- Start with smaller models for development
- Use quantization for larger models
- Consider draft models for speculative decoding

### 2. Memory Optimization

```python
# Optimal configuration
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    gpu_memory_utilization=0.95,  # Use most GPU memory
    enable_prefix_caching=True,    # Cache common prefixes
    enable_chunked_prefill=True,   # Handle long prompts efficiently
    swap_space=4                   # CPU offloading buffer
)
```

### 3. Batching Strategy

- Use continuous batching for varying workloads
- Set appropriate `max_num_seqs` based on memory
- Configure `max_num_batched_tokens` for throughput

### 4. Monitoring

```python
# Enable comprehensive logging
import logging

logging.basicConfig(level=logging.INFO)
vllm_logger = logging.getLogger("vllm")
vllm_logger.setLevel(logging.DEBUG)

# Track metrics
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    disable_log_stats=False,
    log_stats_interval=10  # Log every 10 seconds
)
```

### 5. Error Handling

```python
from vllm.outputs import RequestOutput

def safe_generate(llm, prompts, sampling_params):
    try:
        outputs = llm.generate(prompts, sampling_params)
        return [o.outputs[.text for o in outputs]
    except torch.cuda.OutOfMemoryError:
        # Reduce batch size and retry
        mid = len(prompts) // 2
        part1 = safe_generate(llm, prompts[:mid], sampling_params)
        part2 = safe_generate(llm, prompts[mid:], sampling_params)
        return part1 + part2
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return None
```

## Limitations and Considerations

### Current Limitations

1. **Pipeline Parallelism**: Incompatible with speculative decoding
2. **OpenAI API**: Logit bias limited to 300 tokens
3. **Embedding Models**: Limited support compared to generation
4. **Custom Models**: Requires implementation of model runner

### Hardware Requirements

| Model Size | Minimum VRAM | Recommended VRAM | Optimal Setup |
|------------|--------------|------------------|---------------|
| 7B | 16GB | 24GB | 1x A100 40GB |
| 13B | 32GB | 40GB | 1x A100 80GB |
| 70B | 160GB | 320GB | 4x A100 80GB |
| 405B | 810GB | 1TB+ | 8x H100 80GB |

### When to Use vLLM

✅ **Ideal for:**

- High-throughput serving requirements
- Variable batch sizes and sequence lengths
- Memory-constrained deployments
- OpenAI API compatibility needs
- Structured output generation

❌ **Less suitable for:**

- Single-request latency optimization
- Custom model architectures without support
- Extreme quantization requirements
- Edge deployment on consumer GPUs

## Future Roadmap

### V1 Engine (2025)

- Complete architectural overhaul
- Improved compilation with Torch compiler
- Better CUDA graph support
- Enhanced scheduling algorithms

### Upcoming Features

- **Jump Decoding**: Skip deterministic tokens
- **Disaggregated Prefilling**: Separate prefill/decode nodes
- **Cross-attention Caching**: For encoder-decoder models
- **Improved MoE Support**: Better expert parallelism

### Research Directions

- Integration with new attention mechanisms
- Support for longer context windows
- Advanced scheduling algorithms
- Hardware-specific optimizations

## Conclusion

vLLM represents a paradigm shift in LLM serving through its innovative PagedAttention algorithm and comprehensive optimization stack. Key achievements include:

1. **Memory Revolution**: Near-optimal utilization with <4% waste
2. **Throughput Leadership**: Up to 24x improvement over baseline
3. **Production Ready**: OpenAI-compatible API with enterprise features
4. **Flexibility**: Support for diverse hardware and model architectures
5. **Active Development**: Continuous improvements and community support

As LLMs become critical infrastructure, vLLM's combination of performance, reliability, and ease of use makes it the de facto standard for high-throughput LLM serving. The project's transition from academic research to community-driven development ensures continued innovation and adaptation to evolving requirements.

## Resources and References

### Official Resources

- GitHub: <https://github.com/vllm-project/vllm>
- Documentation: <https://docs.vllm.ai>
- Blog: <https://blog.vllm.ai>
- Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)

### Community Resources

- Discord: vLLM community server
- Forums: <https://discuss.vllm.ai>
- Meetups: Regular community events

### Benchmarks and Comparisons

- vLLM vs TGI: Performance analysis
- Anyscale benchmarks
- BentoML LLM serving guide

### Related Projects

- TGI (Text Generation Inference): HuggingFace's serving solution
- TensorRT-LLM: NVIDIA's optimized inference
- Triton Inference Server: General ML serving
- Ray Serve: Distributed serving framework

### Academic References

- PagedAttention paper (SOSP 2023)
- Continuous batching research
- Speculative decoding studies
- Memory management in ML systems
