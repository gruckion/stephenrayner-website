# Using Vercel AI SDK with vLLM: Complete Integration Guide

## Executive Summary

vLLM's OpenAI-compatible API server enables seamless integration with the Vercel AI SDK, allowing developers to leverage self-hosted models with the same ease as commercial providers. This guide demonstrates how to configure, deploy, and utilize vLLM with Vercel AI SDK for production applications, including streaming, tool calling, structured outputs, and advanced features.

## Why vLLM + Vercel AI SDK?

### Benefits of This Integration

1. **Self-Hosted Infrastructure**: Complete control over data, models, and infrastructure
2. **Cost Efficiency**: Predictable costs without per-token pricing
3. **Performance**: vLLM's PagedAttention provides up to 24x throughput improvement
4. **Privacy & Compliance**: Keep sensitive data on-premises
5. **Model Flexibility**: Use any compatible open-source model
6. **React Server Components**: Native RSC support through Vercel AI SDK
7. **Unified API**: Same interface for both commercial and self-hosted models

### Use Cases

- **Healthcare**: HIPAA-compliant AI applications
- **Financial Services**: On-premises deployment for sensitive data
- **Government**: Secure, air-gapped environments
- **Enterprise**: Custom models with proprietary data
- **Research**: Experimental models and architectures

## Setting Up vLLM Server

### 1. Starting vLLM with OpenAI-Compatible API

```bash
# Basic setup
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000

# Production configuration
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --api-key $VLLM_API_KEY \
  --served-model-name llama-3-70b
```

### 2. Docker Deployment

```dockerfile
# Dockerfile for vLLM server
FROM vllm/vllm-openai:latest

ENV MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
ENV TENSOR_PARALLEL_SIZE=1
ENV GPU_MEMORY_UTILIZATION=0.9

CMD ["sh", "-c", "vllm serve $MODEL_NAME \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - VLLM_API_KEY=${VLLM_API_KEY}
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      vllm serve meta-llama/Meta-Llama-3-8B-Instruct
      --host 0.0.0.0
      --port 8000
      --api-key ${VLLM_API_KEY}
      --gpu-memory-utilization 0.9
```

### 3. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
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
        ports:
        - containerPort: 8000
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-credentials
              key: token
        - name: VLLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: vllm-credentials
              key: api-key
        command:
        - vllm
        - serve
        - meta-llama/Meta-Llama-3-70B-Instruct
        - --host=0.0.0.0
        - --port=8000
        - --tensor-parallel-size=4
        - --gpu-memory-utilization=0.9
        resources:
          limits:
            nvidia.com/gpu: 4
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

## Basic Integration

### 1. Installation

```bash
npm install ai @ai-sdk/openai
# or
pnpm add ai @ai-sdk/openai
# or
yarn add ai @ai-sdk/openai
```

### 2. Configuration

```typescript
// lib/ai-provider.ts
import { createOpenAI } from '@ai-sdk/openai';

// Create vLLM provider instance
export const vllm = createOpenAI({
  baseURL: process.env.VLLM_BASE_URL || 'http://localhost:8000/v1',
  apiKey: process.env.VLLM_API_KEY || 'vllm-api-key',
  compatibility: 'strict', // Important for full compatibility
  name: 'vllm', // Optional: changes provider name in responses
  headers: {
    'X-Custom-Header': 'value', // Optional custom headers
  },
});

// Create model instances
export const llama3Chat = vllm.chat('meta-llama/Meta-Llama-3-8B-Instruct');
export const llama3Completion = vllm.completion('meta-llama/Meta-Llama-3-8B');
```

### 3. Environment Variables

```bash
# .env.local
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=your-vllm-api-key
VLLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

## Text Generation Examples

### Basic Text Generation

```typescript
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';

async function basicGeneration() {
  const { text, usage, finishReason } = await generateText({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    prompt: 'Explain quantum computing in simple terms.',
    temperature: 0.7,
    maxTokens: 500,
  });

  console.log('Generated text:', text);
  console.log('Token usage:', usage);
  console.log('Finish reason:', finishReason);
}
```

### Streaming Text Generation

```typescript
import { streamText } from 'ai';
import { vllm } from '@/lib/ai-provider';

async function streamGeneration() {
  const result = await streamText({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    messages: [
      {
        role: 'system',
        content: 'You are a helpful assistant.',
      },
      {
        role: 'user',
        content: 'Write a story about a robot learning to paint.',
      },
    ],
    temperature: 0.8,
    maxTokens: 1000,
  });

  // Stream the response
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }

  // Get final usage stats
  const usage = await result.usage;
  console.log('\nToken usage:', usage);
}
```

### Chat Conversation

```typescript
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';

async function chatConversation() {
  const messages = [
    { role: 'system', content: 'You are a helpful coding assistant.' },
    { role: 'user', content: 'How do I sort an array in Python?' },
    { role: 'assistant', content: 'You can sort an array in Python using...' },
    { role: 'user', content: 'What about sorting in reverse order?' },
  ];

  const { text } = await generateText({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    messages,
    temperature: 0.3,
  });

  console.log('Response:', text);
}
```

## Structured Outputs

### JSON Generation with Zod Schema

```typescript
import { generateObject } from 'ai';
import { vllm } from '@/lib/ai-provider';
import { z } from 'zod';

const RecipeSchema = z.object({
  name: z.string(),
  servings: z.number(),
  prepTime: z.number().describe('Preparation time in minutes'),
  cookTime: z.number().describe('Cooking time in minutes'),
  ingredients: z.array(
    z.object({
      item: z.string(),
      amount: z.string(),
      unit: z.string(),
    })
  ),
  instructions: z.array(z.string()),
  difficulty: z.enum(['easy', 'medium', 'hard']),
});

async function generateRecipe() {
  const { object } = await generateObject({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    schema: RecipeSchema,
    prompt: 'Generate a detailed recipe for vegetarian lasagna.',
  });

  console.log('Generated recipe:', JSON.stringify(object, null, 2));
}
```

### Streaming Objects

```typescript
import { streamObject } from 'ai';
import { vllm } from '@/lib/ai-provider';
import { z } from 'zod';

const StorySchema = z.object({
  title: z.string(),
  genre: z.string(),
  characters: z.array(
    z.object({
      name: z.string(),
      role: z.string(),
      description: z.string(),
    })
  ),
  plot: z.string(),
  twist: z.string(),
});

async function streamStory() {
  const { partialObjectStream } = await streamObject({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    schema: StorySchema,
    prompt: 'Create an original science fiction story outline.',
  });

  // Stream partial objects as they're generated
  for await (const partialObject of partialObjectStream) {
    console.clear();
    console.log('Story outline:', JSON.stringify(partialObject, null, 2));
  }
}
```

## Tool Calling

### Basic Tool Usage

```typescript
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';
import { z } from 'zod';

// Define tools
const tools = {
  weather: {
    description: 'Get current weather for a location',
    parameters: z.object({
      location: z.string().describe('City and country'),
      unit: z.enum(['celsius', 'fahrenheit']).optional(),
    }),
    execute: async ({ location, unit = 'celsius' }) => {
      // Simulate weather API call
      return {
        location,
        temperature: Math.round(Math.random() * 30),
        unit,
        condition: ['sunny', 'cloudy', 'rainy'][Math.floor(Math.random() * 3)],
      };
    },
  },
  calculator: {
    description: 'Perform mathematical calculations',
    parameters: z.object({
      expression: z.string().describe('Mathematical expression to evaluate'),
    }),
    execute: async ({ expression }) => {
      try {
        // In production, use a safe math parser
        const result = eval(expression);
        return { expression, result };
      } catch (error) {
        return { error: 'Invalid expression' };
      }
    },
  },
};

async function toolCallingExample() {
  const { text, toolCalls, toolResults } = await generateText({
    model: vllm('NousResearch/Hermes-2-Pro-Llama-3-8B'),
    messages: [
      {
        role: 'user',
        content: 'What\'s the weather in Tokyo and calculate 25 * 4 + 10?',
      },
    ],
    tools,
    toolChoice: 'auto',
  });

  console.log('Response:', text);
  console.log('Tool calls:', toolCalls);
  console.log('Tool results:', toolResults);
}
```

### Streaming Tool Calls

```typescript
import { streamText } from 'ai';
import { vllm } from '@/lib/ai-provider';

async function streamingToolCalls() {
  const result = await streamText({
    model: vllm('NousResearch/Hermes-2-Pro-Llama-3-8B'),
    messages: [
      {
        role: 'user',
        content: 'Search for the latest AI news and summarize it.',
      },
    ],
    tools: {
      webSearch: {
        description: 'Search the web for information',
        parameters: z.object({
          query: z.string(),
          maxResults: z.number().optional(),
        }),
        execute: async ({ query, maxResults = 5 }) => {
          // Simulate web search
          return {
            results: [
              { title: 'AI breakthrough...', url: 'https://...', snippet: '...' },
            ],
          };
        },
      },
    },
  });

  // Stream tool calls and text
  for await (const chunk of result.fullStream) {
    switch (chunk.type) {
      case 'text-delta':
        process.stdout.write(chunk.textDelta);
        break;
      case 'tool-call':
        console.log('\nTool called:', chunk.toolName);
        console.log('Arguments:', chunk.args);
        break;
      case 'tool-result':
        console.log('Tool result:', chunk.result);
        break;
    }
  }
}
```

## React Server Components Integration

### Server Component Example

```tsx
// app/chat/page.tsx
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';

export default async function ChatPage({
  searchParams
}: {
  searchParams: { prompt?: string }
}) {
  if (!searchParams.prompt) {
    return <div>No prompt provided</div>;
  }

  const { text } = await generateText({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    prompt: searchParams.prompt,
    temperature: 0.7,
  });

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-2">Response:</h2>
      <p className="whitespace-pre-wrap">{text}</p>
    </div>
  );
}
```

### Streaming in Server Components

```tsx
// app/stream/page.tsx
import { streamText } from 'ai';
import { vllm } from '@/lib/ai-provider';
import { StreamingResponse } from '@/components/streaming-response';

export default async function StreamPage() {
  const result = await streamText({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    messages: [
      {
        role: 'user',
        content: 'Tell me an interesting fact about space.',
      },
    ],
  });

  return <StreamingResponse stream={result.toAIStream()} />;
}

// components/streaming-response.tsx
'use client';

import { useCompletion } from 'ai/react';

export function StreamingResponse({ stream }: { stream: ReadableStream }) {
  const { completion, isLoading } = useCompletion({
    api: '/api/chat',
    body: { stream },
  });

  return (
    <div className="p-4">
      {isLoading && <div>Generating...</div>}
      <p className="whitespace-pre-wrap">{completion}</p>
    </div>
  );
}
```

## Client-Side Integration with Hooks

### useChat Hook

```tsx
// app/chat/page.tsx
'use client';

import { useChat } from 'ai/react';

export default function ChatInterface() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/chat',
  });

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`mb-4 ${
              message.role === 'user' ? 'text-right' : 'text-left'
            }`}
          >
            <span
              className={`inline-block p-2 rounded ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200'
              }`}
            >
              {message.content}
            </span>
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="p-4 border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={handleInputChange}
            placeholder="Type your message..."
            className="flex-1 p-2 border rounded"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}

// app/api/chat/route.ts
import { streamText } from 'ai';
import { vllm } from '@/lib/ai-provider';

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = await streamText({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    messages,
    temperature: 0.7,
  });

  return result.toAIStreamResponse();
}
```

### useCompletion Hook

```tsx
// components/completion.tsx
'use client';

import { useCompletion } from 'ai/react';

export function CompletionInterface() {
  const {
    completion,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    stop,
  } = useCompletion({
    api: '/api/completion',
  });

  return (
    <div className="p-4">
      <form onSubmit={handleSubmit} className="mb-4">
        <textarea
          value={input}
          onChange={handleInputChange}
          placeholder="Start typing..."
          className="w-full p-2 border rounded"
          rows={4}
          disabled={isLoading}
        />
        <div className="mt-2 flex gap-2">
          <button
            type="submit"
            disabled={isLoading}
            className="px-4 py-2 bg-green-500 text-white rounded"
          >
            Complete
          </button>
          {isLoading && (
            <button
              type="button"
              onClick={stop}
              className="px-4 py-2 bg-red-500 text-white rounded"
            >
              Stop
            </button>
          )}
        </div>
      </form>
      {completion && (
        <div className="p-4 bg-gray-100 rounded">
          <pre className="whitespace-pre-wrap">{completion}</pre>
        </div>
      )}
    </div>
  );
}

// app/api/completion/route.ts
import { streamText } from 'ai';
import { vllm } from '@/lib/ai-provider';

export async function POST(req: Request) {
  const { prompt } = await req.json();

  const result = await streamText({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    prompt,
    temperature: 0.8,
    maxTokens: 500,
  });

  return result.toAIStreamResponse();
}
```

## Advanced Configuration

### Custom Headers and Authentication

```typescript
import { createOpenAI } from '@ai-sdk/openai';

const vllm = createOpenAI({
  baseURL: process.env.VLLM_BASE_URL,
  apiKey: process.env.VLLM_API_KEY,
  compatibility: 'strict',
  headers: {
    'Authorization': `Bearer ${process.env.CUSTOM_TOKEN}`,
    'X-Organization-Id': process.env.ORG_ID,
    'X-Request-Id': () => crypto.randomUUID(), // Dynamic header
  },
  fetch: async (url, options) => {
    // Custom fetch implementation for monitoring
    console.log('API Request:', url);
    const response = await fetch(url, options);
    console.log('API Response:', response.status);
    return response;
  },
});
```

### Provider-Specific Options

```typescript
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';

async function advancedGeneration() {
  const result = await generateText({
    model: vllm('meta-llama/Meta-Llama-3-70B-Instruct'),
    prompt: 'Explain relativity',
    temperature: 0.7,
    maxTokens: 1000,
    topP: 0.95,
    topK: 50,
    frequencyPenalty: 0.5,
    presencePenalty: 0.3,
    stopSequences: ['\n\n', 'END'],
    // vLLM-specific options via providerOptions
    providerOptions: {
      openai: {
        // Enable logprobs for debugging
        logprobs: true,
        // Set number of logprobs to return
        top_logprobs: 5,
        // Enable best_of sampling
        best_of: 3,
        // Use beam search
        use_beam_search: true,
        // Repetition penalty
        repetition_penalty: 1.1,
      },
    },
  });

  // Access logprobs from metadata
  const logprobs = result.providerMetadata?.openai?.logprobs;
  console.log('Logprobs:', logprobs);
}
```

### Multiple Model Configurations

```typescript
// lib/models.ts
import { createOpenAI } from '@ai-sdk/openai';

// Different vLLM instances for different models/purposes
export const vllmChat = createOpenAI({
  baseURL: process.env.VLLM_CHAT_URL,
  apiKey: process.env.VLLM_API_KEY,
  compatibility: 'strict',
  name: 'vllm-chat',
});

export const vllmCode = createOpenAI({
  baseURL: process.env.VLLM_CODE_URL,
  apiKey: process.env.VLLM_API_KEY,
  compatibility: 'strict',
  name: 'vllm-code',
});

export const vllmEmbedding = createOpenAI({
  baseURL: process.env.VLLM_EMBEDDING_URL,
  apiKey: process.env.VLLM_API_KEY,
  compatibility: 'strict',
  name: 'vllm-embedding',
});

// Model instances
export const models = {
  chat: vllmChat('meta-llama/Meta-Llama-3-8B-Instruct'),
  code: vllmCode('codellama/CodeLlama-34b-Instruct'),
  embedding: vllmEmbedding.textEmbedding('BAAI/bge-large-en-v1.5'),
};
```

## Guided Generation with vLLM

### JSON Schema Constraints

```typescript
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';

async function guidedGeneration() {
  // vLLM with guided generation enabled
  const result = await generateText({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    prompt: 'Generate user profile data',
    // Pass guided generation parameters via provider options
    providerOptions: {
      openai: {
        guided_json: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            age: { type: 'integer', minimum: 0, maximum: 120 },
            email: { type: 'string', format: 'email' },
            interests: {
              type: 'array',
              items: { type: 'string' },
              minItems: 1,
              maxItems: 5,
            },
          },
          required: ['name', 'age', 'email'],
        },
      },
    },
  });

  const userData = JSON.parse(result.text);
  console.log('Generated user:', userData);
}
```

### Regex Patterns

```typescript
async function regexConstrainedGeneration() {
  const result = await generateText({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    prompt: 'Generate a valid US phone number',
    providerOptions: {
      openai: {
        guided_regex: '^\\d{3}-\\d{3}-\\d{4}$',
      },
    },
  });

  console.log('Phone number:', result.text);
}
```

### Choice Constraints

```typescript
async function choiceGeneration() {
  const result = await generateText({
    model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
    prompt: 'What is the sentiment of this review: "Amazing product!"',
    providerOptions: {
      openai: {
        guided_choice: ['positive', 'negative', 'neutral'],
      },
    },
  });

  console.log('Sentiment:', result.text);
}
```

## Error Handling and Retries

### Comprehensive Error Handling

```typescript
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';
import {
  APICallError,
  RetryError,
  InvalidResponseDataError
} from 'ai';

async function robustGeneration(prompt: string) {
  try {
    const result = await generateText({
      model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
      prompt,
      maxRetries: 3,
      abortSignal: AbortSignal.timeout(30000), // 30s timeout
    });

    return result.text;
  } catch (error) {
    if (error instanceof APICallError) {
      console.error('API call failed:', {
        url: error.url,
        status: error.statusCode,
        message: error.message,
        responseBody: error.responseBody,
      });

      // Handle specific status codes
      switch (error.statusCode) {
        case 429:
          console.log('Rate limited, waiting before retry...');
          await new Promise(resolve => setTimeout(resolve, 5000));
          return robustGeneration(prompt); // Retry
        case 503:
          console.log('Service unavailable, vLLM might be loading model');
          break;
        default:
          throw error;
      }
    } else if (error instanceof RetryError) {
      console.error('Max retries exceeded:', error.message);
      console.error('Last error:', error.cause);
    } else if (error instanceof InvalidResponseDataError) {
      console.error('Invalid response format:', error.message);
    } else {
      console.error('Unknown error:', error);
    }

    throw error;
  }
}
```

### Custom Retry Logic

```typescript
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';

class VLLMClient {
  private baseURL: string;
  private maxRetries: number;

  constructor(baseURL: string, maxRetries = 3) {
    this.baseURL = baseURL;
    this.maxRetries = maxRetries;
  }

  async generateWithFallback(
    prompt: string,
    models: string[]
  ): Promise<string> {
    for (const model of models) {
      try {
        const provider = createOpenAI({
          baseURL: this.baseURL,
          apiKey: process.env.VLLM_API_KEY,
          compatibility: 'strict',
        });

        const result = await generateText({
          model: provider(model),
          prompt,
          maxRetries: this.maxRetries,
        });

        return result.text;
      } catch (error) {
        console.log(`Model ${model} failed, trying next...`);
        continue;
      }
    }

    throw new Error('All models failed');
  }
}

// Usage
const client = new VLLMClient('http://localhost:8000/v1');
const text = await client.generateWithFallback(
  'Hello, world!',
  [
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.2',
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
  ]
);
```

## Monitoring and Observability

### Request Logging

```typescript
import { createOpenAI } from '@ai-sdk/openai';

const vllmWithLogging = createOpenAI({
  baseURL: process.env.VLLM_BASE_URL,
  apiKey: process.env.VLLM_API_KEY,
  compatibility: 'strict',
  fetch: async (url, options) => {
    const startTime = Date.now();
    const requestId = crypto.randomUUID();

    console.log('vLLM Request:', {
      requestId,
      url,
      method: options?.method,
      timestamp: new Date().toISOString(),
    });

    try {
      const response = await fetch(url, options);
      const duration = Date.now() - startTime;

      console.log('vLLM Response:', {
        requestId,
        status: response.status,
        duration: `${duration}ms`,
        headers: Object.fromEntries(response.headers.entries()),
      });

      return response;
    } catch (error) {
      console.error('vLLM Error:', {
        requestId,
        error: error.message,
        duration: `${Date.now() - startTime}ms`,
      });
      throw error;
    }
  },
});
```

### Metrics Collection

```typescript
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';

interface Metrics {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  latency: number;
  tokensPerSecond: number;
  model: string;
  temperature: number;
}

async function generateWithMetrics(
  prompt: string,
  model: string
): Promise<{ text: string; metrics: Metrics }> {
  const startTime = Date.now();

  const result = await generateText({
    model: vllm(model),
    prompt,
    temperature: 0.7,
  });

  const latency = Date.now() - startTime;
  const tokensPerSecond =
    result.usage.completionTokens / (latency / 1000);

  const metrics: Metrics = {
    promptTokens: result.usage.promptTokens,
    completionTokens: result.usage.completionTokens,
    totalTokens: result.usage.totalTokens,
    latency,
    tokensPerSecond,
    model,
    temperature: 0.7,
  };

  // Send to monitoring service
  await sendMetrics(metrics);

  return { text: result.text, metrics };
}

async function sendMetrics(metrics: Metrics) {
  // Send to Prometheus, DataDog, etc.
  console.log('Metrics:', metrics);
}
```

## Production Best Practices

### 1. Load Balancing Multiple vLLM Instances

```typescript
// lib/vllm-load-balancer.ts
import { createOpenAI } from '@ai-sdk/openai';

class VLLMLoadBalancer {
  private instances: ReturnType<typeof createOpenAI>[];
  private currentIndex: number = 0;
  private healthStatus: Map<string, boolean>;

  constructor(urls: string[]) {
    this.instances = urls.map(url =>
      createOpenAI({
        baseURL: url,
        apiKey: process.env.VLLM_API_KEY,
        compatibility: 'strict',
      })
    );
    this.healthStatus = new Map(urls.map(url => [url, true]));
    this.startHealthChecks();
  }

  private startHealthChecks() {
    setInterval(async () => {
      for (let i = 0; i < this.instances.length; i++) {
        try {
          // Simple health check
          await fetch(`${this.instances[i].baseURL}/health`);
          this.healthStatus.set(this.instances[i].baseURL, true);
        } catch {
          this.healthStatus.set(this.instances[i].baseURL, false);
        }
      }
    }, 30000); // Check every 30 seconds
  }

  getNextInstance() {
    // Round-robin with health checking
    const startIndex = this.currentIndex;
    do {
      this.currentIndex = (this.currentIndex + 1) % this.instances.length;
      if (this.healthStatus.get(this.instances[this.currentIndex].baseURL)) {
        return this.instances[this.currentIndex];
      }
    } while (this.currentIndex !== startIndex);

    throw new Error('No healthy vLLM instances available');
  }

  async generateText(params: any) {
    const instance = this.getNextInstance();
    return generateText({
      ...params,
      model: instance(params.model),
    });
  }
}

// Usage
const balancer = new VLLMLoadBalancer([
  'http://vllm1:8000/v1',
  'http://vllm2:8000/v1',
  'http://vllm3:8000/v1',
]);
```

### 2. Caching Responses

```typescript
import { LRUCache } from 'lru-cache';
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';
import crypto from 'crypto';

class CachedVLLMClient {
  private cache: LRUCache<string, string>;

  constructor() {
    this.cache = new LRUCache({
      max: 500, // Maximum cache entries
      ttl: 1000 * 60 * 60, // 1 hour TTL
      updateAgeOnGet: true,
    });
  }

  private getCacheKey(params: any): string {
    const normalized = {
      model: params.model,
      messages: params.messages,
      temperature: params.temperature,
      maxTokens: params.maxTokens,
    };
    return crypto
      .createHash('sha256')
      .update(JSON.stringify(normalized))
      .digest('hex');
  }

  async generateText(params: any): Promise<string> {
    const cacheKey = this.getCacheKey(params);

    // Check cache
    const cached = this.cache.get(cacheKey);
    if (cached) {
      console.log('Cache hit');
      return cached;
    }

    // Generate new response
    const result = await generateText({
      ...params,
      model: vllm(params.model),
    });

    // Cache the result
    this.cache.set(cacheKey, result.text);

    return result.text;
  }
}
```

### 3. Rate Limiting

```typescript
import { RateLimiter } from 'limiter';
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';

class RateLimitedVLLMClient {
  private limiter: RateLimiter;

  constructor(requestsPerMinute: number = 60) {
    this.limiter = new RateLimiter({
      tokensPerInterval: requestsPerMinute,
      interval: 'minute',
      fireImmediately: true,
    });
  }

  async generateText(params: any): Promise<string> {
    // Wait for rate limit token
    await this.limiter.removeTokens(1);

    return generateText({
      ...params,
      model: vllm(params.model),
    });
  }
}
```

### 4. Environment-Specific Configuration

```typescript
// lib/vllm-config.ts
interface VLLMConfig {
  baseURL: string;
  apiKey: string;
  model: string;
  maxRetries: number;
  timeout: number;
}

function getVLLMConfig(): VLLMConfig {
  const env = process.env.NODE_ENV;

  const configs: Record<string, VLLMConfig> = {
    development: {
      baseURL: 'http://localhost:8000/v1',
      apiKey: 'dev-key',
      model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
      maxRetries: 3,
      timeout: 30000,
    },
    staging: {
      baseURL: process.env.STAGING_VLLM_URL!,
      apiKey: process.env.STAGING_API_KEY!,
      model: 'meta-llama/Meta-Llama-3-8B-Instruct',
      maxRetries: 5,
      timeout: 60000,
    },
    production: {
      baseURL: process.env.PROD_VLLM_URL!,
      apiKey: process.env.PROD_API_KEY!,
      model: 'meta-llama/Meta-Llama-3-70B-Instruct',
      maxRetries: 5,
      timeout: 120000,
    },
  };

  return configs[env] || configs.development;
}

export const vllmConfig = getVLLMConfig();
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Connection Refused

```typescript
// Problem: ECONNREFUSED error
// Solution: Check vLLM server is running and accessible

async function checkVLLMConnection() {
  try {
    const response = await fetch(`${process.env.VLLM_BASE_URL}/health`);
    if (response.ok) {
      console.log('vLLM server is healthy');
    } else {
      console.error('vLLM server returned:', response.status);
    }
  } catch (error) {
    console.error('Cannot connect to vLLM:', error);
    // Start vLLM server or check firewall settings
  }
}
```

#### 2. Model Not Found

```typescript
// Problem: Model not found error
// Solution: List available models

async function listAvailableModels() {
  const response = await fetch(`${process.env.VLLM_BASE_URL}/v1/models`);
  const data = await response.json();
  console.log('Available models:', data.data);
  return data.data.map(m => m.id);
}
```

#### 3. Out of Memory

```typescript
// Problem: CUDA out of memory
// Solution: Reduce batch size or use smaller model

const vllmOptimized = createOpenAI({
  baseURL: process.env.VLLM_BASE_URL,
  apiKey: process.env.VLLM_API_KEY,
  compatibility: 'strict',
});

// Use smaller batches
async function generateWithReducedMemory(prompts: string[]) {
  const batchSize = 5; // Smaller batch size
  const results = [];

  for (let i = 0; i < prompts.length; i += batchSize) {
    const batch = prompts.slice(i, i + batchSize);
    const batchResults = await Promise.all(
      batch.map(prompt =>
        generateText({
          model: vllmOptimized('model'),
          prompt,
          maxTokens: 100, // Limit token generation
        })
      )
    );
    results.push(...batchResults);
  }

  return results;
}
```

#### 4. Compatibility Mode Issues

```typescript
// Problem: Features not working as expected
// Solution: Ensure strict compatibility mode

const vllmStrict = createOpenAI({
  baseURL: process.env.VLLM_BASE_URL,
  apiKey: process.env.VLLM_API_KEY,
  compatibility: 'strict', // IMPORTANT: Must be 'strict' for full compatibility
});

// Verify compatibility
async function verifyCompatibility() {
  try {
    const result = await generateText({
      model: vllmStrict('model'),
      prompt: 'Test',
      // These features require strict compatibility:
      providerOptions: {
        openai: {
          logprobs: true, // Requires strict mode
        },
      },
    });

    console.log('Usage stats:', result.usage); // Works with strict mode
  } catch (error) {
    console.error('Compatibility issue:', error);
  }
}
```

## Performance Optimization Tips

### 1. Enable vLLM Optimizations

```bash
# Start vLLM with optimizations
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-prefix-caching \      # Cache common prefixes
  --enable-chunked-prefill \      # Better handling of long prompts
  --use-v2-block-manager \        # Improved memory management
  --max-parallel-loading-workers 4 \  # Faster model loading
  --gpu-memory-utilization 0.95  # Maximize GPU usage
```

### 2. Optimize Client Configuration

```typescript
const optimizedVLLM = createOpenAI({
  baseURL: process.env.VLLM_BASE_URL,
  apiKey: process.env.VLLM_API_KEY,
  compatibility: 'strict',
  // Add keep-alive for connection reuse
  fetch: (url, options) => {
    return fetch(url, {
      ...options,
      keepalive: true,
      // Add compression
      headers: {
        ...options?.headers,
        'Accept-Encoding': 'gzip, deflate',
      },
    });
  },
});
```

### 3. Batch Processing

```typescript
import { generateText } from 'ai';
import { vllm } from '@/lib/ai-provider';

async function batchGenerate(prompts: string[]) {
  // vLLM handles batching internally for better throughput
  const results = await Promise.all(
    prompts.map(prompt =>
      generateText({
        model: vllm('meta-llama/Meta-Llama-3-8B-Instruct'),
        prompt,
        temperature: 0.7,
      })
    )
  );

  return results.map(r => r.text);
}
```

## Conclusion

The integration of Vercel AI SDK with vLLM provides a powerful, flexible solution for deploying LLMs in production. Key advantages include:

1. **Seamless Integration**: Drop-in replacement for OpenAI with minimal code changes
2. **Full Feature Support**: Streaming, tool calling, structured outputs all work out-of-the-box
3. **Production Ready**: Built-in error handling, retries, and monitoring
4. **Cost Effective**: Self-hosted infrastructure with predictable costs
5. **Privacy Compliant**: Keep sensitive data on-premises
6. **Performance**: vLLM's optimizations provide excellent throughput

By following this guide and best practices, you can build robust, scalable AI applications that leverage the best of both worlds: Vercel AI SDK's developer experience and vLLM's performance.

## Resources

- [vLLM Documentation](https://docs.vllm.ai)
- [Vercel AI SDK Documentation](https://sdk.vercel.ai)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Vercel AI SDK GitHub](https://github.com/vercel/ai)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)