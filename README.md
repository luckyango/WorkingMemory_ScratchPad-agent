# WorkingMemory ScratchPad Agent

A lightweight AI Agent prototype for multi-step reasoning tasks. The project explores how an explicit Scratchpad-style working memory can help an LLM preserve intermediate results, reduce repeated reasoning, and maintain state consistency across multiple reasoning steps.

The current implementation uses Python and OpenAI Chat Completions tool calling. The agent can save intermediate findings, read previous results, list available memory keys, and dynamically inject the latest Scratchpad state into the system prompt. This creates a simple but complete reasoning loop: think, store, retrieve, and continue reasoning.

## Highlights

- Built a pluggable Scratchpad working memory module for structured short-term state management.
- Exposed memory operations through OpenAI tool calling, allowing the model to decide when to save or retrieve intermediate results.
- Implemented a multi-step agent loop that refreshes the system prompt with the latest memory state before each model call.
- Validated the mechanism with a mock financial analysis task involving calculations, comparison, anomaly detection, and final summarization.
- Kept the codebase intentionally lightweight so it can be extended into persistent memory, task planning, trace visualization, RAG, or tool orchestration systems.

## Tech Stack

- **Language**: Python
- **LLM API**: OpenAI Chat Completions
- **Agent Patterns**: Tool Calling, Working Memory, Multi-step Reasoning Loop
- **Data Handling**: JSON, Python dict/list
- **Engineering Focus**: State management, prompt injection, tool schema design, agent control flow

## Core Components

### 1. Scratchpad Working Memory

The `Scratchpad` class manages task-level short-term memory:

- `write(key, value, description)`: Saves an intermediate result or important finding.
- `read(key)`: Reads a previously saved value.
- `list_keys()`: Lists all currently available memory keys.
- `to_prompt_text()`: Converts structured memory into model-readable prompt text.
- `clear()`: Clears memory at the beginning of a new task to avoid cross-task contamination.

This component acts as an explicit state layer for the agent. Instead of relying only on conversation history, important reusable information is stored in a structured object with a clear lifecycle.

### 2. Tool Calling Interface

`ScratchpadAgent._tools_for_scratchpad()` exposes Scratchpad operations as model-callable tools:

- `save_to_scratchpad`
- `read_from_scratchpad`
- `list_scratchpad_keys`

The model does not directly mutate Python objects. It requests memory operations through a tool protocol, and the program executes those operations. This design is closer to real-world agent systems where LLMs interact with databases, APIs, caches, search tools, and business services through controlled tool interfaces.

### 3. Multi-step Agent Runtime

`ScratchpadAgent.solve()` implements a simplified agent runtime:

1. Clear memory from the previous task.
2. Build initial system and user messages.
3. Call the model with access to Scratchpad tools.
4. Execute tool calls and append tool results back into the message history.
5. Refresh the system prompt before each model call so the model sees the latest Scratchpad state.
6. Return the final answer once the model stops requesting tools.

This flow demonstrates several core engineering skills in agent development: message management, tool execution, loop control, state synchronization, and termination handling.

## Example Task

The project includes a mock Q1 financial analysis scenario. The agent is asked to:

- Calculate the profit margin for each product.
- Identify the fastest-growing product.
- Detect products with abnormally low profit margins below 20%.
- Generate a final analysis summary.

This scenario demonstrates why Scratchpad memory is useful. Intermediate values such as profit, profit margin, growth rate, and anomaly labels can be saved step by step and reused later, instead of forcing the model to repeatedly infer them from the raw input.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variable

```bash
export OPENAI_API_KEY="your_api_key"
```

You can also copy `.env.example` as a reference when configuring your local environment.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

### 3. Run the demo

```bash
python examples/financial_analysis.py
```

### 4. Run tests

```bash
python -m unittest discover -s tests
```

## Project Structure

```text
.
|-- README.md
|-- ScratchPad_Agent.py
|-- examples/
|   `-- financial_analysis.py
|-- tests/
|   `-- test_scratchpad.py
|-- requirements.txt
`-- .env.example
```

Main classes:

- `Scratchpad`: Manages short-term working memory.
- `ScratchpadAgent`: Manages prompts, tool definitions, tool execution, and the reasoning loop.

Example entry point:

- `examples/financial_analysis.py`: Runs the mock Q1 financial analysis scenario.

Tests:

- `tests/test_scratchpad.py`: Covers memory read/write behavior, prompt formatting, and Scratchpad tool execution.

## Interview Talking Points

### Why Build a Scratchpad?

LLMs can struggle with complex multi-step tasks because:

- Intermediate results are mixed into natural language conversation history.
- Later steps may forget, misread, or recompute earlier results.
- Long contexts can introduce noise and make important state harder to reuse reliably.

The Scratchpad addresses this by separating reusable intermediate state from ordinary conversation text and managing it as structured data.

### Why Use Tool Calling?

Tool calling lets the model decide when it needs to read or write memory, while the actual state mutation is handled by deterministic program logic. This is more reliable than asking the model to maintain state purely in free-form text, and it mirrors how production agents call tools, APIs, databases, and internal services.

### Engineering Skills Demonstrated

- Translating an AI Agent concept into a runnable engineering prototype.
- Understanding OpenAI tool calling message flow and execution semantics.
- Designing a lightweight state management layer for LLM workflows.
- Synchronizing external state with prompt context across multiple model calls.
- Building a focused mock case to validate the agent mechanism beyond a prompt-only demo.
- Identifying clear paths toward production hardening, including persistence, testing, observability, and modular architecture.

### Current Design Trade-offs

- The Scratchpad currently uses an in-memory dictionary. This is simple and effective for prototyping, but it does not persist across process restarts.
- The agent loop uses a fixed `max_steps=10` to avoid infinite loops. A production version could use a more explicit task state machine or planner.
- The current tool set only covers memory operations. Future versions could add calculator tools, file operations, web search, database queries, or domain-specific business tools.
- The demo code currently lives at the bottom of the main file. A more production-ready structure should separate core logic, examples, tests, and CLI entry points.

## Improvement Roadmap

### Short-term Improvements

- Add structured logs for each model call, tool call, tool argument, and tool result.

### Mid-term Improvements

- Add persistence with SQLite or JSONL so each task run can be replayed.
- Build a trace viewer to visualize agent steps, tool calls, and memory state transitions.
- Add more realistic demo scenarios such as customer support troubleshooting, data analysis, research summarization, or code review assistance.
- Use Pydantic to validate tool arguments and Scratchpad entries.
- Add error handling and retry logic for API failures, malformed JSON, missing tool arguments, and step-limit failures.

### Long-term Improvements

- Evolve the Scratchpad into a multi-layer memory system with working memory, task memory, and long-term memory.
- Connect a vector database or retrieval layer to recall relevant information from previous tasks.
- Split the agent runtime into planner, executor, memory manager, and tool registry modules.
- Build a Web UI or API service so the project becomes an interactive product instead of a script-only demo.
- Add an evaluation suite with metrics such as task completion rate, tool call accuracy, memory hit rate, and reasoning cost.

## Resume Description

Long version:

> Built a Working Memory Scratchpad Agent using Python and OpenAI Tool Calling. Implemented a multi-step reasoning loop where the agent can save, retrieve, and reuse intermediate results through a structured short-term memory module. The system dynamically injects the latest memory state into the prompt to improve state consistency and traceability across complex tasks. The project demonstrates tool schema design, agent control flow, state management, and validation through a financial analysis use case.

Short version:

> Built a Python-based multi-step reasoning agent with OpenAI Tool Calling and a Scratchpad working memory module, enabling structured intermediate-state management and dynamic prompt injection for more traceable LLM workflows.

## Project Positioning

This project is currently an agent mechanism prototype. Its strength is that it clearly demonstrates the core execution chain of a tool-using LLM agent while keeping the implementation small enough to explain in an interview. With stronger project structure, tests, logging, evaluation, persistence, and visualization, it can evolve from a concept demo into a more production-like AI Agent engineering project.
