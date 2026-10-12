# Langchain-learning

## Chapter 1: `chapter-1/basics.ipynb`

Calling NVIDIA's hosted LLM API (`https://integrate.api.nvidia.com/v1`) in three ways. The endpoint is OpenAI-compatible, so OpenAI-style clients work by setting `base_url`.

### Setup
- Put your NVIDIA key (`nvapi-...`) in a `.env` file as `OPENAI_API_KEY`.
- Model used: `nvidia/nemotron-3.5-lightning-30b-a3b`.

### Approaches
1. **OpenAI SDK** (`openai.OpenAI`): works with `base_url` set to the NVIDIA endpoint.
2. **LangChain `ChatOpenAI`** (`langchain-openai`): works the same way; use `llm.invoke(...).content`.
3. **`init_chat_model(model_provider="nvidia")`**: did not work. It needs `langchain-nvidia-ai-endpoints`, which isn't installed.



## Chapter 1: `chapter-1/messages.ipynb`

Builds on `basics.ipynb`: the same NVIDIA endpoint through `ChatOpenAI`, now with structured messages and prompt templates.

### Messages
`llm.invoke([...])` accepts a list of message objects:
- `SystemMessage`: sets the model's role/behavior.
- `HumanMessage`: the user's input.
- `AIMessage`: a previous model reply (imported but not used; needed to replay chat history).

### Prompt templates
Flow: template -> `.invoke({vars})` -> ready prompt -> `llm.invoke(ready_prompt)`.
- **`PromptTemplate`**: one string with `{placeholders}`, e.g. `"write a fun fact about {topic}"`. The topic comes from `input()`.
- **`ChatPromptTemplate.from_messages`**: a list of `(role, text)` pairs, giving a system + user message from one template.


## Chapter 1: `chapter-1/structured_output.ipynb`

Make the LLM return a typed object instead of free text.
- Plain `llm.invoke("... key value pair")` gives only a string, which you must parse yourself.
- Define a Pydantic `BaseModel` with `Field(description=...)`; the descriptions act as instructions to the model.
- `llm.with_structured_output(Schema)` returns a runnable whose `.invoke(...)` gives a validated `Schema` object (`result.setup`, `result.punchline`).



## Chapter 2: chains (LCEL)

A chain is `Prompt | LLM | OutputParser`, linked with the `|` operator (LangChain Expression Language).

### `1_first_langchain.ipynb`
- Manual run: `prompt.invoke(...)` -> `llm.invoke(...)` -> `.content`.
- Chain run: `prompt_template | llm | StrOutputParser()`; `StrOutputParser` extracts the text from the `AIMessage`.
- `RunnableSequence(a, b, c)` is the explicit form of `a | b | c`.
- A plain string input works when the prompt has only one variable.

### `2_chain_with_customrunnable.ipynb`
- Problem: the first LLM returns a string, but the next prompt needs a dict (`{"text": ...}`).
- `RunnableLambda(fn)` wraps any Python function so it can sit in a chain and adapt data between steps.
- Chain: question -> answer -> `{"text": answer}` -> LinkedIn post prompt -> LLM -> string.

### `3_parallel_chains.ipynb`
- `RunnableParallel(branches={...})` runs several chains on the same input at once; results come back in a dict keyed by branch name.
- Branches can be `prompt | llm | parser` chains (LinkedIn) or any function wrapped in `RunnableLambda` (Instagram).
- The output is nested (`{"branches": {...}}`); one more `RunnableLambda` flattens it to `{"linkedin": ..., "instagram": ...}`.

### `4_conditional_chains.ipynb`
- `with_structured_output` with a `Literal["positive", "Negative"]` field classifies a movie review.
- `RunnableBranch((condition, chain), ..., default_chain)` works like if / else: the first true condition runs, otherwise the default.
- Positive -> LinkedIn chain, otherwise -> Instagram chain.
- Known quirks: the branch receives only the label, not the review text, so the post is about the word "Negative". The `"positive" in x` check is case-sensitive. Passing the review along (e.g. `RunnablePassthrough` / `RunnableParallel`) would fix the first.



## Chapter 3: `chapter-3/1_ReAct_agent.ipynb`

ReAct = Reasoning + Acting. The LLM loops: reason -> call a tool -> observe the result -> repeat until it can answer.

### Tools
A tool is a function plus a name, description and input schema; the description is what the LLM uses to pick it.
1. **`DuckDuckGoSearchRun`**: web search, no API key (needs `region="us-en"` and `time=None` on the wrapper to avoid DNS errors and empty results).
2. **`WikipediaQueryRun`**: factual / background lookups.
3. **`@tool` custom function** (`enterprise_tool`): the docstring becomes the description, type hints become the schema. Here a stub that "sends an email".

### Agent
- `create_agent(model, tools=ToolKit, middleware=[...])` builds the LangGraph reason/act/observe loop.
- `@wrap_tool_call` middleware catches tool exceptions and returns them to the LLM as a `ToolMessage`, so one failed search does not crash the run.
- `agent.stream({...}, stream_mode="values")` shows each step: Human -> AI (tool call) -> Tool result -> AI answer.

### `bind_tools`
`llm.bind_tools(ToolKit)` only lets the model *request* a tool call (see `.tool_calls`); the agent loop is what actually runs the tool.



## Setup
- Python >= 3.11, dependencies in `pyproject.toml` (`uv sync`).
- `.env` with `OPENAI_API_KEY=nvapi-...` (NVIDIA key); every notebook uses the NVIDIA OpenAI-compatible endpoint.
