# LangChain v1 Multi-Step Reasoning Agent

A multi-step reasoning agent built with **LangChain v1** + **Jev** + **Flask** + **Ollama**.

User question → the routing layer determines the intent → the matching pipeline (knowledge-base
retrieval / math / weather / small talk) → a **token-by-token streaming** answer backed by sources.

**Three key features**:

| Feature | Description |
|---|---|
| **Jev decision routing** | A dedicated **decision model** makes the intent judgement, taking "which tool to use" out of the generative model. Measured routing accuracy **6/6 with confidence 1.0**, turning this step from probabilistic into deterministic. See the *Core feature: Jev decision routing* section below |
| **Streaming output** | SSE token-by-token output with a live elapsed-time counter; retrieved content is rendered in its own block so the process is visible |
| **Determinism first** | If code can do something deterministically (retrieval), the model is not asked to judge it. The model only does what it is good at — understanding semantics and extracting arguments |

> This project is a **v1 rewrite** of a classic ReAct implementation
> (Flask + LangChain classic + Ollama + RAG), used to compare the design differences between
> the two generations of Agent APIs. The original lives in `D:\agent`.

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent | LangChain v1 — `create_agent` (native tool calling) |
| Runtime underneath | LangGraph (`create_agent` returns a compiled graph) |
| Chat model | Ollama `qwen2.5:3b` |
| Embedding model | Ollama `bge-m3` (1024-dim, good for Chinese) |
| Vector store | ChromaDB, persisted to `.chroma/` |
| Routing model | Jev (TypeSafe System One) — optional; falls back to local rules when unavailable |
| Web | Flask 3 with blueprints, SSE streaming |

---

## Architecture

Dependencies flow one way: `web → services → core → config.py`.
**No HTTP-related code is allowed inside `core/`.**

```
Browser
  │  POST /query/stream   (SSE)
  ▼
web/routes.py               Interface layer: HTTP only (receive, validate, serialize)
  ▼
services/qa_service.py      Orchestration layer: route dispatch + timeout + fallback
  ▼
services/router.py          Routing layer: Jev first → local rules as fallback
  │
  ├─ knowledge  ─▶ code retrieves directly ─▶ bare LLM + RAG template   (1 LLM call, deterministic)
  ├─ calculator ─▶ single-tool agent (calculator only)                  (2 LLM calls)
  ├─ weather    ─▶ single-tool agent (get_weather only)                 (2 LLM calls)
  └─ chat       ─▶ bare LLM, no tools                                   (1 LLM call)
  ▼
core/                       Capability layer
  ├── agent_builder.py       Builds and caches one agent per route
  ├── tools.py               Three tools: calculator / get_weather / search_knowledge
  ├── rag.py                 Index building and retrieval
  ├── llm.py                 Model factory
  └── prompts.py             Four task-specific prompts + a fallback prompt
  ▼
Ollama (qwen2.5:3b for chat / bge-m3 for embeddings)
```

**Why four routes?** The local 3b model only picks the right tool out of three about **one third
of the time**. Once routing is delegated to a dedicated model, each route carries
**exactly one tool and exactly one prompt** — the model no longer has to *choose*,
it only has to *fill in arguments*.

---

## Core feature: Jev decision routing

> This is the main difference between this project and a typical Agent / RAG project.

### 1. Why it was introduced

The local `qwen2.5:3b` was being asked to do four things inside a single prompt:

```
Decide "should I use a tool?" → Choose "which one?" → Build "what arguments?" → Generate "the answer"
        └──────────── three of these four are decisions; only the last is generation ────────┘
```

What we measured: **the same question "什么是过拟合" succeeded sometimes and failed other times**,
roughly a **1 in 3** success rate. The cause was not a misconfiguration — it was the
**model's prior distribution**:

- In the training data of a general chat model, "user asks about a concept → model explains it
  directly" is **extremely common**, while "call a tool first, then answer" is **rare**
- A system prompt provides a layer of conditional bias that nudges the model towards calling tools,
  but a 3b model does not have the capacity for that nudge to **reliably outweigh the prior**
- So every generation is a tug-of-war between the nudge and the prior, and the outcome is
  **probabilistic**

Worse: **the failure is silent**. The model does not raise an error — it calmly makes up an answer
from memory.

### 2. What Jev is

**Jev = the System One decision model from TypeSafe AI** (released September 2026).

Its positioning is completely different from a general-purpose LLM:

> **A probabilistic decision model for software. It takes a state and returns typed answers with
> probabilities — it does not generate free-form text.**

| | A general LLM making decisions | **Jev** |
|---|---|---|
| Output shape | Generated text / structured JSON | **Typed fields + probabilities** |
| Parsing required? | Yes, and you still have to handle malformed output | **No** |
| Confidence available? | No | **Yes — `confidence` plus a full probability distribution** |
| Can it hallucinate? | Yes | **It generates no text, so there is no room to hallucinate** |
| Multiple judgements | Stuffed into one prompt, contaminating each other | **Evaluated in parallel, no interference** |
| Best fit | General conversation and generation | **Judgement only** |

It supports three question types: `Choice`, `Noul` (yes/no probability) and `Score` (ordinal).
This project only uses `Choice`.

### 3. Request and response

**Request** (the `_QUESTIONS` dict in `services/router.py`):

```json
{
  "model": "jev-latest",
  "state": "什么是过拟合",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which way should this sentence be handled?",
      "criteria": {
        "knowledge":  "A question about machine learning concepts, terminology, principles or methods",
        "calculator": "A question that requires a mathematical calculation",
        "weather":    "A question asking about the weather somewhere",
        "chat":       "Small talk, greetings, or anything unrelated to the above"
      }
    }
  }
}
```

> `criteria` is **the candidate set you define yourself** — the model can only pick from it and
> **cannot invent a fifth option**. That is far more reliable than letting a model free-form an
> answer and then parsing it.

**Response** (measured, model version `jev-1.13.0`):

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "knowledge",
      "probabilities": {"knowledge": 1.0, "chat": 0.0, "weather": 0.0, "calculator": 0.0},
      "confidence": 1.0
    }
  },
  "usage": {"input_tokens": 395, "output_tokens": 45}
}
```

**There is not a single character of generated text in the whole response** — those 45 output tokens
*are* the label.

| Field | Use |
|---|---|
| `choice` | The selected branch, **directly usable** |
| `probabilities` | Per-candidate probability — tells you whether it was hesitating |
| `confidence` | How concentrated the distribution is (1.0 = completely certain) |

### 4. Integration: two-tier routing

```
User question
   │
   ▼
┌──────────────────────────────────────┐
│ services/router.py                   │
│                                      │
│  ① Jev (primary)                     │
│     confidence >= 0.6 → accept       │
│          │                           │
│          └─ failure / low confidence │
│                            ▼         │
│  ② Local keyword rules (fallback)    │
│     weather words  → weather         │
│     verb + digits  → calculator      │
│     greetings      → chat            │
│     everything else → knowledge      │
└──────────────────────────────────────┘
   │
   ▼  intent
Four pipelines (knowledge / calculator / weather / chat)
```

```python
def route(question):
    intent, confidence = _route_by_jev(question)        # 1. Jev first
    if intent and confidence >= JEV_MIN_CONFIDENCE:
        return intent, confidence

    intent, confidence = _route_by_rules(question)      # 2. Local rules as fallback
    return intent, confidence
```

**Three design principles**:

| Principle | Explanation |
|---|---|
| **Loose coupling** | Jev **only outputs a label** — it never touches prompts, tools, or generation. Swapping the main model or adding a tool does not affect it |
| **No blind trust** | When `confidence < 0.6` we **do not accept** the answer and fall back to rules — better not to route at all than to route wrongly |
| **Degrade to the tier you are sure about** | If Jev fails we **never** fall back to "a three-tool agent deciding on its own" (only ~1/3 correct). We fall back to the **always-reliable** local rules |

### 5. Results

#### ① Routing accuracy: 6 / 6, all with confidence 1.0

`check_jev.py` tested six Chinese questions:

| Question | Expected | Predicted | Confidence | Latency |
|---|---|---|---|---|
| 什么是过拟合 | knowledge | **knowledge** | 1.0 | 1.1s |
| 正则化有什么作用 | knowledge | **knowledge** | 1.0 | 1.1s |
| 请计算 sqrt(16) + 2**8 | calculator | **calculator** | 1.0 | 1.1s |
| 北京今天天气怎么样 | weather | **weather** | 1.0 | 2.3s |
| 你好啊 | chat | **chat** | 1.0 | 1.0s |
| 模型选择是什么意思 | knowledge | **knowledge** | 1.0 | 1.0s |

Result: **6 / 6 correct**.

**Look at that confidence column — all 1.0, and every non-selected probability is 0.0.**
It is not "guessing right"; it is **completely certain**. That is a different thing from a local 3b
model oscillating between two options (similar probabilities on both sides, argmax flipping at random).

#### ② The decision step: from probabilistic to deterministic

| Capability | Before Jev (agent decides on its own) | After Jev |
|---|---|---|
| Which pipeline to take | Judged on the fly, measured **~1/3** correct | Judged by Jev, **6/6** correct |
| Result for the same question | **Right sometimes, wrong other times** (`temperature=0` does not help) | **Identical every time** |
| Is certainty visible? | No | **`confidence` plus a full probability distribution** |
| Behaviour on failure | **Silently picks wrong** — the user cannot tell | Returns low confidence, so it **can be intercepted** |
| Decision latency | Buried inside generation, cannot be measured separately | **1.0 – 2.3 s** on its own |
| Cost | 0 | ~**440 tokens per call** |
| External dependency | None | **Yes** (overseas service, needs a proxy) |

#### ③ It enables the four pipelines, which directly improves RAG usability

Jev's decision is the precondition for "each route carries one tool".
With it in place, the model no longer has to make a three-way choice.

| Metric | Before | After |
|---|---|---|
| Retrieval rate on the knowledge route | **0 / 4** | **100%** |
| Latency for a concept question | ~33 – 37 s | **19.7 s** |
| The "knowledge retrieved" block in the UI | Intermittent (depended on whether the model called the tool that time) | **Appears consistently** |
| Prompts | One prompt covering every case | **One prompt per case** |
| Chance of picking the wrong tool | Three-way choice, ~1/3 wrong | **One-to-one, near-zero** |

> Note: the jump from 0/4 to 100% in retrieval rate is the combined result of
> **Jev routing** *and* **rewriting the knowledge route to pre-retrieve** —
> Jev is responsible for sending the question down the right path, and pre-retrieval is responsible
> for making retrieval actually happen. Neither works without the other.

#### ④ End-to-end measurements (`check_e2e.py`, 24 / 24 passing)

| Question | Route hit | Tool content | Time to first token | Server-side latency |
|---|---|---|---|---|
| 什么是过拟合 | knowledge | retrieved **605 chars** | 14.1s | **19.7s** |
| 请计算 sqrt(16) + 2**8 | calculator | `260.0` | 5.9s | 7.2s |
| 北京今天天气怎么样 | weather | live weather | 5.7s | 7.4s |
| 你好啊 | chat | no tool | — | ~1s |

### 6. How to enable it

**It runs without any configuration** — when Jev is unavailable the router automatically degrades to
local keyword rules and functionality is unaffected.

To enable Jev routing:

```powershell
[Environment]::SetEnvironmentVariable("TYPESAFE_API_KEY", "your-key", "User")
```

Get an API key from https://console.typesafe.ai (a trial quota is included).

> ⚠️ Two known caveats:
> 1. **You must restart your IDE / terminal after setting the variable** — a process only inherits
>    the environment snapshot taken when it started. Otherwise `app.py` will not see the key and the
>    log will show `[router] 未配置 TYPESAFE_API_KEY`.
> 2. **Jev is an overseas service and needs a proxy.** If some process reports `SSLEOFError`, it is
>    almost certainly because that process did not inherit the proxy variables
>    (`requests` only reads proxy settings from the process environment). The router then degrades to
>    local rules and the service keeps working — which is precisely the point of not putting it on
>    the critical path.

---

## Quick start

### 1. Pull the models

```bash
ollama pull qwen2.5:3b
ollama pull bge-m3
```

### 2. Install dependencies

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 3. Prepare the knowledge base

The repository **does not include** PDFs (they are large). Put a well-structured PDF into `knowledge/`.

Recommended demo document: **《动手学深度学习》 (Dive into Deep Learning)** —
https://zh-v2.d2l.ai/d2l-zh.pdf
(The bundled `scripts/split_pdf.py` can split a big PDF by chapter.)

Then build the index (slow — it calls the embedding model once per chunk):

```bash
venv/Scripts/python.exe scripts/build_index.py
```

> **Retrieval-time** parameters such as `RETRIEVE_K` and the `page_content` truncation length take
> effect immediately. **Index-time** parameters such as `chunk_size` and `separators` require
> deleting `.chroma/` and rebuilding.

### 4. Configure Jev routing (optional, but recommended)

**It runs without this** — if Jev is unavailable the router degrades to local keyword rules and
functionality is unaffected.

To enable Jev routing:

```powershell
[Environment]::SetEnvironmentVariable("TYPESAFE_API_KEY", "your-key", "User")
```

Get an API key from https://console.typesafe.ai (a trial quota is included).

> You **must restart PyCharm / your terminal** afterwards — a process only inherits the environment
> snapshot taken at startup. For the rationale, the measured results and the two known caveats,
> see *Core feature: Jev decision routing* above.

### 5. Start the service

```bash
venv/Scripts/python.exe app.py
```

Then open http://127.0.0.1:5023/

### 6. Run the tests

```bash
# Smoke tests (no AI, no network)
venv/Scripts/python.exe -B -m pytest tests/ -v -p no:cacheprovider

# End-to-end tests (real HTTP + SSE, on its own port 5099, shuts down automatically)
venv/Scripts/python.exe -B check_e2e.py
```

---

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Front-end page |
| POST | `/query` | Non-streaming Q&A, returns `{answer, elapsed, used_agent}` |
| **POST** | **`/query/stream`** | **Streaming Q&A over SSE** |
| GET | `/health` | Health check |

**SSE event protocol**:

```
data: {"type": "start"}
data: {"type": "tool", "name": "search_knowledge", "content": "...retrieved chunks..."}
data: {"type": "reset"}
data: {"type": "token", "text": "过拟合"}
data: {"type": "done", "elapsed": 19.7}
```

| Event | Purpose |
|---|---|
| `tool` | The front end renders the "knowledge retrieved" block from this |
| `reset` | **Discards text already displayed** (the first LLM round's filler) |
| `token` | An answer delta |

---

## Directory structure

```
agent_remake/
├── app.py                        Entry point: create_app() factory
├── config.py                     All configuration (paths anchored to the project root)
├── README.md                     Chinese README
├── README_EN.md                  English README (this file)
├── requirements.txt
├── core/                         Capability layer (must not import flask)
│   ├── llm.py                    Model factory (the only place LLMs are created)
│   ├── prompts.py                Four task-specific prompts + SYSTEM_PROMPT fallback
│   ├── tools.py                  The three tools
│   ├── rag.py                    Index building and retrieval
│   └── agent_builder.py          Builds one agent per route (cached)
├── services/
│   ├── router.py                 Routing layer: Jev + local rules
│   └── qa_service.py             Orchestration: dispatch + timeout + fallback
├── web/
│   ├── routes.py                 Blueprint: / /query /query/stream /health
│   └── templates/index.html      Front-end page (streaming render + elapsed badge)
├── scripts/
│   ├── build_index.py            One-off vector index build
│   ├── split_pdf.py              Split a large PDF by chapter
│   └── peek_pdf.py               Inspect extracted text quality / locate keywords
├── tests/
│   └── test_smoke.py             Smoke tests (10 cases)
├── check_e2e.py                  End-to-end tests (24 assertions)
├── check_route.py                Routing + streaming event regression
├── check_*.py                    Other development-time diagnostic scripts (safe to delete)
├── knowledge/                    Put PDFs here (not committed)
└── .chroma/                      Vector store (not committed, rebuildable)
```

---

## Design decisions

- **One-way layering**: dependencies flow `web → services → core`; no HTTP code inside `core/`
- **Tiered routing**: Jev only outputs a label — it never touches prompts, tools or generation
  (loose coupling). When it is unavailable the router falls back to **local keyword rules**, and
  **not** to "a three-tool agent deciding on its own" — the latter is only ~1/3 accurate
- **Determinism first**: whatever code can do deterministically (retrieval) is not handed to the
  model to judge. The model only does what it is good at — understanding semantics and extracting
  arguments from natural language
- **Single-tool routes**: each route binds exactly one tool, downgrading the model's "pick one of
  three" into a "one-to-one match"
- **Index built offline**: vectorisation is expensive, so it runs via `scripts/build_index.py`
  separately and never inside a web request
- **Lazy loading + degradation**: when no index exists, `get_retriever()` returns `None` and RAG
  degrades to "tool unavailable" while the service still starts normally
- **Tools swallow their own exceptions**: an escaping exception aborts the whole agent reasoning
  round, so tools must convert errors into strings
- **Chinese-aware chunking**: `separators` explicitly includes Chinese punctuation, and PDF
  "visual line breaks" are cleaned up beforehand
- **Anchored paths**: every directory is derived from
  `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` instead of relying on the current working
  directory
- **No buffering while streaming**: tokens go out immediately; when something must be retracted we
  emit a `reset` event (the earlier "buffer first, flush later" approach degraded every answer that
  did not use a tool into a single dump)

---

## Development history

Five hurdles cleared over three days (2026-09-20 → 09-22).
Below is what changed, why, and what came of it.

### Stage 1 · Sep 20 — Rebuilding the project (classic → v1)

Thirteen files were rewritten in dependency order:

| Batch | Files |
|---|---|
| 1 | `config.py`, `core/__init__.py`, `core/llm.py`, `core/prompts.py` |
| 2 | `core/rag.py`, `core/tools.py`, `core/agent_builder.py` |
| 3 | `services/qa_service.py`, `web/routes.py`, `app.py` |
| 4 | `requirements.txt`, `tests/test_smoke.py` |

**Three pitfalls hit along the way**:

| Pitfall | Symptom | Root cause and fix |
|---|---|---|
| Models swapped | `"bge-m3" does not support chat` | `CHAT_MODEL` and `EMBED_MODEL` were written the wrong way round. **Models have specific roles — an embedding model cannot chat** |
| Missing dependencies | `No module named 'pypdf'` / `'chromadb'` | Installed them |
| **Relative paths** | An empty `knowledge/` appeared under `scripts/` | `KNOWLEDGE_DIR = "knowledge"` depends on the **process's current working directory**, and PyCharm runs a script with the CWD set to the script's folder. Fixed by anchoring to `BASE_DIR` |

### Stage 2 · Sep 20 — `llama3.2:1b` is unusable for tool calling

Three failure modes, all from the same root cause (the model cannot tell "things in the context"
apart from "what I am supposed to do"):

| # | Symptom |
|---|---|
| 1 | Filled the argument **schema definition** in as the argument value |
| 2 | Echoed tool descriptions and schema fragments inside `content` |
| 3 | **Fabricated a tool result** (after the tool errored, it invented a "successful" JSON response) |

Verified online: `llama3.2:1b` **does** support tool calling, but is unreliable with multiple tools
and when arguments must be *constructed* rather than *extracted*.
**Switching to `qwen2.5:3b` fixed it immediately** — correct argument nesting, the tool actually ran,
and the noise disappeared.

> Lesson: **a model capability problem is solved by changing the model, not by changing the code.**

### Stage 3 · Sep 22, morning — Three layers of RAG quality

**① Document layer.** Similarity scores for "什么是过拟合" were bunched at 0.23–0.32 and
**the genuinely relevant chunk only ranked second**. Adding a threshold made things *worse* —
the root cause was the source document. The original PDF was a **mind-map / card layout**
(alternating titles and blurbs, hard line breaks cutting words apart). Switching to the continuous
prose of *Dive into Deep Learning* spread the scores to **0.43–0.51 with correct ordering**.

**② Chunking layer.** Sentences were being cut in half, for two reasons: `separators` listed `\n`
before `。` (a PDF line break is not a semantic boundary), and PDF **"visual line breaks"** split
words apart (`保` / `留`). The fix: clean visual line breaks → move Chinese punctuation ahead of
newlines → drop fragments shorter than 50 characters.
Result: the shortest chunk went from 12 to **51 characters**, all three chunks now break at a full
stop, and retrieval scores improved across the board.

**③ Performance layer.** Stage-by-stage timing located the bottleneck: generation ran at
**6.4 tokens/s** while reading input ran at **165.6 tokens/s** (26× faster).
→ **The leverage ratio is 26:1, so the only effective optimisation is "make the model write less".**
Adding `num_predict=192` (an Ollama-level hard cap) plus a "keep it under 200 characters" prompt
took answers from 338 to 186 characters and a single Q&A from **61s → 33s**.

### Stage 4 · Sep 22, morning — Engineering

- Smoke tests in `tests/test_smoke.py`: **10/10 passing** (covering tools, configuration, RAG text
  cleaning, agent assembly and the web layer)
- First Git commit; repository `langchain-v1-agent`
- Wrote `README.md`

### Stage 5 · Sep 22, afternoon — Interaction and routing

**① Front end rebuilt (15 lines → ~400 lines).**
Chat bubbles, example questions, a **live elapsed counter while waiting**, an elapsed-time badge
(`<8s` green / `<45s` amber / `≥45s` red), a mode badge, and a retrieved-content block.

**② Streaming output (SSE).**
A probe script first confirmed the return shape of LangGraph's
`stream_mode=["messages","updates"]`, then `/query/stream` was implemented.
The key design is the `reset` event: an agent has two LLM rounds and the first one emits filler like
"let me look that up", so when `chunk.tool_call_chunks` appears we send `reset` and the front end
clears the bubble.

> The earlier approach was "buffer the tokens, discard them if a tool is called" —
> **which buffered every answer that did not use a tool**, degrading streaming into a single dump.
> That failed approach is worth remembering too.

**③ Jev routing.**
`Jev` (TypeSafe System One) is a **decision model that generates no text** — it returns only typed
labels with probabilities. Measured Chinese routing accuracy: **6/6, confidence 1.0**.
The integration is two-tier: Jev first, and **local keyword rules** when it is unavailable.

> Pitfall: when `app.py` was started from PyCharm it could not see the proxy and Jev failed with
> `SSLEOFError`. Probing showed that `api.typesafe.ai` itself was reachable (405 through the proxy) —
> it was purely a **process environment snapshot** problem. This incident also proved the rule that
> "you must degrade to the tier you are sure about".

**④ Rewriting the knowledge route to pre-retrieve (the single most important fix in this project)**

After the four routes were in place, the single-tool knowledge route **skipped retrieval on four
consecutive runs** (0/4), always producing the identical answer 「知识库中没有找到相关内容」.

**Root cause**: `KNOWLEDGE_PROMPT` contained the line
`资料不足以回答时，明确说明"知识库中没有找到相关内容"` — the model reasoned:
"the instructions say answers must come from the material → I have no material → so output that line",
**skipping the step of fetching the material in the first place**.

**This was the third outbreak of the same mechanism**:

| # | The line in the prompt | What the model did |
|---|---|---|
| 1 | "If a tool call fails, tell the user **检索失败**" | Output 「检索失败」 directly |
| 2 | "For small talk… **do not call a tool**" | Stopped calling tools entirely |
| 3 | "When the material is insufficient, say **知识库中没有找到相关内容**" | Output that line *and* skipped retrieval |

> ### 📌 The rule
> **Any "complete, copy-pasteable reply text" in a prompt will be treated by a weak model as a
> shortcut answer usable in the current situation.** It must be rewritten as a statement that has a
> precondition and cannot stand on its own.

**The fix**: the knowledge route now goes **code retrieves directly → push the retrieved content →
bare LLM + RAG template**, bypassing the agent entirely. The model has no room to "skip" anything.

### Data comparison across stages

| Metric | Initial | Now |
|---|---|---|
| Tool-calling success rate | ~0% (1b) / ~33% (3b, three tools) | **100%** (routing + single tool) |
| Retrieval rate on the knowledge route | 0/4 | **100%** |
| Latency per question | 60s timeout (actually >120s) | **19.7s** (concept questions) |
| Interaction | Wait 30 seconds for a single dump | **Token-by-token streaming + live counter** |
| Retrieval visibility | Logs only | **Retrieved-content block in the UI** |
| Tests | None | Smoke 10/10 + **E2E 24/24** |

---

## Lessons learned

1. **Probe before you code.** For framework APIs you are unsure about (`stream_mode` shapes, the
   `create_agent` signature), a 10-line probe script beats writing the full implementation from
   memory and then reading a traceback.

2. **Degrade to the tier you are sure about.** When an external dependency fails, falling back to
   "the old approach with a 1/3 success rate" is not a fallback at all. Fall back to something weak
   but guaranteed.

3. **When something looks wrong, check the latency first.** A complete tool call needs at least two
   LLM calls. **An "answer" that took 0.9–2.5 seconds definitely did not call a tool** — the
   cheapest possible diagnostic.

4. **Let code do what is deterministic; let the model do what needs understanding.** Retrieval needs
   no judgement (the query is the question itself) → give it to code. Extracting arguments from
   natural language (a city name, an expression) needs understanding → give it to the model.

5. **"Lines" in a prompt get treated as answers** (see the three outbreaks above).

6. **Format matters as much as wording.** After the tool description was changed from full Chinese
   sentences into a `- condition → action` arrow list, the model *stopped* calling tools — symbolic
   notation falls outside a small model's training distribution.

7. **Stop when you hit the target.** Once performance was good enough at 20 seconds, we stopped
   trading retrieval quality for another second.

8. **Do not re-run destructive operations just to verify them.** `Chroma.from_documents`
   **appends** to an existing store rather than overwriting it, so re-running `build_index.py`
   duplicates vectors — you **must delete `.chroma/` first** when changing documents or embedding
   models.

---

## How v1 differs from classic

| Dimension | classic | v1 |
|---|---|---|
| Create the agent | `create_react_agent(llm, tools, prompt)` | `create_agent(model, tools, system_prompt=...)` |
| Executor | `AgentExecutor(agent=..., tools=...)` | Does not exist; the return value of `create_agent` is directly executable |
| Call input | `{"input": q}` | `{"messages": [{"role": "user", "content": q}]}` |
| Read the result | `result["output"]` | `result["messages"][-1].content` |
| Prompts | One long ReAct format contract + few-shot examples | One prompt per task, each describing a single thing |
| Output parsing | `ReActSingleInputOutputParser` + `handle_parsing_errors` | The concept disappears (the model returns native `tool_calls`) |
| Step limit | `max_iterations=4` | `config={"recursion_limit": N}` (N counts graph steps; one tool round = 2 steps) |
| Streaming | `astream_events` | `agent.stream(..., stream_mode=["messages","updates"])` |
| Underlying mechanism | Prompt-enforced text format + regex parsing | A LangGraph graph + native structured output from the model |

The core shift: **the old path relied on "a prompt-enforced text format → regex parsing", while v1
relies on "the model returning a native structured `tool_calls` field"** — which is why the parser
became pointless.

> ⚠️ But note: **v1 demands more from the model than classic did.** classic only required the model to
> imitate a text format; v1 requires it to emit correct nested structures — at the 1B scale, the old
> approach actually works more easily.

---

## Possible next steps

| Direction | Notes |
|---|---|
| **Conversation memory** | `create_agent(..., checkpointer=InMemorySaver())` + `config={"configurable": {"thread_id": ...}}`. The project is currently **completely stateless**, so elliptical follow-ups like "天津呢" cannot be answered |
| **Citation tracing** | `Document.metadata` carries `source` and `page`; including them in the material lets answers say "from page 16" |
| Fix the Jev proxy | Set the proxy as a user-level environment variable and restart the IDE so routing reliably uses Jev |
| Try a smaller model | `qwen2.5:1.5b` roughly doubles generation speed |
