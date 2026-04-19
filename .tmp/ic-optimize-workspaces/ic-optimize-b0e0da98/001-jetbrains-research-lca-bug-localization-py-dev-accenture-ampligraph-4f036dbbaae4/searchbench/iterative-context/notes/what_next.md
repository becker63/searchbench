Perfect — that actually tightens this up a lot. Now you’re not just “pushing JSON”, you’re defining a **real contract** and getting **typed clients for free**.

Let’s reframe the plan with **OpenAPI as the source of truth**.

---

# 🧭 What you’re building (updated)

> **A typed TestRun ingestion + replay system, defined via OpenAPI**

---

# 🧱 Core Architecture (with OpenAPI)

```text
Python (Pydantic models)
        ↓
FastAPI → OpenAPI schema
        ↓
Generated TS client (Next.js)
        ↓
UI replay via RxJS
```

---

# 🔥 Phase 2.5 Plan (OpenAPI-first)

## 1. Define models (SOURCE OF TRUTH)

### Python (Pydantic)

```python
from pydantic import BaseModel
from typing import Literal, Union, List
from datetime import datetime

class AddNodes(BaseModel):
    type: Literal["addNodes"]
    step: int
    payload: dict

class AddEdges(BaseModel):
    type: Literal["addEdges"]
    step: int
    payload: dict

GraphEvent = Union[AddNodes, AddEdges]


class TestRun(BaseModel):
    name: str
    timestamp: datetime
    phase: str
    events: List[GraphEvent]
```

---

👉 This drives:

* validation
* OpenAPI schema
* client generation

---

## 2. Build FastAPI server (protocol layer)

This is your **contract server**, not your solver.

```python
from fastapi import FastAPI

app = FastAPI()

runs: dict[str, TestRun] = {}
```

---

### POST `/runs`

```python
@app.post("/runs")
def create_run(run: TestRun):
    runs[run.name] = run
    return {"ok": True}
```

---

### GET `/runs`

```python
@app.get("/runs")
def list_runs():
    return list(runs.keys())
```

---

### GET `/runs/{name}`

```python
@app.get("/runs/{name}")
def get_run(name: str):
    return runs[name]
```

---

👉 Now you get:

```
/docs → OpenAPI → typed clients
```

---

## 3. Generate client for Next.js

Use something like:

```bash
openapi-typescript http://localhost:8000/openapi.json -o types.ts
```

or:

```bash
orval
openapi-generator
```

---

👉 Now your frontend has:

```ts
type TestRun
type GraphEvent
```

---

# 🔧 4. Python publisher (now trivial + typed)

```python
import httpx

def publish_run(run: TestRun):
    httpx.post(
        "http://localhost:8000/runs",
        json=run.model_dump(mode="json"),
    )
```

---

👉 Fully validated on both ends

---

# 🎛️ 5. Next.js UI integration

Now instead of guessing types:

```ts
import { TestRun } from "@/generated/types"
```

---

### Fetch runs

```ts
const runs = await fetch("/runs").then(res => res.json())
```

---

### Fetch selected run

```ts
const run: TestRun = await fetch(`/runs/${name}`).then(res => res.json())
```

---

### Replay

```ts
from(run.events).subscribe(eventBus)
```

---

# 🧪 6. Testing flow (unchanged, but cleaner)

```python
def test_node_variants():
    events = run_graph(...)

    run = TestRun(
        name="test_node_variants",
        timestamp=datetime.utcnow(),
        phase="phase_2",
        events=events,
    )

    publish_run(run)
```

---

# 🧠 What OpenAPI just unlocked

## ✅ Shared schema (no drift)

Python ↔ TS guaranteed aligned

---

## ✅ Typed UI

No more:

```ts
event.type === "something"
```

Guessing — now it’s enforced

---

## ✅ Extensibility

Later you can add:

```python
class ScoreEvent(BaseModel):
    type: Literal["score"]
    score: float
```

👉 UI gets it automatically

---

## ✅ Tooling

* Swagger UI
* request validation
* client generation
* contract testing

---

# ⚠️ Keep this boundary clean

## IMPORTANT:

FastAPI server = **protocol layer only**

NOT:

* ❌ graph logic
* ❌ traversal
* ❌ scoring

---

👉 It should only:

```text
validate → store → serve
```

---

# 🚀 Clean execution order

## Step-by-step

1. ✅ Define `GraphEvent` + `TestRun` (Pydantic)
2. ✅ Spin up FastAPI server
3. ✅ Verify `/docs` looks correct
4. ✅ Generate TS client
5. ✅ Implement `publish_run`
6. ✅ Hook UI dropdown → `/runs`
7. ✅ Replay events

---

# 🔜 After this (important)

Once this is working, you unlock:

## Phase 3 (Scoring)

Now you can:

* run multiple scoring strategies
* compare runs visually

---

## Phase 7 (your big goal)

```text
jCodeMunch vs your system
```

becomes:

```text
Run A vs Run B
```

---

# 🧠 Final framing

You are now building:

> **a typed execution trace protocol**

Not just:

> “a graph thing”

---

And that’s why this will scale cleanly into:

* experimentation
* benchmarking
* explainability

---

If you want next, I can:

* generate a **strict discriminated union for GraphEvent (best practice)**
* or help you structure **OpenAPI → TS client generation cleanly in your Next.js repo**
* or design a **run diff UI (this is where things get 🔥)**

But this plan is solid and ready to execute.
