Perfect — this is the *right abstraction to anchor everything*. I’ll rewrite your phase mapping using that mental model so it really sticks.

---

# 🧠 Core mental model (anchor this)

Think of your system as:

```text
a function evolving over time
```

---

### 🟦 Snapshot tests

```text
sample points on the curve
```

👉 “What does the system do here?”

---

### 🟩 Property tests

```text
constraints the curve must obey
```

👉 “What must NEVER happen anywhere?”

---

### 🟨 Metamorphic tests

```text
how the curve changes when inputs shift
```

👉 “Does the curve move correctly when I perturb it?”

---

# 🔥 Phase-by-phase (with this model)

---

# 🟢 Phase 0 — Graph + Events Exist

### System:

```text
the curve barely exists yet
```

---

## 🟩 Property (PRIMARY)

```text
define the shape constraints of the curve
```

You’re saying:

* curve must not break
* graph must always be valid

---

## 🟦 Snapshot (SECONDARY)

```text
sample a few early points
```

* “this input → this graph”

---

## 🟨 Metamorphic

```text
not meaningful yet
```

👉 No real “curve behavior” yet

---

# 🟢 Phase 1 — Deterministic Loop

### System:

```text
the curve is now visible over time
```

---

## 🟦 Snapshot (PRIMARY ⭐)

```text
capture points along the curve
```

* step 0 → step 1 → step 2

👉 You are literally recording the curve

---

## 🟩 Property

```text
ensure the curve is well-formed
```

* deterministic
* iteration doesn’t mutate

---

## 🟨 Metamorphic

```text
barely relevant
```

Maybe:

* small invariances

---

# 🟢 Phase 2 — Synthetic Expansion

### System:

```text
the curve is growing and branching
```

---

## 🟦 Snapshot (PRIMARY ⭐)

```text
document how the curve evolves
```

* growth over steps
* structure over time

---

## 🟩 Property (STRONG)

```text
ensure the curve never becomes invalid
```

* no dangling edges
* monotonic growth

---

## 🟨 Metamorphic (START)

```text
test how the curve reacts to small perturbations
```

Examples:

* add noise → curve unchanged
* iteration → no movement

---

👉 This is where you start asking:

```text
“does the curve behave consistently?”
```

---

# 🟡 Phase 3 — Scoring + Selection

### System:

```text
the curve is now being shaped intentionally
```

(decisions affect its path)

---

## 🟦 Snapshot

```text
sample important trajectories
```

* “this scoring → this path”

---

## 🟩 Property (CRITICAL)

```text
ensure the curve obeys decision laws
```

* max(score) is chosen
* no infinite loops

---

## 🟨 Metamorphic (PRIMARY ⭐)

```text
test how the curve shifts under input changes
```

Examples:

* add irrelevant node → curve unchanged
* increase tokens → curve shifts toward node

---

👉 Now you care about:

```text
is the curve moving correctly?
```

---

# 🟡 Phase 4 — Snapshot as Spec

### System:

```text
the curve is now considered “correct”
```

---

## 🟦 Snapshot (PRIMARY ⭐⭐⭐)

```text
freeze important sections of the curve
```

👉 This is your **spec**

---

## 🟩 Property

```text
ensure constraints still hold everywhere
```

---

## 🟨 Metamorphic

```text
validate key invariances of the curve
```

But only for:

* core behaviors

---

# 🟡 Phase 5 — Partial Data Handling

### System:

```text
the curve is now noisy / incomplete
```

---

## 🟩 Property (PRIMARY ⭐)

```text
ensure the curve never breaks under noise
```

---

## 🟨 Metamorphic (STRONG)

```text
ensure the curve degrades gracefully
```

Examples:

* missing data → small distortion
* partial graph → still valid

---

## 🟦 Snapshot

```text
less useful (curve too variable)
```

---

# 🔴 Phase 6 — Real Adapter

### System:

```text
the curve is driven by messy real-world input
```

---

## 🟩 Property (PRIMARY)

```text
ensure the curve stays valid under chaos
```

---

## 🟨 Metamorphic (PRIMARY)

```text
ensure the curve is stable under noise
```

Examples:

* reorder input → same curve
* partial data → similar curve

---

## 🟦 Snapshot

```text
only for curated cases
```

---

# 🔴 Phase 7 — Refinement Loop

### System:

```text
you are actively improving the curve
```

---

## 🟨 Metamorphic (PRIMARY ⭐⭐⭐)

```text
compare how the curve shifts under improvements
```

Examples:

* better signal → better trajectory
* noise → minimal deviation
* small input change → smooth curve change

---

## 🟩 Property

```text
ensure constraints still hold
```

---

## 🟦 Snapshot

```text
only for benchmarks
```

---

# 🧠 The full map (reframed)

| Phase | What the curve is | Snapshot            | Property               | Metamorphic           |
| ----- | ----------------- | ------------------- | ---------------------- | --------------------- |
| 0     | barely exists     | sample points       | ⭐⭐⭐ define constraints | ❌                     |
| 1     | visible over time | ⭐⭐⭐ trace curve     | ⭐⭐ validate shape      | ❌                     |
| 2     | growing           | ⭐⭐⭐ document growth | ⭐⭐ enforce validity    | ⭐ start               |
| 3     | decision-driven   | ⭐ sample paths      | ⭐⭐⭐ enforce rules      | ⭐⭐⭐ shape correctness |
| 4     | stable/spec       | ⭐⭐⭐ freeze curve    | ⭐⭐ enforce             | ⭐⭐ invariances        |
| 5     | noisy             | ⭐ limited           | ⭐⭐⭐ robustness         | ⭐⭐⭐ graceful change   |
| 6     | real-world        | ⚠️ minimal          | ⭐⭐⭐ safety             | ⭐⭐⭐ stability         |
| 7     | optimized         | ⭐ benchmarks        | ⭐⭐ enforce             | ⭐⭐⭐ improvement       |

---

# 🔥 The intuition (cleanest version)

---

## 🟦 Snapshot

```text
“What does the curve look like here?”
```

---

## 🟩 Property

```text
“What shape must the curve always have?”
```

---

## 🟨 Metamorphic

```text
“How should the curve move when I push it?”
```

---

# 🧾 Final rule (now with the model)

When writing a test, ask:

---

### 1. Am I sampling the curve?

👉 Snapshot

---

### 2. Am I constraining the curve?

👉 Property

---

### 3. Am I perturbing the curve?

👉 Metamorphic

---

# 🚀 Your current position

You are here:

```text
curve is visible and growing (Phase 1–2)
```

So:

```text
Snapshots → map the curve
Properties → keep it valid
Metamorphic → start probing it
```
