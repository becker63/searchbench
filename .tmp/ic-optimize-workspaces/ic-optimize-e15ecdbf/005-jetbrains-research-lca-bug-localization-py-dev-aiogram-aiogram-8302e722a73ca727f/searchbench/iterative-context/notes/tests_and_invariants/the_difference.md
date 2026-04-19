This is a great moment to crystallize things — you now have **three tools that look similar but do very different jobs**.

I’ll give you a **tight, side-by-side comparison**, then a **decision framework you can actually use while coding**.

---

# 🧠 The clean comparison (core idea)

| Type        | What it checks                    | Shape                        |
| ----------- | --------------------------------- | ---------------------------- |
| Snapshot    | “Did this exact behavior change?” | single run → saved output    |
| Property    | “Is this always valid?”           | single run → invariant       |
| Metamorphic | “Does behavior change correctly?” | multiple runs → relationship |

---

# 🔬 Same system, three perspectives

Let’s use your system:

```text
graph + events → replay → graph evolution
```

---

## 🟦 Snapshot testing

```text
input → output → compare to saved output
```

```python
steps = replay(...)
snapshot.assert_match(render_steps(steps))
```

### What it gives you

* locks behavior exactly
* shows diffs when behavior changes
* documents system evolution

---

## 🟩 Property testing

```text
input → output → check invariant
```

```python
assert no_dangling_edges(graph)
```

### What it gives you

* guarantees safety
* catches edge cases automatically
* protects core correctness

---

## 🟨 Metamorphic testing

```text
input A → output A
input B → output B
assert relation(output A, output B)
```

```python
assert normalize(r1) == normalize(r2)
```

### What it gives you

* validates behavior under change
* catches subtle logic bugs
* ensures consistency

---

# 🧠 What each one ADDS (this is the important part)

---

## 🟦 Snapshot adds: **regression visibility**

```text
“What changed?”
```

Without snapshots:

* you don’t notice behavioral drift
* debugging is harder

---

## 🟩 Property adds: **safety guarantees**

```text
“What must never break?”
```

Without properties:

* silent corruption possible
* edge cases slip through

---

## 🟨 Metamorphic adds: **behavioral correctness**

```text
“Does the system respond correctly to change?”
```

Without metamorphic:

* system looks “valid” but behaves wrong
* subtle bugs survive

---

# 🔥 What each one CANNOT do

This is key.

---

## ❌ Snapshot cannot:

* prove correctness
* catch invariants
* generalize beyond examples

---

## ❌ Property cannot:

* tell you what changed
* validate relative behavior
* ensure “good” outcomes

---

## ❌ Metamorphic cannot:

* detect invalid states (alone)
* define absolute correctness
* replace invariants

---

# 🧠 When to use each (practical rules)

---

## 🟦 Use SNAPSHOT when:

```text
you KNOW what the output should look like
```

Examples:

* event replay
* graph evolution steps
* deterministic scenarios

---

## 🟩 Use PROPERTY when:

```text
you KNOW something must ALWAYS be true
```

Examples:

* no dangling edges
* determinism
* monotonic growth

---

## 🟨 Use METAMORPHIC when:

```text
you KNOW how behavior SHOULD CHANGE
```

Examples:

* adding noise → no effect
* renaming nodes → same structure
* more evidence → better ranking

---

# 🧠 The decision tree (use this while coding)

---

### Step 1

```text
Do I know the exact expected output?
```

👉 YES → Snapshot

---

### Step 2

```text
Do I know a rule that must always hold?
```

👉 YES → Property

---

### Step 3

```text
Do I know how output should change if input changes?
```

👉 YES → Metamorphic

---

# 🔥 How they work together (this is the good part)

---

## Example: Your test today

```python
snapshot.assert_match(rendered_steps)
```

👉 Snapshot ensures:

```text
behavior didn’t change
```

---

## Add property:

```python
assert no_dangling_edges(graph)
```

👉 Property ensures:

```text
system isn’t broken
```

---

## Add metamorphic:

```python
assert same_behavior_with_noise(...)
```

👉 Metamorphic ensures:

```text
behavior is meaningful
```

---

## Combined result

```text
stable + safe + correct
```

---

# 🧠 Mental model (best one)

Think of your system as:

```text
a function evolving over time
```

---

## Snapshot tests

```text
sample points on the curve
```

---

## Property tests

```text
constraints the curve must obey
```

---

## Metamorphic tests

```text
how the curve changes when inputs shift
```

---

# 🔥 Where each dominates (important)

---

## Early system

```text
“I need to see what happens”
→ Snapshot heavy
```

---

## Stabilizing system

```text
“I need to ensure nothing breaks”
→ Property heavy
```

---

## Intelligent system (your future)

```text
“I need to ensure behavior is good”
→ Metamorphic heavy
```

---

# 🧾 Final distilled comparison

---

### Snapshot

```text
Locks behavior
Best for: known scenarios
Failure means: behavior changed
```

---

### Property

```text
Enforces invariants
Best for: correctness guarantees
Failure means: system is broken
```

---

### Metamorphic

```text
Validates behavior under change
Best for: heuristics / scoring
Failure means: logic is wrong
```

---

# 🚀 Final recommendation for YOU

Right now (Phase 1–2):

```text
Snapshots → primary
Properties → essential safety
Metamorphic → start introducing
```

---

Soon (Phase 3):

```text
All three equally important
```

---

If you want next, we can:

👉 turn one of your existing snapshot tests into a **combined snapshot + property + metamorphic test** (this is where it really clicks)
