This is a really good question — and the answer is subtle.

---

# 🧠 Short answer

```text
Property testing and metamorphic testing are peers, not layers.
```

* ❌ Metamorphic is NOT “more important”
* ❌ Metamorphic is NOT “enabled by” property testing
* ✅ They solve **different correctness gaps**

---

# 🔥 The clean distinction

## Property testing

```text
Does each output satisfy fundamental laws?
```

## Metamorphic testing

```text
Does behavior change correctly when inputs change?
```

---

# 🧠 Think of it like axes

```
            INPUT VARIATION
                 ↑
                 |
     Metamorphic |   (relations across runs)
                 |
-----------------+----------------→ OUTPUT VALIDITY
                 |
     Property    |   (invariants within one run)
                 |
```

---

# 🧱 What each one protects (in your system)

## Property tests protect:

```text
“Nothing illegal ever happens”
```

Examples:

* no dangling edges
* iteration does not mutate
* determinism holds

👉 These are **safety guarantees**

---

## Metamorphic tests protect:

```text
“The system behaves correctly under change”
```

Examples:

* adding noise doesn’t affect result
* renaming nodes doesn’t change structure
* more evidence improves ranking

👉 These are **behavioral guarantees**

---

# 🔥 Why you need BOTH

Because each one misses things the other catches.

---

## ❌ Property tests miss this

Bug:

```text
scoring depends on total node count
```

Add irrelevant node → ranking changes

---

### Property test:

```text
✔ graph valid
✔ no invariants violated
→ passes
```

---

### Metamorphic test:

```text
❌ result changed when it shouldn’t
→ fails
```

---

## ❌ Metamorphic tests miss this

Bug:

```text
edge created with missing node
```

---

### Metamorphic test:

```text
might still satisfy relation
→ passes
```

---

### Property test:

```text
❌ dangling edge detected
→ fails
```

---

# 🧠 So which is “more important”?

## Phase 1–2 (your current stage)

👉 **Property testing is slightly more important**

Because you’re establishing:

```text
system correctness + safety
```

---

## Phase 3+ (scoring, heuristics)

👉 **Metamorphic becomes equally or more important**

Because now you care about:

```text
quality + behavior under change
```

---

# 🔥 The real progression (this is key)

## Early system (you now)

```text
Goal: “system doesn’t break”
→ property tests dominate
```

---

## Mid system (Phase 3)

```text
Goal: “system behaves well”
→ metamorphic tests rise
```

---

## Mature system

```text
Goal: “system improves over time”
→ differential + metamorphic dominate
```

---

# 🧠 Are metamorphic tests “enabled” by property tests?

No — but they often **use the same machinery**.

---

## Shared infrastructure

Both use:

* Hypothesis (often)
* random input generation
* replayable systems

---

## But conceptually different

### Property test:

```python
@given(x)
assert invariant(f(x))
```

---

### Metamorphic test:

```python
@given(x)
x2 = transform(x)

assert relation(f(x), f(x2))
```

---

👉 Same tool (Hypothesis), different idea.

---

# 🔥 Important insight (this is the good part)

Property tests are:

```text
absolute correctness
```

Metamorphic tests are:

```text
relative correctness
```

---

# 🧠 Why this matters for YOU

Your system is:

```text
deterministic ✔
structured ✔
but heuristic (soon) ✔
```

---

That means:

* property tests → protect the engine
* metamorphic tests → validate behavior

---

# ⚡ A very concrete way to think about it

## Property test says:

```text
“This output is valid”
```

## Metamorphic test says:

```text
“This output is consistent with another output”
```

---

# 🧾 Final answer (clean)

* Metamorphic testing is **not more important** than property testing
* It is **not enabled by** property testing
* They are **complementary and orthogonal**

---

## In your project:

```text
Property tests → guarantee system safety
Metamorphic tests → guarantee behavioral correctness
```

---

# 🚀 My recommendation for you

Right now:

```text
70% property tests
30% metamorphic tests
```

Then as you hit scoring:

```text
50% property
50% metamorphic
```

---

If you want next, we can:

👉 design a **single test that combines both** (very powerful pattern)
👉 or implement your first metamorphic test using your existing DSL + snapshot setup
