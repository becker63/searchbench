Perfect — let’s make this **actionable and high-signal**.

I’ll give you a **curated set of metamorphic tests** tailored to your system, grouped by what they protect. These are not random — they map directly to:

```text
data layer → control flow → scoring (future)
```

---

# 🧠 Quick structure

Each test =

```text
Transformation (T)
Expected Relation (R)
Why it matters
```

---

# 🔹 GROUP 1 — Graph invariance (Phase 1/2 MUST-HAVE)

These protect your **core semantics**.

---

## 1. Irrelevant node invariance ⭐ (start here)

### T

Add a node with no edges

### R

Output graph should be identical

```python
def test_irrelevant_node_invariance():
    g1 = base_graph()
    g2 = base_graph()

    apply_events(g2, [{"type": "addNodes", "nodes": ["noise"]}])

    r1 = run_traversal(g1)
    r2 = run_traversal(g2)

    assert normalize(r1) == normalize(r2)
```

### Why

Prevents hidden global coupling / noise sensitivity

---

## 2. Disconnected subgraph invariance

### T

Add an entirely separate subgraph

### R

Traversal result should ignore it

### Why

Ensures locality — traversal only cares about reachable context

---

## 3. Node renaming invariance

### T

Rename all node IDs consistently

### R

Graphs should be isomorphic

```python
assert isomorphic(r1, r2)
```

### Why

Prevents ID-based bias (VERY common subtle bug)

---

## 4. Event order invariance (commutative ops)

### T

Reorder independent events

### R

Final graph must be identical

### Why

Ensures deterministic + order-independent semantics

---

# 🔹 GROUP 2 — Event semantics (Phase 1/2)

These validate your **event DSL correctness**.

---

## 5. Duplicate addNodes is a no-op

### T

Apply addNodes twice

### R

Graph unchanged after first application

### Why

Locks idempotency

---

## 6. Iteration event invariance ⭐

### T

Insert/remove iteration events

### R

Graph must be identical

```python
assert normalize(g_with) == normalize(g_without)
```

### Why

Ensures iteration is purely observational

---

## 7. Equivalent update merging

### T

Split update into multiple patches vs single patch

```text
update(state=resolved)
update(tokens=3)
```

vs

```text
update({state=resolved, tokens=3})
```

### R

Final graph identical

### Why

Ensures update semantics are composable

---

# 🔹 GROUP 3 — Structural symmetry

These test **fairness and consistency**

---

## 8. Duplicate subgraph symmetry

### T

Duplicate part of graph

### R

Both copies behave the same

### Why

Prevents structural bias

---

## 9. Edge direction consistency (if applicable)

### T

Flip edge direction (if semantics allow)

### R

Behavior should change *predictably*, not randomly

### Why

Catches misuse of directionality

---

# 🔹 GROUP 4 — Robustness (LLM / adversarial alignment)

Even though LLM isn’t core, this is VERY relevant.

---

## 10. Noise robustness ⭐

### T

Inject random valid-but-irrelevant events

### R

Graph remains valid + stable

```python
assert graph_is_valid(result)
```

### Why

System must tolerate garbage suggestions

---

## 11. Invalid suggestion rejection

### T

Inject invalid edge

### R

Either:

* clean failure OR
* no corruption

### Why

Locks your validation boundary

---

# 🔹 GROUP 5 — Growth / monotonicity

These protect your **core graph guarantees**

---

## 12. Monotonic node growth

### T

Add events

### R

Nodes never disappear

---

## 13. Monotonic edge growth

### T

Add edges over time

### R

Edges only increase (unless explicitly changed later)

---

# 🔹 GROUP 6 — Phase 3 (SCORING — future but important)

These will become your *most valuable tests*

---

## 14. Irrelevant node does not affect ranking ⭐

### T

Add irrelevant node

### R

Ranking order unchanged

---

## 15. More evidence improves ranking

### T

Increase tokens / evidence

### R

Node rank should not decrease

```python
assert rank_after >= rank_before
```

---

## 16. Symmetric nodes rank equally

### T

Create two identical nodes

### R

Scores equal

---

## 17. Small perturbation → small change

### T

Slightly modify graph

### R

Ranking changes minimally

### Why

Prevents unstable scoring

---

## 18. Adding connectivity increases importance

### T

Increase degree of node

### R

Score should increase (given your heuristic)

---

# 🔹 GROUP 7 — Determinism (critical)

---

## 19. Transformation preserves determinism ⭐

### T

Apply transformation (rename, reorder, etc.)

### R

Still deterministic

```python
f(x) == f(x)
f(T(x)) == f(T(x))
```

---

# 🧠 How many should you actually implement?

Don’t do all at once.

---

## Start with these 5:

```text
1. irrelevant node invariance
2. iteration invariance
3. event order invariance
4. duplicate addNodes idempotency
5. noise robustness
```

👉 This already gives you HUGE coverage

---

## Then add:

```text
6. node renaming invariance
7. disconnected subgraph invariance
```

---

## Then (Phase 3):

```text
8. ranking invariance under noise
9. evidence improves ranking
```

---

# 🔥 The deeper structure (important insight)

You’re covering:

| Layer      | What you're testing   |
| ---------- | --------------------- |
| Data       | graph invariance      |
| Events     | DSL correctness       |
| Control    | determinism, ordering |
| Robustness | adversarial input     |
| Scoring    | behavior under signal |

---

# 🧾 Final takeaway

A good metamorphic suite for your system ensures:

```text
- irrelevant changes don’t affect behavior
- equivalent inputs produce equivalent outputs
- meaningful changes produce meaningful effects
- garbage input doesn’t break the system
```

---

# 🚀 If you want next

We can:

👉 define a reusable **metamorphic test harness** (so adding these is trivial)
👉 or implement the **first 3 tests directly in your codebase style** (very fast win)
