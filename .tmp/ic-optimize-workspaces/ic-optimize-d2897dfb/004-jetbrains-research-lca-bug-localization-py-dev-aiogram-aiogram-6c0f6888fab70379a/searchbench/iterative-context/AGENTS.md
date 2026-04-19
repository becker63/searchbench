# AGENTS.md

## 🧠 Project Overview

This is a deterministic, event-driven graph traversal system.

Key properties:

* deterministic
* event-driven (RxPY)
* graph-based (NetworkX)
* strict models (Pydantic)
* snapshot tested

---

## ⚙️ Development Environment

This project uses **Nix flakes**.

### Enter dev shell:

nix develop

All tools are available inside the shell.

---

## 🚀 Commands (via uv)

All development commands are executed via uv.

Run commands like:

uv run <command>

---

### Core commands

Run tests:

uv run pytest

Run linting:

uv run ruff check .

Format code:

uv run black .

Type check:

uv run basedpyright

Run full check:

uv run basedpyright && uv run pytest

---

## 🧪 Testing

* Uses pytest
* Tests must be deterministic
* Prefer snapshot-style validation where possible

---

## 🧱 Architecture Rules

* Graph is monotonic (never delete)
* All mutations emit events
* No randomness or time-based logic
* Keep implementations simple and explicit

---

## 🚫 Constraints

* Do NOT introduce async
* Do NOT add external services
* Do NOT add unnecessary abstractions
* Do NOT modify flake.nix unless explicitly asked

---
## Hard Rules

- NEVER run pytest directly
- NEVER run pyright directly
- NEVER use pip
- NEVER use PYTHONPATH hacks
- NEVER assume system Python has dependencies

## 🧠 Notes for Agents

* Always run `uv run basedpyright && uv run ruff check && uv run pytest` before finalizing changes
* Assume the Nix environment is required
* Do not attempt to install dependencies manually
* Do not use pip or virtualenv

---

## ✅ TL;DR

nix develop
uv run basedpyright && uv run pytest
