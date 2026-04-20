## Loop visualization

Use the built-in python-statemachine rendering to inspect the optimization and repair loops.

Examples:

- Mermaid to stdout (both machines):
  ```bash
  python tools/loop_viz.py --format mermaid
  ```
- DOT to file:
  ```bash
  python tools/loop_viz.py --format dot --output /tmp/loop.dot
  ```
- PNG (Graphviz required):
  ```bash
  python tools/loop_viz.py --format png --machine repair --output /tmp/repair.png
  ```

By default both machines are rendered; use `--machine optimization` or `--machine repair` to narrow output.
