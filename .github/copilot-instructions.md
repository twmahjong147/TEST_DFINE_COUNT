<!-- Copilot instructions for working in the DFINE count repo -->
# DFINE Count — Copilot instructions

Purpose: help an AI coding assistant be productive quickly in this repository.

- Quick start: run the CLI to count objects:
  - Example: `python -m dfine_count count --image samples/sample.jpg --device cpu`
  - CI-friendly test runs force CPU (see `tests/test_cli.py`).

- Where to look for core logic:
  - CLI entry: `dfine_count/cli.py` — parse flags and prints a JSON result.
  - Inference and model integration: `dfine_count/infer.py` — loads model weights, adapts `D-FINE-master` via runtime `sys.path` insertion, and returns `(labels, boxes, scores)`.
  - Defaults: `dfine_count/_config.py` defines `DEFAULT_WEIGHTS` and `DEFAULT_CONFIG` (paths are often absolute and environment-specific).

- Important integration details (do not change lightly):
  - The package expects a sibling directory `D-FINE-master` containing D-FINE source under `src/` (the code imports `src.core.YAMLConfig`). `infer._add_dfine_path()` prepends that folder to `sys.path`.
  - `load_model()` expects checkpoint dicts that may contain `ema` or `model` keys; it builds a wrapper `Model` that calls `.deploy()` on both `cfg.model` and `cfg.postprocessor`.
  - The CLI prints a JSON object with `total_count` and optional `per_class` and `visualization` fields — tests and downstream tooling rely on this stable JSON shape.

- Editing & debugging guidance:
  - To change defaults, update `dfine_count/_config.py` or prefer exposing values via CLI flags (`--weights`, `--config`).
  - For reproducing CI-local behavior, run device as `cpu` since tests use CPU-only runs (`--device cpu`).
  - Visualization files are written under the repository `outputs/` directory by `_save_visualization()`.

- Tests and verification:
  - Unit smoke test: `pytest -q tests/test_cli.py` (runs the CLI as a module and expects JSON output).
  - When adding features that change CLI output, update `tests/test_cli.py` to assert the exact JSON schema.

- Repo-specific automation & agent tooling:
  - There is a helper script `.specify/scripts/bash/update-agent-context.sh` that looks for agent files. That script expects a Copilot file at `.github/agents/copilot-instructions.md` (if you use the generator, mirror this file there).

- When making changes that touch D-FINE integration:
  - Document required D-FINE `config.yml` fields (look at `D-FINE-master/configs/*`) because `YAMLConfig(config_path, resume=weights)` is used to construct the model.
  - Preserve the `.deploy()` usage and the tuple output shape `(labels, boxes, scores)` — consumers (including tests) assume this contract.

- Useful references (open these first):
  - `dfine_count/cli.py`
  - `dfine_count/infer.py`
  - `dfine_count/_config.py`
  - `tests/test_cli.py`

If anything above is unclear or you want this shortened/expanded for a specific agent persona, tell me which area to adjust.
