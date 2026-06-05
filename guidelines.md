# Guidelines for Code Review

This document outlines the guidelines for code review within our team. It aims to ensure high-quality code, maintainability, and adherence to best practices.

## Documentation

- The agent cannot remove any comments in the code; it can only add or modify them to complete/enrich the comment.
- Docstrings must be kept up-to-date after each modification (created if missing, updated otherwise).
- Each time an agent modifies the app code, the modification must be documented directly in the code.
- In case of multiple modifications, a summary of the changes must be provided in the comments.
- If the Python package of a modified script contains a `README.md`, its markdown file must be updated to reflect the script modification.



## Project

- The `src` folder is deprecated and must not be used to add any new project content. All new and refactored code must be placed in the `app` or `tests` directories.

## Testing

- When testing with a real model, the Hugging Face model `MODEL_NAME=dulimov/Qwen3-4B-rk3588-1.2.1-unsloth-16k` and `MODEL_FILE=Qwen3-4B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm` must be used.
- To run `pytest` and `pytest-bdd`, `uv` must be used (e.g. `uv run pytest`).
- Blackbox tests must be done using HTTP requests using `httpx.Client` or the FastAPI TestClient framework.

## Pre-commit Checks

- Pre-commit checks must be run after any code modification performed by the agent (e.g., using `uv run pre-commit run --all-files`).

## Git Commit

- At the end of an agent task, if the code has been modified, a Git commit must be performed.
- The commit message must be correct and respect the Git Conventional Commits specification (e.g., `feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`, `test: ...`, `chore: ...`).