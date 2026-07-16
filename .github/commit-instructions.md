# Commit message instructions (GitHub Copilot / VS Code)

Use these rules when generating commit messages for this repo.

## Format

- Prefer Conventional Commits: `type(scope): short summary`
- Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`, `ci`
- Subject: imperative mood (“Add”, not “Added”), ~50–72 chars, no trailing period
- Body (when needed): why + what, short bullets; mention user-facing surfaces (WASM, `/api/`, marimo)

## Project-specific

- Public book lives in `marimo_notebooks/` + `docs/` (WASM + static API)
- Do not invent unrelated changes; summarize only staged diffs
- Call out adaptive multitaper / tSNR / QC / static FastAPI mirror when relevant
- Avoid dumping file lists or secrets

## Example

```
feat(api): ship static FastAPI mirror on GitHub Pages

Freeze feature/QC endpoints as docs/api/*.json with Swagger explorer.
Wire adaptive multitaper QC into WASM chapters and cleaned feature store.
```
