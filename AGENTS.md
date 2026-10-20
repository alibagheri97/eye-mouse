# Project instructions

Use the installed Open-GSD workflow and keep `.planning/` state current. Preserve existing user changes.

## Codebase knowledge graph

Prefer codebase-memory-mcp for discovery: `search_graph`, `trace_path`, `get_code_snippet`,
`query_graph`, `get_architecture`. Index this repository first if it is not indexed.
Use text search for literal strings/configuration/non-code files or insufficient graph results.

## Laboratory contract

- Plain experimental UI; visual redesign is deferred until requested.
- Local MediaPipe inference; never record camera images/video as part of normal acquisition.
- Keep validation data out of fitting and keep repeated calibration targets grouped in CV.
- Do not claim human accuracy from synthetic tests or camera/GUI smoke checks.
- Label monocular distance and angular conversions as approximate with their assumptions.
- Mouse movement requires explicit arming, a working global emergency stop and current valid tracking.
- Run `uv run pytest -q` and `uv run ruff check eye_mouse tests tools` for relevant changes.
- Runtime tools that open the camera or move the pointer are explicit manual checks under `tools/`.
