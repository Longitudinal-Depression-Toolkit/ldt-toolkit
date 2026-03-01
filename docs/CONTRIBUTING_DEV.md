# Contributing to LDT-Toolkit

Contributions are welcome across both toolkit components.

## Contribute to the API Toolkit (Python)

1. Open an issue first to discuss your proposal and avoid duplicated work.
2. Fork the repository and create a feature branch.
3. Add tests and update docs for your changes.

Build from source:

```bash
git clone https://github.com/Longitudinal-Depression-Toolkit/ldt-toolkit.git
cd ldt-toolkit
uv sync
```

## Contribute to the CLI Toolkit (Go)

The CLI lives in its own repository:

```bash
git clone https://github.com/Longitudinal-Depression-Toolkit/CLI.git
cd CLI
make build
make install-bash # or make install-fish
```

Use the CLI `Makefile` targets to build and test local changes.

## Open an Issue and Discuss with Us

Before implementing a major change, open a GitHub issue in the relevant repository:

- API toolkit/issues: <https://github.com/Longitudinal-Depression-Toolkit/ldt-toolkit/issues>
- CLI/issues: <https://github.com/Longitudinal-Depression-Toolkit/CLI/issues>

## Compile Documentation (in `ldt-toolkit`)

```bash
cd ldt-toolkit
uv sync
uv run mkdocs serve
```

To run a production-like build:

```bash
uv run mkdocs build --strict
```
