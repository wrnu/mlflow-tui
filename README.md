# mlflow-tui

A terminal UI for [MLflow](https://mlflow.org) tracking. Browse experiments, inspect runs, and watch live metrics without leaving the shell.

This is a **client** for the same tracking API as `mlflow ui`. It does not replace the MLflow server, and it is not an official MLflow or Databricks project.

## Why

The browser UI is the right tool for some things. It is not the right tool when you are already on a training box, in tmux, watching a run. `mlflow-tui` talks to a local file store or a remote tracking server and keeps the core loop on one screen: pick an experiment, scan runs, plot a metric, inspect params, tags, and artifacts.

## Features

- Experiments and runs in a single dashboard, with live refresh (default 3s)
- Braille metric charts: LogY, TensorBoard-style EMA smoothing, zoom and pan
- Params, tags, and artifacts for the selected run
- Mark runs to compare them, or delete a run
- Mouse and keyboard; `?` lists every key
- Built-in `--demo` workspace so you can try it without a tracking server

## Install

Python 3.10+.

```bash
uv tool install mlflow-tui
# or
pip install mlflow-tui
```

From a clone of this repo:

```bash
uv sync
uv run mlflow-tui --demo
```

## Quick start

```bash
mlflow-tui --demo
```

Against a real store or server:

```bash
mlflow-tui
mlflow-tui --tracking-uri file:./mlruns
mlflow-tui --tracking-uri http://127.0.0.1:5000
mlflow-tui --tracking-uri https://mlflow.example.com
mlflow-tui --experiment my-exp
```

If `MLFLOW_TRACKING_URI` is set, you can omit `--tracking-uri`. `--refresh 0` disables live refresh.

## Authentication

Same options as the official Python client:

- `--username` / `--password` (or `-p`) for HTTP basic auth
- If `--password` is omitted, `MLFLOW_TRACKING_PASSWORD` is used, otherwise you are prompted
- `https://user:password@host` in the tracking URI
- `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_TRACKING_TOKEN` for bearer auth (`--token`)

Basic auth takes precedence when a username is set. `~/.mlflow/credentials` is also honored.

`uv run -p` is uv’s `--python` flag, not this app’s password flag:

```bash
uv run mlflow-tui --tracking-uri https://mlflow.example --username warren --password 'secret'
```

## Status

v0.1 is the MVP: experiments → runs → metric plot → params/tags/artifacts, plus mark/compare and live refresh.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run ruff format src tests
uv run mlflow-tui --demo
```

## License

Apache License 2.0, the same license as [MLflow](https://github.com/mlflow/mlflow). See [LICENSE](LICENSE).

MLflow is a trademark of the Linux Foundation. This project is not affiliated with Databricks or the MLflow project.
