# mlflow-tui

A terminal UI for [MLflow](https://mlflow.org) tracking.

MLflow already exposes experiments, runs, metrics, params, tags, and artifacts through its tracking API. This is another client for that API — the same workflow as `mlflow ui`, without leaving the terminal.

```
mlflow-tui
├── Experiments
│   ├── LT-JEPA
│   ├── transformer-baseline
│   └── ablations
│
├── Runs
│   NAME          STATUS     LOSS      GPU
│   run-0042      ● running  0.0381    91%
│   run-0041      ✓ done     0.0402    -
│   run-0040      ✕ failed   0.0891    -
│
└── Selected Run
    loss
    0.10 ┤╮
    0.08 ┤╰╮
    0.06 ┤ ╰─╮
    0.04 ┤   ╰───────

    lr        3e-4
    epoch     17 / 50
    samples   42.1M
```

## Why

The browser UI is the right tool for some things. It is not the right tool when you are already on a training box, in tmux, watching a run. `mlflow-tui` talks to a local file store or a remote tracking server and keeps the core loop in one screen: pick an experiment, scan runs, plot a metric, inspect params/tags/artifacts.

No MLflow server changes. The TUI is a client:

![mlflow-tui talks to local file stores or a tracking server through MlflowClient](docs/architecture.png)

```mermaid
flowchart LR
  subgraph Backends
    File["Local file store"]
    Server["Tracking server"]
  end

  TUI["mlflow-tui"]
  API["MlflowClient / REST"]

  File --> API
  Server --> API
  API --> TUI
```

## Install

```bash
uv tool install mlflow-tui
# or
pip install mlflow-tui
```

From this repo:

```bash
uv sync
uv run mlflow-tui --demo
```

## Usage

```bash
mlflow-tui
mlflow-tui --tracking-uri http://localhost:5000
mlflow-tui --tracking-uri https://mlflow.mycompany.ca
mlflow-tui --tracking-uri https://mlflow.mycompany.ca --username warren --password 'secret'
mlflow-tui --tracking-uri https://mlflow.mycompany.ca --username warren -p 'secret'
mlflow-tui --tracking-uri https://user:password@mlflow.mycompany.ca
mlflow-tui --tracking-uri file:./mlruns
mlflow-tui --demo
```

Authentication uses the same mechanism as the official Python client:

- `--username` / `--password` (or `-p`) pass basic auth on the command line
- If `--password` is omitted, `MLFLOW_TRACKING_PASSWORD` is used, otherwise you are prompted
- `https://user:password@host` in the tracking URI
- `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_TRACKING_TOKEN` for bearer auth (`--token`)

Basic auth takes precedence when a username is set. `~/.mlflow/credentials` is also honored.

`uv run -p` is uv’s `--python` flag, not this app’s password flag. Use:

```bash
uv run mlflow-tui --tracking-uri https://mlflow.example --username warren --password 'secret'
```

Mouse and touch are on by default (`--no-mouse` to force them off). The UI hides the hardware cursor and repaints the full screen so incomplete PTYs cannot dump cursor-addressing codes into the header or desync the runs list. On hosts that swallow taps (for example a phone terminal pane), use the keyboard: arrows, `/`, `m`, `f`.

### 403 Forbidden

A 403 after you have already passed a username and password is often **not** a bad password:

1. **macOS port 5000** — `localhost:5000` can resolve to IPv6 and hit Control Center AirPlay instead of MLflow. `mlflow-tui` rewrites `localhost` to `127.0.0.1`. You can also pass `--tracking-uri http://127.0.0.1:5000`.
2. **MLflow 3 Host check** — the FastAPI server rejects unknown `Host` headers (`Invalid Host header - possible DNS rebinding attack detected`). `--allowed-hosts localhost,127.0.0.1` does **not** match `Host: 127.0.0.1:5000`. Prefer `--allowed-hosts localhost,127.0.0.1,localhost:*,127.0.0.1:*` (or `*`).
3. **HTTP→HTTPS redirect** — proxies that bounce `http://host:5000` to `https://host` strip the `Authorization` header. Use the final `https://` tracking URI.
4. **ACL** — basic auth succeeded but the user cannot list experiments (`Permission denied`).

### Keys

The footer keeps the everyday keys: `/` filter, `m` metric, `f` focus graph, `?` all bindings, `q` quit. Everything else still works; `?` lists them.

| Key | Action |
| --- | --- |
| `tab` | Cycle panes |
| `/` | Filter experiments and runs |
| `r` | Refresh |
| `m` | Next plotted metric |
| `n` | Previous plotted metric |
| `l` | Toggle log Y (symlog when the series includes 0 or negatives) |
| `s` | Toggle EMA smoothing (raw stays dim underneath) |
| `space` | Mark a run |
| `c` | Compare marked runs |
| `y` | Copy the selected run ID |
| `f` | Focus the graph (zoom and pan) |
| `?` | All key bindings |
| `esc` | Leave graph focus |
| `q` | Quit |

Graph keys work after `f` (they are not in the footer): `=`/`-` zoom both axes, `[`/`]` zoom X, `i`/`o` zoom Y, arrows pan, `0` resets. Mouse wheel zooms; drag pans.

### Mouse

| Action | Result |
| --- | --- |
| Click an experiment or run | Select it |
| Ctrl-click or double-click a run | Mark / unmark for compare |
| Click the plot or run header | Cycle the plotted metric |
| Double-click the plot | Focus / unfocus the graph |
| Wheel on the plot | Zoom in / out |
| Drag on the plot | Pan |
| Click footer keys | Same as the keybinding |
| Click outside the compare dialog | Close it |

Filter accepts a substring, or an MLflow search expression such as `metrics.loss < 0.05`.

Live refresh defaults to every 3 seconds (`--refresh 0` disables it). Running runs keep their metric history moving.

## Status

v0.1 is the MVP: experiments → runs → metric plot → params/tags/artifacts, plus mark/compare and live refresh.

Next: richer run comparison, artifact previews, and tighter streaming for long metric histories.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run ruff format src tests
uv run mlflow-tui --demo
```
