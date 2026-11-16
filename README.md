# convex-opt-mcp

An [MCP](https://modelcontextprotocol.io) server that lets LLM clients (Claude Desktop, Claude Code, Cursor, …) formulate and solve **convex optimization problems** with [CVXPY](https://www.cvxpy.org): LPs, QPs, SOCPs, SDPs, and mixed-integer problems when a MIP solver is installed.

> **Unofficial.** This project is built on CVXPY but is not affiliated with or endorsed by the CVXPY project.

## Tools

| Tool | What it does |
|---|---|
| `solve_problem` | Builds and solves a problem; returns status, optimal value, variable values, constraint duals and solver diagnostics |
| `check_problem` | Builds a problem without solving; reports DCP compliance, curvature, size, and which constraints break DCP rules |
| `list_solvers` | Lists the CVXPY solvers installed on the machine |
| `set_parameter` | Stores a numeric array under a name so later problems can refer to it without resending it |
| `load_parameter` | Loads an array from a `.npy`, `.npz`, `.csv`, `.tsv`, `.txt` or `.json` file and stores it under a name |
| `list_parameters` | Lists stored parameters: shape, value range and, for symmetric matrices, the smallest eigenvalue |
| `clear_parameters` | Deletes some or all stored parameters |

Every tool declares an output schema, so clients get structured, typed results.

### Works beyond Claude

Tool schemas are served in a portable subset of JSON Schema: nested definitions are inlined (no `$ref`/`$defs`), optional arguments have no `null` branches, and numeric data is typed as a number, a vector, or a matrix. This avoids the most common incompatibilities with other clients and model APIs (e.g. OpenAI strict function calling, Gemini function declarations). The tool descriptions include worked examples (least squares, LP, mean-variance QP, SDP, integer program) and a list of common mistakes, which matters most for smaller models.

A problem is described as JSON: variables (with shape and attributes such as `nonneg`, `PSD`, `integer`), numeric parameters, and the objective and constraints as CVXPY expression strings:

```json
{
  "variables":   [{"name": "w", "shape": [5]}],
  "sense":       "maximize",
  "objective":   "mu @ w - (gamma / 2) * cp.quad_form(w, C)",
  "constraints": ["cp.sum(w) == 1"],
  "parameters":  {"mu": [0.01, -0.03, -0.02, -0.12, 0.09], "C": [[...]], "gamma": 1}
}
```

Expressions may use `cp.*`, `np.*`, the declared variables, inline parameters and stored parameters (inline values win if a name is both). They are parsed and checked against an allow-list of syntax before evaluation (no lambdas, comprehensions, imports, or `_`-prefixed names). This is defense in depth for a locally run server, **not** a hardened sandbox — don't expose it as a public service.

### Reusing data

Large or repeated inputs don't need to travel through the conversation on every call:

```text
load_parameter(name="C",  path="~/data/cov.npy")
load_parameter(name="mu", path="~/data/returns.csv", skip_header=1)
solve_problem({... "objective": "mu @ w - (gamma / 2) * cp.quad_form(w, C)",
                   "parameters": {"gamma": 5} ...})
```

**Or skip the store and reference files directly.** Any entry in `parameters` can be a file reference instead of a value. The file is read on every call, so the problem doesn't depend on server state and survives server restarts:

```json
"parameters": {
  "C":     {"file": "~/data/cov.npy"},
  "mu":    {"file": "~/data/returns.csv", "skip_header": 1},
  "both":  {"file": "~/data/model.npz", "key": "C"},
  "gamma": 5
}
```

File references accept the same formats and options as `load_parameter` (`key`, `delimiter`, `skip_header`) and the same allowed-folder restriction. Results list them in `parameters_from_files` with the resolved path. Precedence is: values in `parameters` (inline or file) first, then the store.

Stored parameters live in the server process's memory, so their lifetime is the process's, not a conversation's:

- A client that keeps one server process for all its chats (Claude Desktop does) shares the store across those chats. Results list which names a problem took from the store (`parameters_from_store`) so stale data from another conversation is easy to spot.
- A client that restarts the server (on reconnect, model reload, or app restart) loses the store.
- Different client apps each run their own process, with separate stores.
- In HTTP mode, every connected client shares one store.

Every tool result includes a `server` block — process id, start time, uptime, version and the names currently stored — so you can tell whether two calls reached the same process. `load_parameter` only reads files inside the allowed data folders: your home folder by default, or the folders listed in the `CONVEX_OPT_MCP_DATA_DIRS` environment variable (separated by `:` on macOS/Linux). Symlinks are resolved before the check. Files are loaded as numbers only (`.npy`/`.npz` with pickling disabled).

### Results

`solve_problem` returns:

| Field | Meaning |
|---|---|
| `status` | CVXPY status (`optimal`, `optimal_inaccurate`, `infeasible`, `unbounded`, …) or `build_error` / `not_dcp` / `solver_error` with an `error` message |
| `optimal_value` | Objective value (`null` when infeasible or unbounded) |
| `variables`, `duals` | Values per variable and per constraint (keyed by the constraint string); only for `optimal` / `optimal_inaccurate` |
| `diagnostics` | Solver name, solve/setup/compile times, iteration count, and the largest constraint violation at the returned point |
| `parameters_from_store` | Names taken from the parameter store rather than passed inline |
| `parameters_from_files` | Parameters read from file references in this call, with their resolved paths |
| `server` | Process id, start time, uptime, version and stored names (on every tool's result) |

**Dual sign convention** (CVXPY's):

- *Inequality* duals are always ≥ 0, however the constraint is written. A dual `y` means loosening the constraint by `d` improves the optimal value by about `y·d`.
- *Equality* duals can have either sign. For `expr == c` with constant `c`, raising `c` by `d` changes the optimal value by about `−y·d` when minimizing and `+y·d` when maximizing.
- Equivalently, when minimizing `f` the Lagrangian is `f + Σ yᵢ·(lhsᵢ − rhsᵢ)`, with `a >= b` read as `b <= a`; when maximizing, the same with `−f`.

## Install

```bash
pip install convex-opt-mcp
```

Requires Python 3.10+. Works with MCP Python SDK v1 or v2.

## Configure your client

**Claude Desktop** — add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS), inside `"mcpServers"`:

```json
"convex-opt": {
  "command": "/absolute/path/to/convex-opt-mcp"
}
```

Use the full path from `which convex-opt-mcp`; desktop apps don't inherit your shell's `PATH`. Restart Claude Desktop afterwards.

**Claude Code:**

```bash
claude mcp add convex-opt -- convex-opt-mcp
```

**Other clients (Cursor, VS Code, LM Studio, …)** take the same command in their own MCP config file.

**Over HTTP** — for clients that connect to a URL rather than launching a process:

```bash
convex-opt-mcp --transport http            # serves http://127.0.0.1:8000/mcp
convex-opt-mcp --transport http --port 9000
```

It listens on this machine only by default. The server evaluates model-written expressions and can read files under its data folders, so don't bind it to `0.0.0.0` or expose it to the internet without putting authentication in front of it.

## Example

> Using the convex-opt tools, maximize μᵀw − (γ/2)·wᵀCw subject to Σw = 1 with γ = 1, for this μ and C: …

For this equality-constrained problem the result can be checked against the closed form w* = (1/γ)C⁻¹(μ − λ1); the test suite does exactly that, and also checks that the dual on `cp.sum(w) == 1` equals λ.

## Development

```bash
git clone <repo-url> && cd convex-opt-mcp
python -m pip install -e ".[test]"
pytest
```

To inspect the tools interactively: `npx @modelcontextprotocol/inspector convex-opt-mcp`.

## License

MIT
