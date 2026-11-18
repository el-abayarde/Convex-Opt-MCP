"""
convex-opt-mcp: an MCP server that exposes convex optimization (via CVXPY)
to LLM clients. Unofficial; not affiliated with the CVXPY project.

Works with MCP Python SDK v2 (MCPServer) or v1 (FastMCP, >= 1.10).

Run (stdio):  convex-opt-mcp      (after `pip install convex-opt-mcp`)
Run (HTTP):   convex-opt-mcp --transport http [--host 127.0.0.1] [--port 8000]
              (serves streamable HTTP at http://HOST:PORT/mcp)

Environment:
    CONVEX_OPT_MCP_DATA_DIRS   Folders that load_parameter may read from,
                               separated by os.pathsep (":" on macOS/Linux).
                               Defaults to the user's home folder.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Union

import cvxpy as cp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

try:  # MCP SDK v2
    from mcp.server import MCPServer

    _SDK_V2 = True
except ImportError:  # MCP SDK v1
    from mcp.server.fastmcp import FastMCP as MCPServer

    _SDK_V2 = False

try:
    from convex_opt_mcp import __version__ as _VERSION
except ImportError:  # running server.py outside the package
    _VERSION = "unknown"


# ==========================================================================
# Portable tool schemas
# ==========================================================================
# Pydantic emits JSON Schema with `$ref`/`$defs` for nested models and
# `anyOf: [..., {"type": "null"}]` for optional fields. Claude handles both,
# but several other clients and model APIs (e.g. OpenAI strict function
# calling, Gemini function declarations) accept only a subset of JSON Schema.
# We therefore serve every tool's schemas with references inlined, and input
# schemas with the null branches of optional fields removed (an omitted field
# already means "use the default").


def _inline_refs(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    if depth > 50:
        raise ValueError("Schema nesting too deep (recursive model?)")
    if isinstance(node, dict):
        if "$ref" in node:
            target = defs[node["$ref"].rsplit("/", 1)[-1]]
            merged = {**_inline_refs(target, defs, depth + 1)}
            merged.update({k: v for k, v in node.items() if k != "$ref"})
            return _inline_refs(merged, defs, depth + 1)
        return {k: _inline_refs(v, defs, depth + 1) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs, depth + 1) for v in node]
    return node


def _drop_nulls(node: Any) -> Any:
    if isinstance(node, list):
        return [_drop_nulls(v) for v in node]
    if not isinstance(node, dict):
        return node
    node = {k: _drop_nulls(v) for k, v in node.items()}
    if isinstance(node.get("anyOf"), list):
        branches = [b for b in node["anyOf"] if b != {"type": "null"}]
        if len(branches) != len(node["anyOf"]):
            rest = {k: v for k, v in node.items() if k != "anyOf"}
            node = {**branches[0], **rest} if len(branches) == 1 else {**rest, "anyOf": branches}
            if node.get("default", 0) is None:
                del node["default"]
    return node


def portable_schema(schema: dict[str, Any] | None, *, for_input: bool) -> dict[str, Any] | None:
    """Inline `$ref`s (and, for inputs, drop null branches) in a JSON Schema."""
    if not schema:
        return schema
    out = _inline_refs(copy.deepcopy(schema), schema.get("$defs", {}))
    return _drop_nulls(out) if for_input else out


class _PortableServer(MCPServer):
    """MCP server whose tool listings use the portable schemas above."""

    async def list_tools(self):  # type: ignore[override]
        tools = await super().list_tools()
        fixed = []
        for t in tools:
            fields = type(t).model_fields  # v2: input_schema/output_schema; v1: inputSchema/outputSchema
            in_f = "input_schema" if "input_schema" in fields else "inputSchema"
            out_f = "output_schema" if "output_schema" in fields else "outputSchema"
            update = {in_f: portable_schema(getattr(t, in_f), for_input=True)}
            if out_f in fields:
                update[out_f] = portable_schema(getattr(t, out_f), for_input=False)
            fixed.append(t.model_copy(update=update))
        return fixed


mcp = _PortableServer(
    name="convex-opt-mcp",
    instructions=(
        "Solve convex optimization problems with CVXPY. Declare variables, then "
        "write the objective and constraints as CVXPY expression strings using "
        "`cp.` functions (e.g. cp.sum_squares(A @ x - b)) and `@` for matrix "
        "products. Problems must follow DCP rules; if a solve returns not_dcp, "
        "call check_problem to see which part is at fault. For large or reused "
        "data, store arrays once with set_parameter or load_parameter (from "
        ".npy, .npz, .csv, .txt, .tsv or .json files) and refer to them by name "
        "instead of resending them on every call."
    ),
)

# Inline numeric data: a number, a vector, or a matrix (row-major nested lists).
# Higher-dimensional arrays can still be loaded from .npy/.npz files.
Numeric = Union[float, list[float], list[list[float]]]

# ==========================================================================
# Schemas
# ==========================================================================


class VariableSpec(BaseModel):
    name: str = Field(description="Python identifier used in expressions, e.g. 'x'")
    shape: list[int] = Field(
        default_factory=list, description="[] for a scalar, [n] for a vector, [m, n] for a matrix"
    )
    nonneg: bool = Field(False, description="Constrain all entries to be >= 0")
    pos: bool = Field(False, description="Entries > 0 (required for geometric programs, gp=true)")
    neg: bool = Field(False, description="Entries < 0")
    boolean: bool = Field(False, description="Entries in {0, 1} (needs a mixed-integer solver such as HIGHS)")
    integer: bool = Field(False, description="Integer entries (needs a mixed-integer solver such as HIGHS)")
    symmetric: bool = Field(False, description="Symmetric square matrix")
    PSD: bool = Field(False, description="Symmetric positive semidefinite square matrix")


class FileRef(BaseModel):
    """A parameter whose value is read from a file on the server's machine."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(
        description="Path to a .npy, .npz, .csv, .tsv, .txt or .json file (~ allowed), inside the allowed data folders"
    )
    key: str | None = Field(None, description="Array name in a .npz file, or top-level field in a .json file")
    delimiter: str | None = Field(None, description="Column separator for .csv/.tsv/.txt if not the default")
    skip_header: int = Field(0, description="Header rows to skip in .csv/.tsv/.txt")


# Solver options are passed to CVXPY as keyword arguments. Over JSON they can only
# be numbers, strings, booleans, or lists/objects of those; options that need
# Python objects (callables, protobufs, solver environments) can't be sent.
OptionScalar = Union[bool, int, float, str]
OptionValue = Union[OptionScalar, list[OptionScalar], dict[str, OptionScalar]]


class SolverChoice(BaseModel):
    """One entry of solver_path: a solver plus its own options."""

    model_config = ConfigDict(extra="forbid")

    solver: str = Field(description="Solver name, e.g. 'SCS'")
    options: dict[str, OptionValue] = Field(
        default_factory=dict, description="Options for this solver, e.g. {'max_iters': 10000}"
    )


class ProblemSpec(BaseModel):
    variables: list[VariableSpec] = Field(description="Decision variables")
    sense: Literal["minimize", "maximize"]
    objective: str = Field(description="CVXPY expression string, e.g. 'cp.sum_squares(A @ x - b)'")
    constraints: list[str] = Field(
        default_factory=list,
        description="CVXPY constraint strings using ==, <= or >=, e.g. ['x >= 0', 'cp.sum(x) == 1']",
    )
    parameters: dict[str, Union[Numeric, FileRef]] = Field(
        default_factory=dict,
        description=(
            "Numeric data by name. Each value is either inline (a number, a vector, or a "
            "matrix as a list of rows) or a file reference {\"file\": \"~/data/C.npy\"}, "
            "read fresh on every call, so it does not depend on the parameter store. "
            "Parameters stored with set_parameter/load_parameter are available "
            "automatically; values given here take precedence."
        ),
    )


class ServerInfo(BaseModel):
    """Identifies the server process that handled a call."""

    pid: int = Field(description="Process ID. A different pid between calls means a different (or restarted) server")
    started_at: str = Field(description="When this server process started (local time, ISO 8601)")
    uptime_s: float = Field(description="Seconds since this server process started")
    version: str = Field(description="convex-opt-mcp version")
    stored_parameters: list[str] = Field(description="Names currently in this process's parameter store")


def _server_info() -> ServerInfo:
    return ServerInfo(
        pid=os.getpid(),
        started_at=_STARTED_AT_ISO,
        uptime_s=round(time.time() - _STARTED_AT, 3),
        version=_VERSION,
        stored_parameters=list(_STORE),
    )


class _Reply(BaseModel):
    """Base for every tool result: adds the `server` diagnostics block."""

    server: ServerInfo = Field(
        default_factory=lambda: _server_info(),
        description=(
            "Which server process answered. Stored parameters live in this process's memory, "
            "so if pid or started_at changes between calls, earlier set_parameter/load_parameter "
            "data is gone and must be set again."
        ),
    )


class SolversResult(_Reply):
    solvers: list[str] = Field(description="CVXPY solver names usable as the `solver` argument")


class ParameterInfo(BaseModel):
    name: str
    shape: list[int]
    min: float | None = None
    max: float | None = None
    symmetric: bool | None = Field(None, description="Square matrices only")
    min_eigenvalue: float | None = Field(
        None, description="Symmetric matrices up to 2000x2000; >= 0 (up to rounding) means PSD"
    )
    source: str = Field(description="'inline' or the file it was loaded from")


class StoredParameter(ParameterInfo, _Reply):
    """Result of set_parameter / load_parameter."""


class ParameterList(_Reply):
    parameters: list[ParameterInfo]


class ClearResult(_Reply):
    cleared: list[str]


class CheckResult(_Reply):
    ok: bool
    error: str | None = None
    is_dcp: bool | None = None
    is_dgp: bool | None = Field(None, description="Valid as a geometric program (solve with gp=true)")
    is_dqcp: bool | None = Field(None, description="Valid as a quasiconvex program (solve with qcp=true)")
    is_mixed_integer: bool | None = None
    objective_curvature: str | None = None
    objective_is_dcp: bool | None = None
    non_dcp_constraints: list[str] | None = Field(None, description="Constraints that violate DCP rules")
    n_scalar_variables: int | None = None
    n_constraints: int | None = None
    parameters_from_store: list[str] | None = Field(
        None, description="Names the problem took from the parameter store rather than from inline `parameters`"
    )
    parameters_from_files: dict[str, str] | None = Field(
        None, description="Parameters read from file references in this call, with the resolved path"
    )


class Diagnostics(BaseModel):
    solver: str | None = None
    solve_time_s: float | None = None
    setup_time_s: float | None = None
    compile_time_s: float | None = Field(None, description="Time CVXPY spent canonicalizing the problem")
    num_iters: int | None = None
    max_constraint_violation: float | None = Field(
        None, description="Largest constraint residual at the returned point (0 = exactly feasible)"
    )
    options_used: dict[str, Any] | None = Field(
        None,
        description="Settings actually passed to CVXPY: solver, gp, qcp and solver options "
        "(including any server default such as a time limit)",
    )
    attempts: list[dict[str, Any]] | None = Field(
        None, description="With solver_path: each solver tried, in order, and the error if it failed"
    )


class SolveResult(_Reply):
    status: str = Field(
        description=(
            "CVXPY status (optimal, optimal_inaccurate, infeasible, infeasible_inaccurate, "
            "unbounded, unbounded_inaccurate, infeasible_or_unbounded, user_limit, solver_error) "
            "or a server-side failure (build_error, invalid_options, not_dcp, not_dgp, not_dqcp)"
        )
    )
    error: str | None = None
    optimal_value: float | None = None
    variables: dict[str, Any] | None = Field(None, description="Variable values: number or nested lists")
    duals: dict[str, Any] | None = Field(None, description="Dual value per constraint, keyed by its source string")
    diagnostics: Diagnostics | None = None
    parameters_from_store: list[str] | None = Field(
        None,
        description=(
            "Names the problem took from the parameter store rather than from inline "
            "`parameters`. Check these are the data you meant: the store is shared by "
            "every conversation that uses this server process."
        ),
    )
    parameters_from_files: dict[str, str] | None = Field(
        None, description="Parameters read from file references in this call, with the resolved path"
    )
    installed_solvers: list[str] | None = None


# ==========================================================================
# Parameter store
# ==========================================================================
# Data kept in this server process's memory between calls, so large arrays
# need to be sent or loaded only once. Its lifetime is the process's, not a
# conversation's: a client that keeps one process for all its chats (Claude
# Desktop) shares the store across them, a client that restarts the process
# loses it, and in HTTP mode all connected clients share one store.

_STARTED_AT = time.time()
_STARTED_AT_ISO = datetime.fromtimestamp(_STARTED_AT).astimezone().isoformat(timespec="seconds")
_STORE: dict[str, np.ndarray | float] = {}
_SOURCES: dict[str, str] = {}
_RESERVED = {"cp", "np"}


def _check_name(name: str) -> None:
    if not name.isidentifier() or name.startswith("_") or name in _RESERVED:
        raise ValueError(f"Bad parameter name {name!r}: use a Python identifier not starting with '_'")


def _as_numeric(value: Any, what: str) -> np.ndarray | float:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be numeric (a number or nested lists of numbers)") from None
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} contains NaN or infinite values")
    return float(arr) if arr.ndim == 0 else arr


def _describe(name: str, value: np.ndarray | float) -> ParameterInfo:
    arr = np.asarray(value)
    info = ParameterInfo(
        name=name,
        shape=list(arr.shape),
        min=float(arr.min()) if arr.size else None,
        max=float(arr.max()) if arr.size else None,
        source=_SOURCES.get(name, "inline"),
    )
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        info.symmetric = bool(np.allclose(arr, arr.T))
        if info.symmetric and arr.shape[0] <= 2000:
            info.min_eigenvalue = float(np.linalg.eigvalsh(arr)[0])
    return info


def _data_roots() -> list[Path]:
    env = os.environ.get("CONVEX_OPT_MCP_DATA_DIRS")
    roots = env.split(os.pathsep) if env else [str(Path.home())]
    return [Path(r).expanduser().resolve() for r in roots if r.strip()]


def _resolve_data_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    roots = _data_roots()
    if not any(p.is_relative_to(root) for root in roots):
        allowed = ", ".join(str(r) for r in roots)
        raise ValueError(
            f"{p} is outside the allowed data folders ({allowed}). "
            "Set CONVEX_OPT_MCP_DATA_DIRS in the server's environment to allow more."
        )
    if not p.is_file():
        raise ValueError(f"No such file: {p}")
    return p


def _load_array(p: Path, key: str | None, delimiter: str | None, skip_header: int) -> Any:
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return np.load(p, allow_pickle=False)
    if suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            if key is None:
                if len(z.files) != 1:
                    raise ValueError(f"{p.name} contains {z.files}; pass `key` to choose one")
                key = z.files[0]
            if key not in z.files:
                raise ValueError(f"{p.name} has no array {key!r}; it contains {z.files}")
            return z[key]
    if suffix in (".csv", ".tsv", ".txt"):
        delim = delimiter or {".csv": ",", ".tsv": "\t"}.get(suffix)
        try:
            return np.loadtxt(p, delimiter=delim, skiprows=skip_header, ndmin=1)
        except ValueError as e:
            raise ValueError(
                f"Could not parse {p.name} as numbers ({e}). If it has a header row, "
                "pass skip_header=1; if it isn't comma-separated, pass `delimiter`."
            ) from None
    if suffix == ".json":
        data = json.loads(p.read_text())
        if key is not None:
            if not isinstance(data, dict) or key not in data:
                raise ValueError(f"{p.name} has no top-level key {key!r}")
            data = data[key]
        return data
    raise ValueError(f"Unsupported file type {suffix!r}; use .npy, .npz, .csv, .tsv, .txt or .json")


# ==========================================================================
# Safe-ish expression evaluation
# ==========================================================================
# Expressions come from the model, so never hand them to bare eval().
# We parse them, allow only a small set of AST node types, forbid dunder
# attribute access, and evaluate with almost no builtins. This blocks the
# usual escape routes, but it is not a hardened sandbox: run the server
# locally for your own use, not as a public service.

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Attribute,
    ast.Name, ast.Constant, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Load,
    ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.MatMult, ast.Pow, ast.USub, ast.UAdd,
    ast.Eq, ast.LtE, ast.GtE, ast.Lt, ast.Gt,
)
_SAFE_BUILTINS = {"abs": abs, "sum": sum, "range": range, "len": len, "min": min, "max": max}


def _safe_eval(src: str, namespace: dict[str, Any]) -> Any:
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in {src!r}: {e.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax ({type(node).__name__}) in {src!r}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"Private attribute access is not allowed in {src!r}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError(f"Names starting with '_' are not allowed in {src!r}")
        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            raise ValueError(
                f"Chained comparisons don't work in CVXPY ({src!r}); split into two constraints"
            )
    try:
        return eval(compile(tree, "<expr>", "eval"), {"__builtins__": _SAFE_BUILTINS}, namespace)
    except NameError as e:
        raise ValueError(
            f"{e} in {src!r}. Declare it as a variable, pass it in `parameters`, or store it "
            "with set_parameter/load_parameter (list_parameters shows what is stored)."
        ) from None


def _build(
    spec: ProblemSpec,
) -> tuple[cp.Problem, dict[str, cp.Variable], list[str], list[str], dict[str, str]]:
    """Return (problem, variables, constraint sources, names taken from the store,
    parameters read from files)."""
    ns: dict[str, Any] = {"cp": cp, "np": np, **_STORE}
    from_files: dict[str, str] = {}
    for k, v in spec.parameters.items():
        _check_name(k)
        if isinstance(v, dict):  # not yet validated, e.g. a ProblemSpec modified after creation
            v = FileRef.model_validate(v)
        if isinstance(v, FileRef):
            p = _resolve_data_path(v.file)
            raw = _load_array(p, v.key, v.delimiter, v.skip_header)
            ns[k] = _as_numeric(raw, f"Data in {p.name} (parameter {k!r})")
            from_files[k] = str(p) + (f" [{v.key}]" if v.key else "")
        else:
            ns[k] = _as_numeric(v, f"Parameter {k!r}")

    variables: dict[str, cp.Variable] = {}
    for vs in spec.variables:
        if not vs.name.isidentifier() or vs.name.startswith("_") or vs.name in _RESERVED:
            raise ValueError(f"Bad variable name {vs.name!r}")
        if vs.name in variables:
            raise ValueError(f"Duplicate variable name {vs.name!r}")
        if vs.name in ns:
            raise ValueError(f"Variable {vs.name!r} clashes with a parameter of the same name")
        attrs = {a: True for a in ("nonneg", "pos", "neg", "boolean", "integer", "symmetric", "PSD") if getattr(vs, a)}
        variables[vs.name] = ns[vs.name] = cp.Variable(tuple(vs.shape), name=vs.name, **attrs)

    obj_expr = _safe_eval(spec.objective, ns)
    objective = cp.Minimize(obj_expr) if spec.sense == "minimize" else cp.Maximize(obj_expr)

    constraints = []
    for c in spec.constraints:
        con = _safe_eval(c, ns)
        if not isinstance(con, cp.constraints.constraint.Constraint):
            raise ValueError(f"{c!r} is not a constraint (use ==, <=, >=, or a cp.* cone)")
        constraints.append(con)

    return cp.Problem(objective, constraints), variables, spec.constraints, _names_from_store(spec), from_files


def _names_from_store(spec: ProblemSpec) -> list[str]:
    used: set[str] = set()
    for src in [spec.objective, *spec.constraints]:
        used |= {n.id for n in ast.walk(ast.parse(src, mode="eval")) if isinstance(n, ast.Name)}
    return sorted(n for n in used if n in _STORE and n not in spec.parameters)


def _to_json(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):  # e.g. second-order-cone duals: one array per part
        return [_to_json(x) for x in v]
    arr = np.asarray(v, dtype=float)
    return arr.item() if arr.ndim == 0 else arr.tolist()


def _finite_or_none(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _int_or_none(v: Any) -> int | None:
    f = _finite_or_none(v)
    return None if f is None else int(f)


def _max_violation(constraints: list) -> float | None:
    worst = 0.0
    try:
        for c in constraints:
            viol = np.asarray(c.violation(), dtype=float)
            if viol.size:
                worst = max(worst, float(np.max(viol)))
    except Exception:  # some cones don't report violations; don't fail the solve
        return None
    return worst


# ==========================================================================
# Solve options
# ==========================================================================

# Arguments of solve_problem itself; passing them inside solver options would
# bypass the checks below or clash with CVXPY's own keywords.
_RESERVED_OPTION_NAMES = {
    "solver", "solver_path", "verbose", "gp", "qcp", "requires_grad", "enforce_dpp", "ignore_dpp",
}

# Option-name words that indicate reading or writing files (HiGHS log_file,
# write_model_file, SCS write_data_filename, MOSEK *_WRITE_* parameters, ...).
_FILE_WORDS = {"file", "filename", "filepath", "path", "dir", "directory", "folder", "write", "save", "dump"}

# Per-solver name of the time-limit option used for CONVEX_OPT_MCP_TIME_LIMIT.
_TIME_LIMIT_OPTION = {"CLARABEL": "time_limit", "OSQP": "time_limit", "SCS": "time_limit_secs", "HIGHS": "time_limit"}


def _option_words(name: str) -> list[str]:
    words = []
    for part in re.split(r"[^A-Za-z0-9]+", name):
        words += re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", part)
    return [w.lower() for w in words]


def _is_file_option(name: str) -> bool:
    words = _option_words(name)
    return any(w in _FILE_WORDS or w.endswith(("file", "filename", "path")) for w in words)


def _file_options_allowed() -> bool:
    return os.environ.get("CONVEX_OPT_MCP_ALLOW_FILE_OPTIONS", "").strip().lower() in ("1", "true", "yes")


def _check_options(options: dict[str, Any], where: str) -> None:
    def walk(opts: dict[str, Any], top: bool) -> None:
        for key, value in opts.items():
            if top and key in _RESERVED_OPTION_NAMES:
                raise ValueError(
                    f"{where}: {key!r} is an argument of solve_problem itself; pass it directly, "
                    "not inside solver options"
                )
            if _is_file_option(key) and not _file_options_allowed():
                raise ValueError(
                    f"{where}: option {key!r} looks like it reads or writes files, which is disabled. "
                    "Set CONVEX_OPT_MCP_ALLOW_FILE_OPTIONS=1 in the server's environment to allow it."
                )
            if isinstance(value, dict):
                walk(value, top=False)

    walk(options, top=True)


def _with_time_limit(solver: str | None, options: dict[str, Any]) -> dict[str, Any]:
    """Add the server's default time limit, if configured and not set by the caller."""
    raw = os.environ.get("CONVEX_OPT_MCP_TIME_LIMIT", "").strip()
    key = _TIME_LIMIT_OPTION.get((solver or "").upper())
    if not raw or key is None or key in options:
        return dict(options)
    return {**options, key: float(raw)}


def _rule_violations(prob: cp.Problem, con_src: list[str], kind: str) -> str | None:
    """Describe which parts break DCP/DGP/DQCP rules, or None if the problem follows them."""
    check = {"dcp": "is_dcp", "dgp": "is_dgp", "dqcp": "is_dqcp"}[kind]
    if getattr(prob, check)():
        return None
    where = []
    if not getattr(prob.objective, check)():
        where.append("the objective")
    bad = [src for src, c in zip(con_src, prob.constraints) if not getattr(c, check)()]
    if bad:
        where.append("constraints " + ", ".join(repr(b) for b in bad))
    return "; ".join(where) or "the problem as a whole"


# ==========================================================================
# Tools: solvers and data
# ==========================================================================


@mcp.tool()
def list_solvers() -> SolversResult:
    """List the CVXPY solvers installed on this machine."""
    return SolversResult(solvers=cp.installed_solvers())


@mcp.tool()
def set_parameter(name: str, value: Numeric) -> StoredParameter:
    """Store a number, vector (list of numbers) or matrix (list of rows) under
    `name` for use in later problems.

    Stored parameters are available by name in every objective and constraint,
    so large data (a covariance matrix, say) is sent once instead of on every
    call. Overwrites any existing parameter with the same name.
    Example: set_parameter(name="C", value=[[0.04, 0.01], [0.01, 0.09]])

    The store lives in the server process's memory: it is lost when the server
    restarts, and it may be shared with other conversations using the same
    server. Every result's `server` block shows the process id and stored names;
    if the pid changes or a name is missing, set the data again.
    """
    _check_name(name)
    _STORE[name] = _as_numeric(value, f"Parameter {name!r}")
    _SOURCES[name] = "inline"
    return StoredParameter(**_describe(name, _STORE[name]).model_dump())


@mcp.tool()
def load_parameter(
    name: str,
    path: str,
    key: str | None = None,
    delimiter: str | None = None,
    skip_header: int = 0,
) -> StoredParameter:
    """Load a numeric array from a file on this machine and store it under `name`.

    Supported: .npy, .npz (`key` picks the array if there are several),
    .csv / .tsv / .txt (`delimiter` and `skip_header` for header rows), and
    .json (a number or nested lists; `key` picks a top-level field).
    Paths may use ~. Only files inside the allowed data folders can be read
    (the user's home folder unless CONVEX_OPT_MCP_DATA_DIRS says otherwise).
    Returns the array's shape and summary statistics, not its contents.
    """
    _check_name(name)
    p = _resolve_data_path(path)
    raw = _load_array(p, key, delimiter, skip_header)
    _STORE[name] = _as_numeric(raw, f"Data in {p.name}")
    _SOURCES[name] = str(p) + (f" [{key}]" if key else "")
    return StoredParameter(**_describe(name, _STORE[name]).model_dump())


@mcp.tool()
def list_parameters() -> ParameterList:
    """List stored parameters with their shapes, value ranges and, for symmetric
    matrices, the smallest eigenvalue (useful to confirm a covariance is PSD)."""
    return ParameterList(parameters=[_describe(k, v) for k, v in _STORE.items()])


@mcp.tool()
def clear_parameters(names: list[str] | None = None) -> ClearResult:
    """Delete the named stored parameters, or all of them if `names` is omitted."""
    targets = list(_STORE) if names is None else [n for n in names if n in _STORE]
    for n in targets:
        _STORE.pop(n, None)
        _SOURCES.pop(n, None)
    return ClearResult(cleared=targets)


# ==========================================================================
# Tools: problems
# ==========================================================================


@mcp.tool()
def check_problem(problem: ProblemSpec) -> CheckResult:
    """Build the problem without solving it and report DCP compliance, curvature
    and size, including which constraints (if any) break DCP rules. Use this to
    debug a formulation that fails to solve."""
    try:
        prob, _, con_src, from_store, from_files = _build(problem)
    except Exception as e:
        return CheckResult(ok=False, error=str(e))
    return CheckResult(
        ok=True,
        is_dcp=prob.is_dcp(),
        is_dgp=prob.is_dgp(),
        is_dqcp=prob.is_dqcp(),
        is_mixed_integer=prob.is_mixed_integer(),
        objective_curvature=prob.objective.expr.curvature,
        objective_is_dcp=prob.objective.is_dcp(),
        non_dcp_constraints=[s for s, c in zip(con_src, prob.constraints) if not c.is_dcp()],
        n_scalar_variables=prob.size_metrics.num_scalar_variables,
        n_constraints=len(prob.constraints),
        parameters_from_store=from_store,
        parameters_from_files=from_files,
    )


@mcp.tool()
def solve_problem(
    problem: ProblemSpec,
    solver: str | None = None,
    solver_options: dict[str, OptionValue] | None = None,
    solver_path: list[Union[str, SolverChoice]] | None = None,
    gp: bool = False,
    qcp: bool = False,
    verbose: bool = False,
) -> SolveResult:
    """Solve a convex (DCP), geometric (gp) or quasiconvex (qcp) problem with CVXPY.

    Write the objective and constraints as Python/CVXPY expression strings over
    the declared variables, the parameters, `cp` (cvxpy) and `np` (numpy).
    Use `@` for matrix-vector products and `*` for scalar or elementwise ones.

    Examples of `problem`:
      Least squares with bounds:
        {"variables": [{"name": "x", "shape": [3]}], "sense": "minimize",
         "objective": "cp.sum_squares(A @ x - b)", "constraints": ["x >= 0", "x <= 1"],
         "parameters": {"A": [[1, -0.5, 0], [0.5, 2, 0], [0, 1, 1]], "b": [2, 1, -1]}}
      Linear program:
        {"variables": [{"name": "x", "shape": [2], "nonneg": true}], "sense": "maximize",
         "objective": "c @ x", "constraints": ["A @ x <= b"],
         "parameters": {"c": [3, 5], "A": [[1, 0], [0, 2], [3, 2]], "b": [4, 12, 18]}}
      Mean-variance portfolio (mu and C stored earlier with set_parameter):
        {"variables": [{"name": "w", "shape": [5]}], "sense": "maximize",
         "objective": "mu @ w - (gamma / 2) * cp.quad_form(w, C)",
         "constraints": ["cp.sum(w) == 1", "w >= 0"], "parameters": {"gamma": 5}}
      Same problem with data read from files on every call (no stored state needed):
        "parameters": {"C": {"file": "~/data/cov.npy"},
                       "mu": {"file": "~/data/mu.csv", "skip_header": 1}, "gamma": 5}
      Semidefinite program:
        {"variables": [{"name": "X", "shape": [3, 3], "PSD": true}], "sense": "minimize",
         "objective": "cp.trace(C @ X)", "constraints": ["cp.trace(X) == 1"],
         "parameters": {"C": [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]}}
      Integer program (needs a mixed-integer solver, e.g. solver="HIGHS"):
        {"variables": [{"name": "z", "shape": [3], "boolean": true}], "sense": "maximize",
         "objective": "v @ z", "constraints": ["wt @ z <= 10"],
         "parameters": {"v": [6, 10, 12], "wt": [1, 5, 6]}}

    Common mistakes:
      - Chained comparisons (`0 <= x <= 1`): write two constraints instead.
      - Strict inequalities (`<`, `>`) are not allowed; use `<=` and `>=`.
      - `cp.quad_form(x, P)` needs a symmetric PSD matrix P; use cp.sum_squares
        or cp.norm for other quadratic forms.
      - Minimizing a concave or maximizing a convex expression is not DCP; so is
        multiplying two variables (x * y), unless the problem is a geometric
        program (see gp below).
      - `x * y` with a vector x and a matrix y is elementwise, not matrix multiplication.

    Solve options (all optional):
      solver: e.g. "CLARABEL", "SCS", "OSQP", "HIGHS"; list_solvers shows what is installed.
      solver_options: passed to the solver as keyword arguments, e.g.
        OSQP {"max_iter": 20000, "eps_abs": 1e-8, "eps_rel": 1e-8}; CLARABEL {"max_iter": 100,
        "time_limit": 10}; SCS {"max_iters": 10000, "eps": 1e-6}; HIGHS {"time_limit": 10,
        "presolve": "off"}; SCIPY {"scipy_options": {"method": "highs"}}.
      solver_path: solvers to try in order until one runs without error (instead of
        `solver`), e.g. ["CLARABEL", {"solver": "SCS", "options": {"max_iters": 10000}}].
      gp: solve as a geometric program. Variables must be declared "pos": true and data
        positive; products and powers of variables are then allowed, e.g.
        {"variables": [{"name": "x", "shape": [3], "pos": true}], "sense": "minimize",
         "objective": "cp.sum(x**2)", "constraints": ["x >= 1", "x[0] * x[1] <= 5"]}
      qcp: solve a quasiconvex problem (e.g. with cp.ceil, cp.floor or ratios) by bisection.
      Options naming files or folders (e.g. log_file) are refused, and options that are
      arguments of this tool (gp, qcp, solver, verbose) can't go inside solver_options.

    Returns status, optimal value, variable values, constraint duals and solver
    diagnostics (iterations, timings, max constraint violation, options used).

    Dual sign convention (CVXPY's):
      - Inequality duals are always >= 0, however the constraint is written. A
        dual y means loosening the constraint by d improves the optimal value
        by about y*d (lowers it when minimizing, raises it when maximizing).
      - Equality duals can have either sign. For `expr == c` with c constant,
        increasing c by d changes the optimal value by about -y*d when
        minimizing and +y*d when maximizing.
      - In Lagrangian terms: minimizing f, L = f + sum_i y_i * (lhs_i - rhs_i),
        with `a >= b` read as `b <= a`; maximizing f, the same with -f.
    The optimal value, variables and duals are only reported when the status is
    optimal or optimal_inaccurate (not for user_limit, infeasible, ...).
    Mixed-integer and qcp problems have no duals (they are null). With gp, duals are
    log-log sensitivities: a 1% loosening of a constraint improves the optimal
    value by about y%.
    """
    # 1. Check the options before building anything.
    try:
        if gp and qcp:
            raise ValueError("gp and qcp can't both be true")
        if solver and solver_path:
            raise ValueError("Give either solver or solver_path, not both")
        if solver_path is not None and not solver_path:
            raise ValueError("solver_path is empty")
        choices: list[tuple[str | None, dict[str, Any]]] = []
        if solver_path:
            for entry in solver_path:
                if isinstance(entry, dict):  # not yet validated (direct Python calls)
                    entry = SolverChoice.model_validate(entry)
                if isinstance(entry, SolverChoice):
                    choices.append((entry.solver, dict(entry.options)))
                else:
                    choices.append((entry, {}))
        else:
            choices.append((solver, dict(solver_options or {})))
        if solver_path and solver_options:
            raise ValueError("With solver_path, give options per solver: {'solver': ..., 'options': {...}}")
        for name, opts in choices:
            _check_options(opts, f"Options for {name or 'the default solver'}")
    except ValueError as e:
        return SolveResult(status="invalid_options", error=str(e))

    # 2. Build the problem.
    try:
        prob, variables, con_src, from_store, from_files = _build(problem)
    except Exception as e:
        return SolveResult(status="build_error", error=str(e))

    # 3. Check it follows the rules of the requested problem class.
    kind = "dgp" if gp else "dqcp" if qcp else "dcp"
    where = _rule_violations(prob, con_src, kind)
    if where is not None:
        hint = {
            "dcp": "Reformulate it; check_problem gives details (is_dgp/is_dqcp show if gp or qcp would work).",
            "dgp": "Geometric programs need variables declared \"pos\": true, positive data, and "
            "monomial/posynomial expressions.",
            "dqcp": "Reformulate it; check_problem gives details.",
        }[kind]
        return SolveResult(
            status=f"not_{kind}",
            error=f"Problem does not follow {kind.upper()} rules ({where}). {hint}",
            parameters_from_store=from_store,
            parameters_from_files=from_files,
        )

    # 4. Solve, trying each solver in turn when solver_path is given.
    mode = {"gp": True} if gp else {"qcp": True} if qcp else {}
    attempts: list[dict[str, Any]] = []
    used: dict[str, Any] | None = None
    for name, opts in choices:
        opts = _with_time_limit(name, opts)
        try:
            prob.solve(solver=name, verbose=verbose, **mode, **opts)
        except Exception as e:  # unknown solver, bad option, solver failure
            attempts.append({"solver": name, "ok": False, "error": f"{type(e).__name__}: {e}"})
            continue
        attempts.append({"solver": name, "ok": True})
        used = {"solver": name, "gp": gp, "qcp": qcp, "solver_options": opts}
        break

    if used is None:
        errors = "; ".join(f"{a['solver']}: {a['error']}" for a in attempts)
        return SolveResult(
            status="solver_error",
            error=errors if len(attempts) == 1 else f"All solvers in solver_path failed ({errors})",
            diagnostics=Diagnostics(attempts=attempts) if solver_path else None,
            parameters_from_store=from_store,
            parameters_from_files=from_files,
            installed_solvers=cp.installed_solvers(),
        )

    stats = prob.solver_stats
    diagnostics = Diagnostics(
        solver=getattr(stats, "solver_name", None) or used["solver"],
        solve_time_s=_finite_or_none(getattr(stats, "solve_time", None)),
        setup_time_s=_finite_or_none(getattr(stats, "setup_time", None)),
        compile_time_s=_finite_or_none(getattr(prob, "compilation_time", None)),
        num_iters=_int_or_none(getattr(stats, "num_iters", None)),
        options_used=used,
        attempts=attempts if solver_path else None,
    )

    solved = prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    if solved:
        diagnostics.max_constraint_violation = _max_violation(prob.constraints)

    return SolveResult(
        status=prob.status,
        optimal_value=_finite_or_none(prob.value) if solved else None,
        variables={k: _to_json(v.value) for k, v in variables.items()} if solved else None,
        duals={s: _to_json(c.dual_value) for s, c in zip(con_src, prob.constraints)} if solved else None,
        diagnostics=diagnostics,
        parameters_from_store=from_store,
        parameters_from_files=from_files,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="convex-opt-mcp", description="MCP server for convex optimization with CVXPY."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="stdio (default) for clients that launch the server; http to serve streamable HTTP at /mcp",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP only. Default 127.0.0.1 (this machine only). Only use 0.0.0.0 on a trusted network.",
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP only. Default 8000.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point."""
    args = _parse_args(argv)
    if args.transport == "stdio":
        mcp.run()
    elif _SDK_V2:
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:  # SDK v1 reads host/port from settings
        mcp.settings.host, mcp.settings.port = args.host, args.port
        mcp.run("streamable-http")


if __name__ == "__main__":
    main()
