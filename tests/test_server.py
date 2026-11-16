import asyncio
import json

import numpy as np
import pytest
from pydantic import ValidationError

from convex_opt_mcp import server
from convex_opt_mcp.server import (
    ProblemSpec,
    check_problem,
    clear_parameters,
    list_parameters,
    list_solvers,
    load_parameter,
    portable_schema,
    set_parameter,
    solve_problem,
)

MU = [0.00945, -0.02614, -0.02065, -0.12207, 0.08999]
C = [
    [0.06559, -0.00378, -0.01774, -0.00917, -0.02367],
    [-0.00378, 0.08752, -0.00557, 0.00104, -0.02366],
    [-0.01774, -0.00557, 0.04721, -0.0284, 0.00843],
    [-0.00917, 0.00104, -0.0284, 0.08743, 0.02254],
    [-0.02367, -0.02366, 0.00843, 0.02254, 0.05559],
]


@pytest.fixture(autouse=True)
def empty_store():
    clear_parameters()
    yield
    clear_parameters()


def mean_variance(gamma: float, inline: bool = True) -> ProblemSpec:
    return ProblemSpec(
        variables=[{"name": "w", "shape": [5]}],
        sense="maximize",
        objective="mu @ w - (gamma / 2) * cp.quad_form(w, C)",
        constraints=["cp.sum(w) == 1"],
        parameters={"mu": MU, "C": C, "gamma": gamma} if inline else {"gamma": gamma},
    )


def closed_form(gamma: float):
    mu, Cm, one = np.array(MU), np.array(C), np.ones(5)
    Ci_mu, Ci_1 = np.linalg.solve(Cm, mu), np.linalg.solve(Cm, one)
    lam = (one @ Ci_mu - gamma) / (one @ Ci_1)
    w = (Ci_mu - lam * Ci_1) / gamma
    return w, mu @ w - gamma / 2 * w @ Cm @ w, lam


def scalar_problem(sense, objective, constraint) -> ProblemSpec:
    return ProblemSpec(
        variables=[{"name": "x", "shape": []}], sense=sense, objective=objective, constraints=[constraint]
    )


# --- solving -------------------------------------------------------------


@pytest.mark.parametrize("gamma", [1.0, 5.0, 20.0])
def test_mean_variance_matches_closed_form(gamma):
    w_star, obj_star, lam = closed_form(gamma)
    res = solve_problem(mean_variance(gamma))

    assert res.status == "optimal"
    np.testing.assert_allclose(res.variables["w"], w_star, atol=1e-5)
    assert res.optimal_value == pytest.approx(obj_star, abs=1e-6)
    assert res.duals["cp.sum(w) == 1"] == pytest.approx(lam, abs=1e-5)


def test_nonneg_bounds_are_respected():
    spec = mean_variance(1.0)
    spec.constraints.append("w >= 0")
    res = solve_problem(spec)
    assert res.status == "optimal"
    assert min(res.variables["w"]) >= -1e-6
    assert sum(res.variables["w"]) == pytest.approx(1.0, abs=1e-6)


def test_diagnostics_are_reported():
    res = solve_problem(mean_variance(1.0))
    d = res.diagnostics
    assert d.solver
    assert d.solve_time_s is None or d.solve_time_s >= 0
    assert d.max_constraint_violation is not None and d.max_constraint_violation < 1e-6


def test_infeasible_problem_has_no_values():
    res = solve_problem(
        ProblemSpec(
            variables=[{"name": "x", "shape": []}],
            sense="minimize",
            objective="x",
            constraints=["x >= 1", "x <= 0"],
        )
    )
    assert res.status.startswith("infeasible")
    assert res.variables is None and res.duals is None and res.optimal_value is None


# --- dual sign convention (documented in solve_problem's docstring) -------


@pytest.mark.parametrize(
    "constraint", ["x <= 1", "1 >= x"]
)
def test_inequality_dual_is_nonnegative_either_way_round(constraint):
    res = solve_problem(scalar_problem("minimize", "cp.square(x - 2)", constraint))
    assert res.duals[constraint] == pytest.approx(2.0, abs=1e-4)


def test_inequality_dual_nonnegative_for_ge():
    res = solve_problem(scalar_problem("minimize", "cp.square(x - 2)", "x >= 3"))
    assert res.duals["x >= 3"] == pytest.approx(2.0, abs=1e-4)


def test_equality_dual_sensitivity_minimize():
    # p*(c) = (c - 2)^2, dp*/dc at c=3 is +2  ->  y = -2 (change = -y * d)
    res = solve_problem(scalar_problem("minimize", "cp.square(x - 2)", "x == 3"))
    assert res.duals["x == 3"] == pytest.approx(-2.0, abs=1e-4)


def test_equality_dual_sensitivity_maximize():
    # p*(c) = -(c - 2)^2, dp*/dc at c=3 is -2  ->  y = -2 (change = +y * d)
    res = solve_problem(scalar_problem("maximize", "-cp.square(x - 2)", "x == 3"))
    assert res.duals["x == 3"] == pytest.approx(-2.0, abs=1e-4)


# --- DCP and safety ------------------------------------------------------


def test_check_problem_reports_dcp():
    info = check_problem(mean_variance(1.0))
    assert info.ok and info.is_dcp and info.objective_is_dcp
    assert info.non_dcp_constraints == []


def test_non_dcp_is_rejected_and_explained():
    spec = mean_variance(1.0)
    spec.sense = "minimize"  # minimizing a concave objective
    res = solve_problem(spec)
    assert res.status == "not_dcp"
    assert "objective" in res.error


def test_non_dcp_constraint_is_named():
    spec = scalar_problem("minimize", "x", "cp.square(x) >= 1")
    assert check_problem(spec).non_dcp_constraints == ["cp.square(x) >= 1"]


@pytest.mark.parametrize(
    "bad",
    ["().__class__", "__import__('os')", "lambda: 1", "[i for i in range(3)]", "0 <= w <= 1"],
)
def test_unsafe_or_invalid_expressions_are_blocked(bad):
    spec = mean_variance(1.0)
    spec.constraints = [bad]
    assert solve_problem(spec).status == "build_error"


def test_unknown_name_points_to_parameter_tools():
    res = solve_problem(mean_variance(1.0, inline=False))
    assert res.status == "build_error"
    assert "list_parameters" in res.error


# --- solver list ---------------------------------------------------------


def test_list_solvers_returns_one_object():
    out = list_solvers()
    assert isinstance(out.solvers, list) and out.solvers


# --- parameter store -----------------------------------------------------


def test_stored_parameters_are_used_by_name():
    set_parameter("mu", MU)
    info = set_parameter("C", C)
    assert info.shape == [5, 5] and info.symmetric and info.min_eigenvalue > 0

    w_star, _, _ = closed_form(1.0)
    res = solve_problem(mean_variance(1.0, inline=False))
    assert res.status == "optimal"
    np.testing.assert_allclose(res.variables["w"], w_star, atol=1e-5)
    assert {p.name for p in list_parameters().parameters} == {"mu", "C"}


def test_inline_parameters_override_stored():
    set_parameter("mu", [0.0] * 5)
    w_star, _, _ = closed_form(1.0)
    res = solve_problem(mean_variance(1.0))  # inline MU wins
    np.testing.assert_allclose(res.variables["w"], w_star, atol=1e-5)


def test_clear_parameters():
    set_parameter("a", 1)
    set_parameter("b", [1, 2])
    assert clear_parameters(["a", "missing"]).cleared == ["a"]
    assert [p.name for p in list_parameters().parameters] == ["b"]


@pytest.mark.parametrize("bad", ["cp", "_x", "1abc", "a-b"])
def test_bad_parameter_names_rejected(bad):
    with pytest.raises(ValueError):
        set_parameter(bad, 1)


def test_non_finite_values_rejected():
    with pytest.raises(ValueError):
        set_parameter("v", [1.0, float("nan")])


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVEX_OPT_MCP_DATA_DIRS", str(tmp_path))
    return tmp_path


def test_load_npy_csv_npz_json(data_dir):
    np.save(data_dir / "C.npy", np.array(C))
    np.savetxt(data_dir / "mu.csv", np.array(MU), delimiter=",", header="mu", comments="")
    np.savez(data_dir / "both.npz", mu=np.array(MU), C=np.array(C))
    (data_dir / "mu.json").write_text(json.dumps({"mu": MU}))

    assert load_parameter("C", str(data_dir / "C.npy")).shape == [5, 5]
    assert load_parameter("mu", str(data_dir / "mu.csv"), skip_header=1).shape == [5]
    assert load_parameter("C2", str(data_dir / "both.npz"), key="C").shape == [5, 5]
    assert load_parameter("mu2", str(data_dir / "mu.json"), key="mu").shape == [5]

    w_star, _, _ = closed_form(1.0)
    res = solve_problem(mean_variance(1.0, inline=False))
    np.testing.assert_allclose(res.variables["w"], w_star, atol=1e-5)


def test_csv_header_gives_helpful_error(data_dir):
    np.savetxt(data_dir / "mu.csv", np.array(MU), delimiter=",", header="mu", comments="")
    with pytest.raises(ValueError, match="skip_header"):
        load_parameter("mu", str(data_dir / "mu.csv"))


def test_npz_with_several_arrays_needs_key(data_dir):
    np.savez(data_dir / "both.npz", mu=np.array(MU), C=np.array(C))
    with pytest.raises(ValueError, match="key"):
        load_parameter("x", str(data_dir / "both.npz"))


def test_paths_outside_data_dirs_are_refused(data_dir, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "C.npy"
    np.save(outside, np.array(C))
    with pytest.raises(ValueError, match="outside the allowed"):
        load_parameter("C", str(outside))


def test_symlink_escape_is_refused(data_dir, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "C.npy"
    np.save(outside, np.array(C))
    link = data_dir / "link.npy"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="outside the allowed"):
        load_parameter("C", str(link))


# --- docstring examples solve as advertised --------------------------------


def test_lp_example():
    res = solve_problem(
        ProblemSpec(
            variables=[{"name": "x", "shape": [2], "nonneg": True}],
            sense="maximize",
            objective="c @ x",
            constraints=["A @ x <= b"],
            parameters={"c": [3, 5], "A": [[1, 0], [0, 2], [3, 2]], "b": [4, 12, 18]},
        )
    )
    assert res.optimal_value == pytest.approx(36, abs=1e-5)
    np.testing.assert_allclose(res.variables["x"], [2, 6], atol=1e-5)


def test_sdp_example_gives_smallest_eigenvalue():
    Cm = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
    res = solve_problem(
        ProblemSpec(
            variables=[{"name": "X", "shape": [3, 3], "PSD": True}],
            sense="minimize",
            objective="cp.trace(C @ X)",
            constraints=["cp.trace(X) == 1"],
            parameters={"C": Cm},
        )
    )
    assert res.optimal_value == pytest.approx(np.linalg.eigvalsh(Cm)[0], abs=1e-4)


def test_integer_example():
    if "HIGHS" not in list_solvers().solvers:
        pytest.skip("HIGHS not installed")
    res = solve_problem(
        ProblemSpec(
            variables=[{"name": "z", "shape": [3], "boolean": True}],
            sense="maximize",
            objective="v @ z",
            constraints=["wt @ z <= 10"],
            parameters={"v": [6, 10, 12], "wt": [1, 5, 6]},
        ),
        solver="HIGHS",
    )
    assert res.optimal_value == pytest.approx(18)
    assert res.variables["z"] == pytest.approx([1, 0, 1])


def test_strict_inequality_is_a_build_error():
    res = solve_problem(scalar_problem("minimize", "cp.square(x - 2)", "x < 1"))
    assert res.status == "build_error"


# --- portable schemas ------------------------------------------------------


def _schema_problems(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("$ref", "$defs"):
                yield f"{path}.{k}"
            if k == "type" and v == "null":
                yield f"{path}.type=null"
            yield from _schema_problems(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _schema_problems(v, f"{path}[{i}]")


def _field(tool, *names):
    for n in names:
        if hasattr(tool, n):
            return getattr(tool, n)
    return None


def test_served_schemas_are_portable():
    tools = asyncio.run(server.mcp.list_tools())
    assert {t.name for t in tools} >= {"solve_problem", "check_problem", "set_parameter", "load_parameter"}
    for t in tools:
        in_schema = _field(t, "input_schema", "inputSchema")
        out_schema = _field(t, "output_schema", "outputSchema") or {}
        assert list(_schema_problems(in_schema)) == [], t.name
        assert [p for p in _schema_problems(out_schema) if "$" in p] == [], t.name


def test_numeric_fields_are_typed_in_schema():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    value = _field(tools["set_parameter"], "input_schema", "inputSchema")["properties"]["value"]
    assert [b.get("type") for b in value["anyOf"]] == ["number", "array", "array"]


def test_portable_schema_inlines_refs_and_drops_nulls():
    schema = {
        "$defs": {"Inner": {"type": "object", "properties": {"a": {"type": "integer"}}}},
        "type": "object",
        "properties": {
            "inner": {"$ref": "#/$defs/Inner", "description": "kept"},
            "opt": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "multi": {"anyOf": [{"type": "number"}, {"type": "array"}, {"type": "null"}]},
        },
    }
    out = portable_schema(schema, for_input=True)
    assert "$defs" not in out
    assert out["properties"]["inner"] == {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "description": "kept",
    }
    assert out["properties"]["opt"] == {"type": "string"}
    assert out["properties"]["multi"] == {"anyOf": [{"type": "number"}, {"type": "array"}]}
    # outputs keep their nulls (results can be null) but still lose refs
    out2 = portable_schema(schema, for_input=False)
    assert "$defs" not in out2 and out2["properties"]["opt"]["anyOf"][1] == {"type": "null"}


@pytest.mark.parametrize("bad", ["abc", [[[1.0]]], {"a": 1}])
def test_inline_parameters_must_be_numbers_vectors_or_matrices(bad):
    with pytest.raises(ValidationError):
        ProblemSpec(variables=[], sense="minimize", objective="0", parameters={"A": bad})


# --- command line / transports ------------------------------------------------


def test_cli_defaults_to_stdio():
    args = server._parse_args([])
    assert (args.transport, args.host, args.port) == ("stdio", "127.0.0.1", 8000)


def test_cli_http_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(server.mcp, "run", lambda *a, **k: calls.append((a, k)))
    server.main(["--transport", "http", "--port", "9123"])
    (args, kwargs), = calls
    assert args == ("streamable-http",)
    if server._SDK_V2:
        assert kwargs == {"host": "127.0.0.1", "port": 9123}
    else:
        assert (server.mcp.settings.host, server.mcp.settings.port) == ("127.0.0.1", 9123)


def test_cli_stdio_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(server.mcp, "run", lambda *a, **k: calls.append((a, k)))
    server.main([])
    assert calls == [((), {})]


# --- server diagnostics on every result ------------------------------------

import os

import convex_opt_mcp


def _all_replies(data_dir):
    np.save(data_dir / "v.npy", np.array([1.0, 2.0]))
    spec = scalar_problem("minimize", "cp.square(x - 2)", "x <= 1")
    return {
        "list_solvers": list_solvers(),
        "set_parameter": set_parameter("a", 1.0),
        "load_parameter": load_parameter("v", str(data_dir / "v.npy")),
        "list_parameters": list_parameters(),
        "check_problem": check_problem(spec),
        "solve_problem": solve_problem(spec),
        "clear_parameters": clear_parameters(),
    }


def test_every_tool_reports_the_server_process(data_dir):
    for name, reply in _all_replies(data_dir).items():
        info = reply.server
        assert info.pid == os.getpid(), name
        assert info.version == convex_opt_mcp.__version__, name
        assert info.uptime_s >= 0, name
        assert info.started_at, name


def test_server_block_tracks_the_store():
    assert set_parameter("a", 1.0).server.stored_parameters == ["a"]
    assert set_parameter("b", [1.0, 2.0]).server.stored_parameters == ["a", "b"]
    assert clear_parameters(["a"]).server.stored_parameters == ["b"]
    assert list_parameters().server.stored_parameters == ["b"]


def test_started_at_is_fixed_and_uptime_grows():
    first, second = list_solvers().server, list_parameters().server
    assert first.started_at == second.started_at
    assert second.uptime_s >= first.uptime_s


def test_stored_parameter_result_keeps_parameter_fields():
    out = set_parameter("C", C)
    assert out.name == "C" and out.shape == [5, 5] and out.symmetric
    assert out.server.stored_parameters == ["C"]


def test_solve_reports_which_parameters_came_from_store():
    set_parameter("mu", MU)
    set_parameter("C", C)
    res = solve_problem(mean_variance(1.0, inline=False))  # gamma inline, mu and C stored
    assert res.parameters_from_store == ["C", "mu"]
    res = solve_problem(mean_variance(1.0))  # everything inline: nothing from store
    assert res.parameters_from_store == []
    assert check_problem(mean_variance(1.0, inline=False)).parameters_from_store == ["C", "mu"]


def test_not_dcp_result_still_reports_store_usage():
    set_parameter("C", C)
    spec = mean_variance(1.0)
    spec.parameters.pop("C")
    spec.sense = "minimize"
    res = solve_problem(spec)
    assert res.status == "not_dcp" and res.parameters_from_store == ["C"]


# --- file references in `parameters` ------------------------------------------


def test_file_reference_parameters_solve(data_dir):
    np.save(data_dir / "C.npy", np.array(C))
    np.savetxt(data_dir / "mu.csv", np.array(MU), delimiter=",", header="mu", comments="")
    spec = mean_variance(1.0, inline=False)
    spec.parameters.update(
        {"C": {"file": str(data_dir / "C.npy")}, "mu": {"file": str(data_dir / "mu.csv"), "skip_header": 1}}
    )
    res = solve_problem(spec)

    w_star, obj_star, _ = closed_form(1.0)
    assert res.status == "optimal"
    np.testing.assert_allclose(res.variables["w"], w_star, atol=1e-5)
    assert set(res.parameters_from_files) == {"C", "mu"}
    assert res.parameters_from_files["C"].endswith("C.npy")
    assert res.parameters_from_store == []
    assert res.server.stored_parameters == []  # nothing was put in the store


def test_file_reference_takes_precedence_over_store(data_dir):
    set_parameter("C", np.eye(5).tolist())  # wrong data in the store
    np.save(data_dir / "C.npy", np.array(C))
    spec = mean_variance(1.0)
    spec.parameters["C"] = {"file": str(data_dir / "C.npy")}
    res = solve_problem(spec)
    w_star, _, _ = closed_form(1.0)
    np.testing.assert_allclose(res.variables["w"], w_star, atol=1e-5)
    assert res.parameters_from_store == []


def test_file_reference_is_read_fresh_on_every_call(data_dir):
    path = data_dir / "b.json"
    spec = ProblemSpec(
        variables=[{"name": "x", "shape": []}],
        sense="minimize",
        objective="x",
        constraints=["x >= b"],
        parameters={"b": {"file": str(path), "key": "b"}},
    )
    path.write_text(json.dumps({"b": 3.0}))
    assert solve_problem(spec).optimal_value == pytest.approx(3.0, abs=1e-6)
    path.write_text(json.dumps({"b": 7.0}))
    assert solve_problem(spec).optimal_value == pytest.approx(7.0, abs=1e-6)


def test_file_reference_npz_key(data_dir):
    np.savez(data_dir / "both.npz", mu=np.array(MU), C=np.array(C))
    spec = mean_variance(1.0)
    spec.parameters["C"] = {"file": str(data_dir / "both.npz"), "key": "C"}
    res = check_problem(spec)
    assert res.ok and res.parameters_from_files["C"].endswith("both.npz [C]")


def test_file_reference_outside_data_dirs_is_a_build_error(data_dir, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "C.npy"
    np.save(outside, np.array(C))
    spec = mean_variance(1.0)
    spec.parameters["C"] = {"file": str(outside)}
    res = solve_problem(spec)
    assert res.status == "build_error" and "outside the allowed" in res.error


def test_missing_file_reference_is_a_build_error(data_dir):
    spec = mean_variance(1.0)
    spec.parameters["C"] = {"file": str(data_dir / "nope.npy")}
    res = solve_problem(spec)
    assert res.status == "build_error" and "No such file" in res.error


def test_file_reference_with_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        ProblemSpec(variables=[], sense="minimize", objective="0", parameters={"C": {"fiel": "C.npy"}})


def test_schema_offers_file_references():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    schema = _field(tools["solve_problem"], "input_schema", "inputSchema")
    branches = schema["properties"]["problem"]["properties"]["parameters"]["additionalProperties"]["anyOf"]
    file_branch = [b for b in branches if b.get("type") == "object"]
    assert file_branch and file_branch[0]["required"] == ["file"]
