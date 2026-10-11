from __future__ import annotations

import itertools
import operator
from types import SimpleNamespace

import pytest
import torch
from torch.utils._pytree import tree_unflatten

from megabake import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import (
    Accumulator,
    Binary,
    Cast,
    Constant,
    Load,
    ScalarExpression,
    TensorProgram,
    Unary,
    Where,
    dump_lowering_graph,
    lower_semantics,
)


def _context(graph: torch.fx.GraphModule, *values: torch.Tensor) -> SimpleNamespace:
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    return SimpleNamespace(
        input_order=[node.name for node in placeholders],
        input_specs=[
            SimpleNamespace(
                name=node.name,
                kind="runtime",
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                stride=tuple(value.stride()),
            )
            for node, value in zip(placeholders, values)
        ],
    )


def _import(model: torch.nn.Module, *values: torch.Tensor):
    exported = torch.export.export(model, values)
    prepared = prepare_fx_graph(
        exported.graph_module,
        values,
        output_tree_spec=exported.call_spec.out_spec,
    )
    return import_fx(prepared.graph, prepared.context)


def _dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name.removeprefix("torch."))
    assert isinstance(dtype, torch.dtype)
    return dtype


def _read_load(program: TensorProgram, load_id: str, coordinates: dict[str, int], values: dict):
    load = next(item for item in program.loads if item.id == load_id)
    offset = 0
    for axis, stride in zip(load.axes, load.strides):
        coordinate = coordinates[axis] if isinstance(axis, str) else axis
        offset += coordinate * stride
    source = values[load.value_id]
    return source.as_strided((1,), (1,), source.storage_offset() + offset)[0]


def _loads(
    program: TensorProgram, coordinates: dict[str, int], values: dict
) -> dict[str, torch.Tensor]:
    return {
        load.id: _read_load(program, load.id, coordinates, values)
        for load in program.loads
        if all(
            not isinstance(axis, str) or axis in coordinates
            for axis in load.axes
        )
    }


def _evaluate_expression(
    expression: ScalarExpression,
    loads: dict[str, torch.Tensor],
    accumulator: torch.Tensor | None,
):
    if isinstance(expression, Constant):
        return expression.value
    if isinstance(expression, Load):
        return loads[expression.load_id]
    if isinstance(expression, Accumulator):
        assert accumulator is not None
        return accumulator
    if isinstance(expression, Cast):
        return torch.as_tensor(
            _evaluate_expression(expression.operand, loads, accumulator)
        ).to(_dtype(expression.dtype))
    if isinstance(expression, Unary):
        value = _evaluate_expression(expression.operand, loads, accumulator)
        functions = {
            "relu": torch.relu,
            "neg": torch.neg,
            "abs": torch.abs,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }
        return functions[expression.operator](value)
    if isinstance(expression, Binary):
        lhs = _evaluate_expression(expression.lhs, loads, accumulator)
        rhs = _evaluate_expression(expression.rhs, loads, accumulator)
        functions = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": torch.div,
            "eq": operator.eq,
            "ne": operator.ne,
            "lt": operator.lt,
            "le": operator.le,
            "gt": operator.gt,
            "ge": operator.ge,
        }
        return functions[expression.operator](lhs, rhs)
    if isinstance(expression, Where):
        return torch.where(
            _evaluate_expression(expression.condition, loads, accumulator),
            _evaluate_expression(expression.true_value, loads, accumulator),
            _evaluate_expression(expression.false_value, loads, accumulator),
        )
    raise AssertionError(f"unsupported test expression {expression!r}")


def _evaluate_program(program: TensorProgram, values: dict) -> torch.Tensor:
    reduction_axes = set(program.reduction_axes)
    output_domains = [
        domain for domain in program.iteration_domains if domain.axis not in reduction_axes
    ]
    reduction_domains = [
        domain for domain in program.iteration_domains if domain.axis in reduction_axes
    ]
    maximum_store_offset = 0
    for size, stride in zip(program.store.shape, program.store.strides):
        maximum_store_offset += max(size - 1, 0) * abs(stride)
    storage = torch.empty(maximum_store_offset + 1, dtype=_dtype(program.store.dtype))
    result = storage.as_strided(
        program.store.shape, program.store.strides, storage_offset=0
    )
    for output_coordinates in itertools.product(
        *(range(domain.size) for domain in output_domains)
    ):
        output_map = {
            domain.axis: coordinate
            for domain, coordinate in zip(output_domains, output_coordinates)
        }
        if reduction_domains:
            assert program.init is not None and program.accumulator_dtype is not None
            accumulator = torch.as_tensor(
                program.init.value, dtype=_dtype(program.accumulator_dtype)
            )
            for reduction_coordinates in itertools.product(
                *(range(domain.size) for domain in reduction_domains)
            ):
                coordinates = output_map | {
                    domain.axis: coordinate
                    for domain, coordinate in zip(reduction_domains, reduction_coordinates)
                }
                accumulator = _evaluate_expression(
                    program.combine, _loads(program, coordinates, values), accumulator
                )
            assert program.post is not None
            value = _evaluate_expression(
                program.post, _loads(program, output_map, values), accumulator
            )
        else:
            value = _evaluate_expression(
                program.combine, _loads(program, output_map, values), None
            )
        result[output_coordinates] = value
    return result


def _evaluate_graph(graph, values: dict):
    environment = {
        value_id: values[graph.values[value_id].binding_name or graph.values[value_id].name]
        for value_id in graph.inputs
    }
    for operation in graph.ops.values():
        assert operation.tensor_program is not None, operation.numerical_rules
        assert len(operation.results) == 1
        environment[operation.results[0]] = _evaluate_program(
            operation.tensor_program, environment
        )
    outputs = [environment[value_id] for value_id in graph.outputs]
    return tree_unflatten(outputs, graph.output_tree_spec)


def test_lower_semantics_gemm_elementwise_reduction_fixture():
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.relu(torch.nn.functional.linear(x, weight, bias)).sum(dim=-1)

    torch.manual_seed(7)
    x = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    model = Model().eval()
    graph = lower_semantics(_import(model, x, weight, bias))

    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    assert [operation.tensor_program.family for operation in graph.ops.values()] == [
        "view",
        "contraction",
        "elementwise",
        "reduction",
    ]
    contraction = graph.ops["op:addmm"].tensor_program
    assert contraction is not None
    assert contraction.reduction_axes == ("r0",)
    assert contraction.accumulator_dtype == "torch.float32"
    assert contraction.post is not None
    assert graph.ops["op:permute"].tensor_program.store.owns_storage is False

    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x, "weight": weight, "bias": bias}),
        model(x, weight, bias),
    )


def test_dump_lowering_graph_reports_programs_and_unknown_reasons(tmp_path):
    class Known(torch.nn.Module):
        def forward(self, x):
            return x + 1

    x = torch.randn(2, 3)
    graph = lower_semantics(_import(Known(), x))
    path = dump_lowering_graph(graph, tmp_path / "nested" / "lowering_graph.txt")

    text = path.read_text(encoding="utf-8")
    assert path.exists()
    assert "# LoweringGraph" in text
    assert "values=2 operations=1 known=1 unknown=0 inputs=1 outputs=1" in text
    assert "## Operations" in text
    assert "## Tensor Programs" in text
    assert "### op:add" in text
    assert "source_target=aten.add.Tensor" in text
    assert "## Output Reconstruction" in text


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lower_semantics_preserves_low_precision_gemm_bias_rounding(dtype):
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.nn.functional.linear(x, weight, bias)

    torch.manual_seed(17)
    x = (torch.randn(32, 32) * 8).to(dtype)
    weight = (torch.randn(32, 32) * 8).to(dtype)
    bias = (torch.randn(32) * 8).to(dtype)
    model = Model().eval()
    graph = lower_semantics(_import(model, x, weight, bias))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x, "weight": weight, "bias": bias}),
        model(x, weight, bias),
        rtol=0,
        atol=0,
    )


def test_lower_semantics_supports_batched_linear_rank_changing_reshapes():
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.nn.functional.linear(x, weight, bias)

    torch.manual_seed(19)
    x = torch.randn(2, 3, 4)
    weight = torch.randn(5, 4)
    bias = torch.randn(5)
    model = Model().eval()
    graph = lower_semantics(_import(model, x, weight, bias))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x, "weight": weight, "bias": bias}),
        model(x, weight, bias),
    )


def test_lower_semantics_supports_zero_size_and_strided_storage_holes():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    torch.manual_seed(23)
    empty_x = torch.empty(0, 3)
    empty_y = torch.empty(0, 3)
    graph = lower_semantics(_import(Model(), empty_x, empty_y))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": empty_x, "y": empty_y}),
        Model()(empty_x, empty_y),
    )

    strided_x = torch.empty(6).as_strided((2,), (5,))
    strided_y = torch.randn(2)
    graph = lower_semantics(_import(Model(), strided_x, strided_y))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": strided_x, "y": strided_y}),
        Model()(strided_x, strided_y),
    )


def test_lower_semantics_cast_broadcast_transpose_and_reductions():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            casted = (x.view(2, 6).t() + y).to(torch.float16)
            return casted.mean(dim=0), casted.sum(dim=1)

    torch.manual_seed(11)
    x = torch.randn(3, 4)
    y = torch.randn(6, 1)
    model = Model().eval()
    graph = lower_semantics(_import(model, x, y))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    assert {operation.tensor_program.family for operation in graph.ops.values()} == {
        "view",
        "elementwise",
        "reduction",
    }

    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x, "y": y}),
        model(x, y),
    )


def test_lower_semantics_constants_comparisons_and_selection():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.where(x > 0.25, x + y, x * -3.0).to(torch.float16)

    torch.manual_seed(13)
    x = torch.randn(2, 3)
    y = torch.randn(1, 3)
    model = Model().eval()
    graph = lower_semantics(_import(model, x, y))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x, "y": y}),
        model(x, y),
    )


def test_lower_semantics_supports_all_tensor_sum_and_mean():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.sum(), x.mean()

    torch.manual_seed(29)
    x = torch.randn(2, 3)
    model = Model().eval()
    graph = lower_semantics(_import(model, x))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x}),
        model(x),
    )


def test_lower_semantics_leaves_unknown_operation_opaque_and_blocks_effects():
    def unknown_operator(value):
        return value * 2

    class Model(torch.nn.Module):
        def forward(self, x):
            return unknown_operator(x)

    value = torch.randn(2, 3)
    graph_module = torch.fx.symbolic_trace(Model())
    unknown_node = next(node for node in graph_module.graph.nodes if node.op == "call_function")
    unknown_node.meta["val"] = value
    graph = import_fx(graph_module, _context(graph_module, value))
    lowered = lower_semantics(graph)
    operation = next(iter(lowered.ops.values()))
    assert operation.tensor_program is None
    assert operation.numerical_rules["known"] is False
    assert "missing lowering rule" in operation.numerical_rules["reason"]
    assert "unsupported effects" in operation.numerical_rules["reason"]


def test_lower_semantics_turns_rule_failure_into_an_opaque_reason():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float16)

    x = torch.randn(2, 3)
    graph = _import(Model(), x)
    cast_node = graph.source_nodes["fx:convert_element_type"]
    cast_node.args = ()

    lowered = lower_semantics(graph)
    operation = next(iter(lowered.ops.values()))
    assert operation.tensor_program is None
    assert operation.numerical_rules["known"] is False
    assert "lowering rule failed" in operation.numerical_rules["reason"]
    assert "IndexError" in operation.numerical_rules["reason"]


def test_lower_semantics_reports_common_unsupported_fx_decompositions():
    class Model(torch.nn.Module):
        def forward(self, x):
            return (
                torch.softmax(x, dim=-1),
                torch.clamp(x, -1, 1),
                torch.nn.functional.gelu(x),
            )

    x = torch.randn(2, 3)
    graph = lower_semantics(_import(Model(), x))
    unknown_targets = {
        operation.attributes["target"]
        for operation in graph.ops.values()
        if operation.tensor_program is None
    }
    assert {
        "aten.amax.default",
        "aten.exp.default",
        "aten.clamp_min.default",
        "aten.clamp_max.default",
        "aten.erf.default",
    }.issubset(unknown_targets)
    assert any(operation.tensor_program is not None for operation in graph.ops.values())


def test_lower_semantics_reports_higher_order_operator_without_crashing():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cond(
                x.sum() > 0,
                lambda: x + 1,
                lambda: x * 2,
                [],
            )

    x = torch.randn(2, 3)
    graph = lower_semantics(_import(Model(), x))
    cond_operation = graph.ops["op:cond"]
    assert cond_operation.tensor_program is None
    assert cond_operation.numerical_rules["known"] is False
    assert "missing lowering rule" in cond_operation.numerical_rules["reason"]
    assert "unsupported effects" in cond_operation.numerical_rules["reason"]


def test_import_fx_rejects_symbolic_dimensions_instead_of_specializing_them():
    from torch.export import Dim

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    batch = Dim("batch")
    features = Dim("features")
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    exported = torch.export.export(
        Model(),
        (x, y),
        dynamic_shapes=({0: batch, 1: features}, {0: batch, 1: features}),
    )
    with pytest.raises(ValueError, match="Symbolic tensor shape"):
        prepare_fx_graph(
            exported.graph_module,
            (x, y),
            output_tree_spec=exported.call_spec.out_spec,
        )
