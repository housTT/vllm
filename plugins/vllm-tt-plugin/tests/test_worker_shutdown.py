# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only lifecycle checks; compiling source avoids importing TTNN devices."""

import ast
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN = Path(__file__).resolve().parents[1]
VLLM = PLUGIN.parents[1]


def _lifecycle(events):
    namespace = {
        "suppress": suppress,
        "get_tt_config": lambda config: config,
        "close_mesh_device": lambda mesh, config: events.append(("close", mesh)),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    sources = [
        (VLLM / "vllm/v1/worker/worker_base.py", "WorkerBase", ["shutdown"]),
        (VLLM / "vllm/v1/worker/worker_base.py", "WorkerWrapperBase", ["shutdown"]),
        (PLUGIN / "src/vllm_tt_plugin/worker.py", "TTWorker", ["shutdown", "__del__"]),
    ]
    for path, name, methods in sources:
        original = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == name
        )
        original.body = [
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
        if name != "TTWorker":
            original.bases = []
        module = ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
    return namespace["TTWorker"], namespace["WorkerWrapperBase"]


def test_explicit_vllm_worker_shutdown_closes_before_finalization():
    events = []
    worker_type, wrapper_type = _lifecycle(events)
    worker = worker_type()
    mesh = object()
    worker.mesh_device = mesh
    worker.vllm_config = {}
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(teardown=lambda: events.append(("teardown", mesh)))
    )
    wrapper = wrapper_type()
    wrapper.worker = worker
    wrapper.shutdown()
    assert events == [("teardown", mesh), ("close", mesh)]
    wrapper.shutdown()
    worker.__del__()
    assert events == [("teardown", mesh), ("close", mesh)]


def test_partial_worker_initialization_can_shutdown_repeatedly():
    events = []
    worker_type, _ = _lifecycle(events)
    worker = worker_type()
    worker.shutdown()
    worker.shutdown()
    worker.__del__()
    assert events == []


def test_model_teardown_failure_still_closes_mesh():
    events = []
    worker_type, _ = _lifecycle(events)
    worker = worker_type()
    mesh = object()
    worker.mesh_device = mesh
    worker.vllm_config = {}

    def fail():
        raise RuntimeError("model teardown failed")

    worker.model_runner = SimpleNamespace(model=SimpleNamespace(teardown=fail))
    with pytest.raises(RuntimeError, match="model teardown failed"):
        worker.shutdown()
    assert events == [("close", mesh)]
    worker.__del__()
    assert events == [("close", mesh)]


def test_close_helper_closes_submeshes_then_parent_without_profiling():
    source = PLUGIN / "src/vllm_tt_plugin/worker.py"
    helper = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "close_mesh_device"
    )
    events = []
    mesh = SimpleNamespace(get_num_devices=lambda: 4, get_submeshes=lambda: ["submesh"])
    namespace = {
        "ttnn": SimpleNamespace(
            close_mesh_device=lambda device: events.append(("close", device))
        ),
        "reset_fabric": lambda config, count: events.append(("reset", count)),
    }
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    namespace["close_mesh_device"](mesh, {})
    assert events == [("close", "submesh"), ("close", mesh), ("reset", 4)]
